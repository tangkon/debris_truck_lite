#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_test.py —— 开发机/测试用【全速】推理脚本（不锁核、不降优先级）
=====================================================================

它和 detect.py 的关系，一句话：

    detect.py      = 上工控机的那份，锁 2 核 + BELOW_NORMAL 优先级，不干扰换电服务
    detect_test.py = 你在自己电脑上跑的那份，吃满所有核，尽快把结果铺出来看

★ 精度相关的一切都不是第二份实现。本文件 `import detect`，切片几何、
  letterbox、ONNX 解码、全局 NMS、mm 口径过滤、grid_signature 校验，全部
  直接复用 detect.py 里的同一份代码（FastDetector 只改了线程/进程/session
  选项这类【不影响数值结果】的东西）。所以这里看到几个框、框在哪、多大，
  就是工控机上会得到的结果，区别只有速度。

为什么不做 batch 前向：
    train.py export 出来的是静态 [1,3,1280,1280]，batch 维写死为 1，物理上
    没法把 16 片一次塞进去。就算改成 dynamic 导出，CPU 上动态 shape 反而更慢
    （无法做图优化），而且批量分支要再写一份"letterbox→切片→整帧"的坐标映射
    ——那正是本项目最忌讳的两份实现悄悄漂移。所以提速只有两条正路：
      1) 单帧用满所有核（ORT intra_op = 核数）
      2) 多帧用多进程并发（--workers），每条进程各自吃一部分核
    这两条都不碰数值路径。

用法：
    # 最快的自检 / 标定（和 detect.py 完全一样的输出）
    python detect_test.py --selftest
    python detect_test.py --calibrate --image frame.jpg

    # 单张，立刻看结果
    python detect_test.py --model deploy/best.onnx --image frame.jpg --show

    # 一个目录，全速跑完 + 画框图 + HTML 报告
    python detect_test.py --model deploy/best.onnx --dir D:/frames ^
                          --out D:/results --save-img --report D:/results/index.html

    # A/B：同一批图，2 核配置 vs 全速配置，逐个框比对，证明限核不掉精度
    python detect_test.py --model deploy/best.onnx --dir D:/frames --compare

⚠ 不要把本脚本拷到工控机上运行。它会吃满全部核心，抢走换电服务的 CPU。
  工控机上只用 detect.py。
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ==========================================================================
# Windows 控制台默认 GBK，中文会乱码；与 detect.py 用同一套加固写法
# ==========================================================================
# 坑：新的 TextIOWrapper 顶替 sys.stdout 后，旧 wrapper 失去引用被 GC，而 GC 会
# 连带关闭共享的底层 BufferedWriter → 之后所有输出被静默吞掉（进程退出码非 0 但
# 屏幕上什么都没有）。而本文件又要 import detect，那段代码会执行第二次，更容易踩。
# 所以：已经是 UTF-8 就别动、旧对象存模块级变量防 GC、开行缓冲。
if sys.platform == "win32":
    def _enc_is_utf8(stream) -> bool:
        e = (getattr(stream, "encoding", "") or "").lower().replace("-", "").replace("_", "")
        return e in ("utf8", "utf8mb4")

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:                       # Python < 3.7 没有 reconfigure
        if not (_enc_is_utf8(sys.stdout) and _enc_is_utf8(sys.stderr)):
            import io
            _OLD_STDOUT, _OLD_STDERR = sys.stdout, sys.stderr      # 防 GC，别删
            if not _enc_is_utf8(sys.stdout):
                sys.stdout = io.TextIOWrapper(_OLD_STDOUT.buffer, encoding="utf-8",
                                              errors="replace", line_buffering=True)
            if not _enc_is_utf8(sys.stderr):
                sys.stderr = io.TextIOWrapper(_OLD_STDERR.buffer, encoding="utf-8",
                                              errors="replace", line_buffering=True)
    except Exception:                            # noqa: BLE001
        pass


# ==========================================================================
# ★ 必须在 import detect 之前解开 BLAS/OMP 限流
# ==========================================================================
# detect.py 在模块顶层用 os.environ.setdefault(...) 把 OMP/MKL/OpenBLAS 等线程数
# 压到 2，而且它必须这么做（那些库在 import 时就读取环境变量，事后再改无效）。
# setdefault 的语义是"已有值就不覆盖"，所以这里用【直接赋值】抢先把它们设成全部
# 核数，detect.py 随后执行到的 setdefault 就变成空操作。
# 顺序反了的话，本脚本会静默地仍然只有 2 个 BLAS 线程，你只会觉得"怎么没变快"。
CPU_COUNT = os.cpu_count() or 1
_FULL = str(CPU_COUNT)
for _v in ("FOD_THREADS",                       # detect.py 用它决定默认 intra_op
           "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _FULL

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import detect as base                        # ← 复用同一份推理实现
except ImportError as _e:                        # pragma: no cover
    print("[FATAL] 无法 import detect.py：%s" % _e)
    print("        detect_test.py 必须和 detect.py 放在同一个目录里。")
    raise SystemExit(1)

BASE_PY = ROOT / "detect.py"


# ==========================================================================
# 线程/进程规划
# ==========================================================================
def plan(workers_arg: int, threads_arg: Optional[int], n_images: int,
         cpu: int = CPU_COUNT) -> Dict[str, Any]:
    """决定用几个进程、每个进程给 ORT 几条线程。

    原则：workers x intra ≈ cpu，不要超订（超订只会互相抢核，总吞吐反而降）。
    YOLO 卷积在 1280 输入上超过 4 线程后加速比就明显衰减，所以核多时
    【多开进程】比【单进程堆线程】更划算。
    """
    if workers_arg and workers_arg > 0:
        workers = min(int(workers_arg), max(1, n_images))
    else:                                        # 0 = 自动
        if n_images <= 1:
            workers = 1
        else:
            workers = max(1, min(cpu // 2, 8, n_images))

    if threads_arg and threads_arg > 0:
        intra = int(threads_arg)
    elif workers > 1:
        intra = max(1, cpu // workers)
    else:
        intra = cpu

    total = workers * intra
    notes = []
    if total > cpu:
        notes.append("workers(%d) x intra(%d) = %d > 本机 %d 核，会超订；"
                     "总吞吐可能不升反降" % (workers, intra, total, cpu))
    return {"workers": workers, "intra": intra, "cpu": cpu,
            "total_threads": total, "oversubscribed": total > cpu,
            "notes": notes}


# ==========================================================================
# FastDetector：只改"跑多快"，不改"跑出什么"
# ==========================================================================
class FastDetector(base.Detector):
    """detect.Detector 的全速版。

    刻意【不重写】的方法（全部继承，保证与工控机行为逐字一致）：
        _check_signature / _letterbox / _to_blob / _infer / _decode /
        _infer_slice / _collect_candidates / detect / detect_file /
        grid_summary / grid_for
    只重写：
        _tune_session  —— 追加 ORT_PARALLEL / 关闭 memory pattern 复用等
        _print_banner  —— 换成测试脚本的横幅
        status         —— 多写几个只有测试端才关心的字段
    """

    def __init__(self,
                 model_path: str,
                 conf_thr: Optional[float] = None,
                 iou_thr: Optional[float] = None,
                 mm_per_px_override: Optional[float] = None,
                 max_det: Optional[int] = None,
                 intra_threads: Optional[int] = None,
                 inter_threads: int = 1,
                 tta: bool = False,
                 allow_grid_mismatch: bool = False,
                 parallel: bool = False,
                 spin: bool = True,
                 workers: int = 1,
                 verbose: bool = True):
        self.parallel = bool(parallel)
        self.spin = bool(spin)
        self.workers = int(workers)
        # 顺序陷阱：base.__init__ 的最后一步会回调 self._print_banner()，那时候
        # 下面这两个属性还没赋值，直接用会 AttributeError。先给默认值兜底，
        # 横幅里改成现算（_probe_shape），super() 返回后再刷新成真实值。
        self.input_shape = []                    # type: List[Any]
        self.batch_dim_static1 = False
        # 不设亲和性、不降优先级 —— 这就是本脚本存在的全部理由。
        # 注意 base.Detector 自己也不会做这两件事（那是 detect.py main() 干的），
        # 所以这里什么都不用"取消"，只要不去调用 confine_to_cores 就行。
        super(FastDetector, self).__init__(
            model_path, conf_thr=conf_thr, iou_thr=iou_thr,
            mm_per_px_override=mm_per_px_override, max_det=max_det,
            intra_threads=intra_threads, inter_threads=inter_threads,
            tta=tta, allow_grid_mismatch=allow_grid_mismatch, verbose=verbose)

        # 导出时的 batch 维是静态 1 还是动态，决定还有没有批量前向的余地
        self.input_shape, self.batch_dim_static1 = self._probe_shape()

    # ------------------------------------------------------------------
    def _probe_shape(self) -> Tuple[List[Any], bool]:
        """读出 ONNX 输入张量的 shape，并判断 batch 维是不是静态 1。"""
        try:
            shape = list(self.sess.get_inputs()[0].shape)
        except Exception:                        # noqa: BLE001
            shape = []                           # 拿不到就当未知，不影响推理
        return shape, bool(shape) and shape[0] == 1

    # ------------------------------------------------------------------
    def _tune_session(self, so, ort) -> None:
        """在 base 设好的 SessionOptions 上追加测试端的选项。

        这些都是【调度/资源】层面的开关，不改变任何算子的数学定义，
        FP32 结果与 detect.py 一致（浮点归约顺序可能因线程数不同而有
        1e-6 级抖动，位置不受影响；--compare 就是用来实测这件事的）。
        """
        if self.parallel:
            # 图级并行：让没有依赖关系的算子并发跑。需要 inter_op > 1 才有意义。
            try:
                so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            except Exception:                    # noqa: BLE001
                pass
        if self.spin:
            # ORT 默认线程干完活会自旋等待一小会儿再睡，这样下一次调用唤醒更快。
            # 工控机上我们希望它赶紧睡（别占核），测试机上正相反：宁可烧 CPU 换延迟。
            for key, val in (("session.intra_op.allow_spinning", "1"),
                             ("session.inter_op.allow_spinning", "1")):
                try:
                    so.add_session_config_entry(key, val)
                except Exception:                # noqa: BLE001
                    pass                         # 老版本 ORT 没这个键，忽略
        try:
            so.enable_mem_pattern = True         # 输入 shape 固定，内存模式可复用
        except Exception:                        # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    def _print_banner(self) -> None:
        print("[model] %s (%.1f MB)"
              % (self.model_path.name, self.model_path.stat().st_size / 1e6))
        print("[cfg] %s" % self.grid_summary())
        print("[fast] 全速模式：ORT intra_op=%d inter_op=%d  execution_mode=%s  "
              "自旋等待=%s" % (self.intra_threads, self.inter_threads,
                              "ORT_PARALLEL" if self.parallel else "ORT_SEQUENTIAL",
                              "开" if self.spin else "关"))
        print("[fast] 进程亲和性：未设置（可用全部 %d 核）  优先级：正常" % CPU_COUNT)
        if self.mm_overridden:
            print("[cfg] mm/px 已被 --mm-per-px 覆盖为 %.3f（meta 里是 %.3f）"
                  % (self.mm_per_px, float(self.meta.get("mm_per_px", 0) or 0)))
        if self.tta:
            print("[cfg] TTA=水平翻转（耗时约 x2，召回一般 +0.5~1.5pt）")
        # 这里必须现算：base.__init__ 在 self.batch_dim_static1 刷新之前就会调本方法
        _shape, static1 = self._probe_shape()
        if static1:
            print("[fast] 本 ONNX 是静态 batch=1 → 16 片只能串行前向。想再快就加 "
                  "--workers（多进程跑多张图），别去动 imgsz 或切片尺寸（会掉精度）。")

    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        st = super(FastDetector, self).status()
        st.update({
            "script": "detect_test.py",
            "mode": "fullspeed",
            "warning": "测试脚本，禁止部署到工控机（会吃满全部核心）",
            "cpu_count": CPU_COUNT,
            "affinity_set": False,
            "priority_lowered": False,
            "workers": self.workers,
            "execution_mode": "ORT_PARALLEL" if self.parallel else "ORT_SEQUENTIAL",
            "allow_spinning": self.spin,
            "onnx_input_shape": [str(d) for d in self.input_shape],
            "blas_thread_env": _FULL,
        })
        return st


# ==========================================================================
# 多进程 worker
# ==========================================================================
# Windows 上 multiprocessing 用 spawn：子进程会重新 import 本模块，于是模块顶层
# 那段"抢在 detect 之前设置环境变量"的代码在子进程里也会执行 —— 这正是我们要的。
# 所以 worker 函数和配置都必须是模块级、可 pickle 的纯数据。
_W: Dict[str, Any] = {}


def _init_worker(cfg: Dict[str, Any]) -> None:
    """每条 worker 进程建一次 session（建 session 是最贵的一步，绝不能每张图建）。"""
    try:
        import cv2
        # 多进程时 OpenCV 别再自己开线程池，否则 workers x cv2线程 x ORT线程 严重超订。
        # 解码/resize 相对 16 次 1280x1280 前向可以忽略，单线程足够。
        cv2.setNumThreads(1)
    except Exception:                            # noqa: BLE001
        pass
    det = FastDetector(verbose=False, **cfg["det_kwargs"])
    det.workers = cfg["plan"]["workers"]
    _W["det"] = det
    _W["cfg"] = cfg


def _work_one(path_str: str) -> Dict[str, Any]:
    cfg = _W["cfg"]
    det = _W["det"]
    p = Path(path_str)
    img = base.imread_u(p)
    if img is None:
        return {"path": str(p), "name": p.name, "error": "unreadable",
                "count": 0, "detections": [], "latency_ms": 0.0}
    t0 = time.time()
    dets = det.detect(img)
    ms = (time.time() - t0) * 1000.0
    rec = {
        "path": str(p),
        "name": p.name,
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "latency_ms": round(ms, 1),
        "count": len(dets),
        "detections": dets,
    }
    if cfg.get("save_img") and cfg.get("out_dir"):
        vis = base.draw(img, dets, det.mm_per_px, max_side=cfg.get("preview_side", 0))
        rec["annotated"] = str(Path(cfg["out_dir"]) / "annotated" / (p.stem + "_det.jpg"))
        base.imwrite_u(Path(rec["annotated"]), vis)
    return rec


# ==========================================================================
# 收集输入
# ==========================================================================
def collect_paths(args) -> List[Path]:
    paths: List[Path] = []
    if args.image:
        for s in args.image:
            p = Path(s)
            if not p.is_absolute():
                p = ROOT / p
            paths.append(p)
    if args.dir:
        d = Path(args.dir)
        if not d.is_absolute():
            d = ROOT / d
        if not d.is_dir():
            print("[ERROR] 目录不存在: %s" % d)
            return []
        paths.extend([p for p in sorted(d.rglob("*"))
                      if p.suffix.lower() in base.IMG_EXTS])
    if args.limit and args.limit > 0:
        paths = paths[:args.limit]
    return paths


# ==========================================================================
# 跑图：串行 / 多进程
# ==========================================================================
def run_serial(det: FastDetector, paths: List[Path], args,
               out_dir: Optional[Path]) -> List[Dict[str, Any]]:
    import cv2
    results: List[Dict[str, Any]] = []
    n = len(paths)
    for i, p in enumerate(paths, 1):
        img = base.imread_u(p)
        if img is None:
            print("  [WARN] 读不了 %s（跳过）" % p)
            results.append({"path": str(p), "name": p.name, "error": "unreadable",
                            "count": 0, "detections": [], "latency_ms": 0.0})
            continue
        t0 = time.time()
        dets = det.detect(img)
        ms = (time.time() - t0) * 1000.0
        rec = {"path": str(p), "name": p.name,
               "width": int(img.shape[1]), "height": int(img.shape[0]),
               "latency_ms": round(ms, 1), "count": len(dets), "detections": dets}
        results.append(rec)
        if not args.quiet:
            print("  [%d/%d] %s  %s  n=%d  %s  %.0fms"
                  % (i, n, "有异物" if dets else "  干净", p.name, len(dets),
                     ",".join("%.0fmm" % d["size_mm"] for d in dets[:6]), ms))
        if out_dir and args.save_img:
            vis = base.draw(img, dets, det.mm_per_px, max_side=args.preview_side)
            dst = out_dir / "annotated" / (p.stem + "_det.jpg")
            base.imwrite_u(dst, vis)
            rec["annotated"] = str(dst)
        if args.show:
            vis = base.draw(img, dets, det.mm_per_px, max_side=args.preview_side or 1280)
            try:
                cv2.imshow("detect_test  (q / ESC = stop)", vis)
                k = cv2.waitKey(0 if args.show_pause else 25) & 0xFF
                if k in (ord("q"), 27):
                    print("[show] 用户中止浏览，剩余图片仍会跑完并计入统计。")
                    args.show = False
            except Exception as e:               # noqa: BLE001
                print("[show][WARN] 无法开窗口（%s），后续不再尝试。" % e)
                args.show = False
    if args.show:
        try:
            cv2.destroyAllWindows()
        except Exception:                        # noqa: BLE001
            pass
    return results


def run_parallel(det_kwargs: Dict[str, Any], pl: Dict[str, Any], paths: List[Path],
                 args, out_dir: Optional[Path]) -> Optional[List[Dict[str, Any]]]:
    """多进程跑图。返回 None 表示起不来，调用方应回退串行。"""
    try:
        import multiprocessing as mp
    except ImportError:                          # pragma: no cover
        return None
    cfg = {
        "det_kwargs": det_kwargs,
        "plan": pl,
        "save_img": bool(args.save_img),
        "out_dir": str(out_dir) if out_dir else None,
        "preview_side": int(args.preview_side),
    }
    try:
        ctx = mp.get_context("spawn" if sys.platform == "win32" else "fork")
        pool = ctx.Pool(processes=pl["workers"], initializer=_init_worker,
                        initargs=(cfg,))
    except Exception as e:                       # noqa: BLE001
        print("[workers][WARN] 进程池起不来（%s），回退单进程。" % e)
        return None

    results: List[Dict[str, Any]] = []
    n = len(paths)
    t0 = time.time()
    try:
        # imap 保序且流式：跑完一张就能立刻打印，不用等全部结束
        for i, rec in enumerate(pool.imap(_work_one, [str(p) for p in paths]), 1):
            results.append(rec)
            if not args.quiet:
                dets = rec.get("detections") or []
                print("  [%d/%d] %s  %s  n=%d  %s  %.0fms"
                      % (i, n, "有异物" if dets else "  干净", rec["name"],
                         rec.get("count", 0),
                         ",".join("%.0fmm" % d["size_mm"] for d in dets[:6]),
                         rec.get("latency_ms", 0.0)))
    except Exception as e:                       # noqa: BLE001
        print("[workers][WARN] 进程池执行失败（%s），已完成的 %d 张保留，"
              "剩余回退单进程。" % (e, len(results)))
        pool.terminate()
        try:
            pool.join()
        except Exception:                        # noqa: BLE001
            pass
        return results if results else None
    finally:
        try:
            pool.close()
            pool.join()
        except Exception:                        # noqa: BLE001
            pass
    print("[workers] %d 进程并发，墙钟 %.1f s（吞吐 %.2f 张/s）"
          % (pl["workers"], time.time() - t0, n / max(1e-9, time.time() - t0)))
    return results


# ==========================================================================
# 统计与报告
# ==========================================================================
def pct(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def summarize(results: List[Dict[str, Any]], det: FastDetector, pl: Dict[str, Any],
              args, wall_s: float) -> Dict[str, Any]:
    lat = [r["latency_ms"] for r in results if r.get("latency_ms")]
    ls = sorted(lat)
    n_ok = len(lat)
    n_obj = sum(r.get("count", 0) for r in results)
    n_hit = sum(1 for r in results if r.get("count", 0) > 0)

    print("")
    print("=" * 72)
    print("推理汇总（%d 张，其中 %d 张成功）" % (len(results), n_ok))
    print("=" * 72)
    if n_ok:
        print("  检出异物的图 : %d 张（%.1f%%）" % (n_hit, 100.0 * n_hit / n_ok))
        print("  目标总数     : %d" % n_obj)
        print("  单帧延迟     : 均值 %.0f ms | P50 %.0f | P90 %.0f | P99 %.0f | 最大 %.0f"
              % (sum(lat) / n_ok, pct(ls, 0.50), pct(ls, 0.90), pct(ls, 0.99), ls[-1]))
        print("  墙钟总耗时   : %.1f s   吞吐 %.2f 张/s" % (wall_s, len(results) / max(1e-9, wall_s)))
        print("  算力配置     : workers=%d x intra_op=%d = %d 线程（本机 %d 核）"
              % (pl["workers"], pl["intra"], pl["total_threads"], pl["cpu"]))
        print("  ORT 后端     : %s  execution_mode=%s"
              % (det.ort_provider, "ORT_PARALLEL" if det.parallel else "ORT_SEQUENTIAL"))
        # 对照：工控机上 detect.py 只有 2 个 intra 线程、单进程
        theo = max(1, pl["total_threads"]) / 2.0
        print("  对比工控机   : detect.py 是 1 进程 x 2 线程；本机是 %d 进程 x %d 线程，"
              % (pl["workers"], pl["intra"]))
        print("                 线程数约 %.1f 倍。实际加速比会低于这个数（卷积并行"
              % theo)
        print("                 有 Amdahl 上限），但【检出结果必须完全一致】——")
        print("                 想验证就跑 --compare。")
        if args.takt_ms > 0:
            p99 = pct(ls, 0.99)
            print("  节拍判定     : P99 %.0f ms vs 预算 %d ms → %s"
                  % (p99, args.takt_ms, "满足" if p99 <= args.takt_ms else "超预算"))
            print("                 注意这是本机全速的数字，不能当作工控机上的表现；")
            print("                 上线判定请以 detect.py 自己打印的延迟为准。")
    print("=" * 72)

    return {
        "images": len(results),
        "ok": n_ok,
        "images_with_object": n_hit,
        "objects": n_obj,
        "latency_ms_mean": round(sum(lat) / n_ok, 1) if n_ok else 0.0,
        "latency_ms_p50": round(pct(ls, 0.50), 1),
        "latency_ms_p90": round(pct(ls, 0.90), 1),
        "latency_ms_p99": round(pct(ls, 0.99), 1),
        "latency_ms_max": round(ls[-1], 1) if ls else 0.0,
        "wall_seconds": round(wall_s, 2),
        "throughput_img_per_s": round(len(results) / max(1e-9, wall_s), 3),
        "workers": pl["workers"],
        "intra_op_num_threads": pl["intra"],
        "total_threads": pl["total_threads"],
    }


# ==========================================================================
# HTML 报告：一屏看完一整批结果
# ==========================================================================
_HTML_TMPL = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>异物检测结果 - __TITLE__</title>
<style>
 body{font-family:"Microsoft YaHei",system-ui,sans-serif;margin:0;background:#f5f6f8;color:#222}
 header{background:#1f2937;color:#fff;padding:14px 20px}
 header h1{margin:0;font-size:18px} header .sub{font-size:12px;opacity:.75;margin-top:4px}
 .bar{padding:10px 20px;background:#fff;border-bottom:1px solid #e5e7eb;display:flex;
      gap:16px;flex-wrap:wrap;align-items:center;font-size:13px}
 .bar b{font-size:16px}
 .bar input,.bar select{padding:4px 8px;font-size:13px}
 table{width:100%;border-collapse:collapse;background:#fff;font-size:13px}
 th,td{padding:6px 10px;border-bottom:1px solid #eee;text-align:left;vertical-align:middle}
 th{background:#f9fafb;cursor:pointer;user-select:none;white-space:nowrap}
 th:hover{background:#eef2ff}
 tr.hit{background:#fff7ed} tr.clean{background:#f8fafc}
 img.thumb{height:64px;border:1px solid #ddd;border-radius:3px;display:block}
 .badge{display:inline-block;padding:1px 6px;border-radius:8px;font-size:11px;color:#fff}
 .b-hit{background:#dc2626} .b-clean{background:#16a34a}
 code{background:#f3f4f6;padding:1px 4px;border-radius:3px}
 .warn{background:#fef2f2;border-left:3px solid #dc2626;padding:8px 20px;font-size:12px}
</style></head><body>
<header><h1>电池包上盖异物检测 —— 测试报告（detect_test.py 全速模式）</h1>
<div class="sub">__SUB__</div></header>
<div class="warn">本页由 <code>detect_test.py</code> 生成，跑在开发机上、未限制 CPU。
检出结果与工控机上的 <code>detect.py</code> 完全一致（同一份推理代码），
但<b>延迟数字不可用于上线判定</b>——那要看 detect.py 自己打印的。</div>
<div class="bar">
 <span>图片 <b>__N_IMG__</b></span>
 <span>有异物 <b>__N_HIT__</b></span>
 <span>目标总数 <b>__N_OBJ__</b></span>
 <span>均值延迟 <b>__LAT__ ms</b></span>
 <span>吞吐 <b>__TPS__ 张/s</b></span>
 <span>配置 <b>__CFG__</b></span>
 <label>筛选 <input id="q" placeholder="文件名包含…"></label>
 <label>只看 <select id="f">
   <option value="all">全部</option><option value="hit">有异物</option>
   <option value="clean">干净</option></select></label>
</div>
<table id="t"><thead><tr>
 <th data-k="i">#</th><th>缩略图</th><th data-k="name">文件</th>
 <th data-k="count">检出</th><th data-k="mm">最大 mm</th>
 <th data-k="score">最高分</th><th data-k="ms">延迟 ms</th>
 <th>框（x1,y1,x2,y2 / mm / score）</th></tr></thead><tbody>
__ROWS__
</tbody></table>
<script>
var tb=document.querySelector('#t tbody');
var rows=Array.prototype.slice.call(tb.rows);
var dir={};
function apply(){
  var q=document.getElementById('q').value.toLowerCase();
  var f=document.getElementById('f').value;
  rows.forEach(function(r){
    var ok=(!q||r.dataset.name.toLowerCase().indexOf(q)>=0)
        &&(f==='all'||r.dataset.cls===f);
    r.style.display=ok?'':'none';
  });
}
document.getElementById('q').oninput=apply;
document.getElementById('f').onchange=apply;
Array.prototype.forEach.call(document.querySelectorAll('th[data-k]'),function(th){
  th.onclick=function(){
    var k=th.dataset.k; dir[k]=!dir[k];
    rows.sort(function(a,b){
      var x=a.dataset[k],y=b.dataset[k];
      var nx=parseFloat(x),ny=parseFloat(y);
      var c=(!isNaN(nx)&&!isNaN(ny))?nx-ny:String(x).localeCompare(String(y));
      return dir[k]?c:-c;
    });
    rows.forEach(function(r){tb.appendChild(r);});
  };
});
</script></body></html>
"""


def _esc(s: Any) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def write_html_report(path: Path, results: List[Dict[str, Any]],
                      det: FastDetector, pl: Dict[str, Any],
                      summ: Dict[str, Any]) -> None:
    rows = []
    for i, r in enumerate(results, 1):
        dets = r.get("detections") or []
        cls = "hit" if dets else "clean"
        thumb = ""
        ann = r.get("annotated")
        if ann:
            try:
                rel = os.path.relpath(ann, str(path.parent)).replace(os.sep, "/")
                thumb = ('<a href="%s" target="_blank"><img class="thumb" src="%s" '
                         'loading="lazy" alt=""></a>' % (_esc(rel), _esc(rel)))
            except ValueError:                   # 不同盘符，退化成绝对路径 file://
                u = "file:///" + str(ann).replace(os.sep, "/")
                thumb = '<a href="%s" target="_blank"><img class="thumb" src="%s"></a>' % (_esc(u), _esc(u))
        max_mm = max([d["size_mm"] for d in dets], default=0.0)
        max_sc = max([d["score"] for d in dets], default=0.0)
        detail = "<br>".join(
            "(%.0f,%.0f,%.0f,%.0f) %.1fmm %.3f"
            % (d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3],
               d["size_mm"], d["score"]) for d in dets[:8])
        if len(dets) > 8:
            detail += "<br>… 共 %d 个" % len(dets)
        rows.append(
            '<tr class="%s" data-i="%d" data-name="%s" data-count="%d" '
            'data-mm="%.1f" data-score="%.4f" data-ms="%.1f" data-cls="%s">'
            '<td>%d</td><td>%s</td><td>%s</td>'
            '<td><span class="badge %s">%d</span></td>'
            '<td>%.1f</td><td>%.3f</td><td>%.0f</td><td>%s</td></tr>'
            % (cls, i, _esc(r["name"]), len(dets), max_mm, max_sc,
               r.get("latency_ms", 0.0), cls,
               i, thumb, _esc(r["name"]),
               "b-hit" if dets else "b-clean", len(dets),
               max_mm, max_sc, r.get("latency_ms", 0.0), detail))

    html = (_HTML_TMPL
            .replace("__TITLE__", _esc(time.strftime("%Y-%m-%d %H:%M:%S")))
            .replace("__SUB__", _esc("模型 %s | 切片 %dx%d→%d (%.3fx) | mm/px=%.3f | "
                                     "口径 %.1fmm | conf=%.2f iou=%.2f | 生成 %s"
                                     % (det.model_path.name, det.slice_w, det.slice_h,
                                        det.imgsz, det.scale, det.mm_per_px, det.min_mm,
                                        det.conf_thr, det.iou_thr,
                                        time.strftime("%Y-%m-%d %H:%M:%S"))))
            .replace("__N_IMG__", str(summ["images"]))
            .replace("__N_HIT__", str(summ["images_with_object"]))
            .replace("__N_OBJ__", str(summ["objects"]))
            .replace("__LAT__", "%.0f" % summ["latency_ms_mean"])
            .replace("__TPS__", "%.2f" % summ["throughput_img_per_s"])
            .replace("__CFG__", _esc("%d 进程 x %d 线程 / 共 %d 核"
                                     % (pl["workers"], pl["intra"], pl["cpu"])))
            .replace("__ROWS__", "\n".join(rows)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print("[out] HTML 报告: %s" % path)
    print("      用浏览器打开即可，支持按检出数/尺寸/延迟排序和文件名筛选。")


# ==========================================================================
# --compare：2 核配置 vs 全速配置，逐个框比对
# ==========================================================================
def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = ua + ub - inter
    return inter / den if den > 1e-9 else 0.0


def diff_results(slow: Dict[str, Any], fast: Dict[str, Any],
                 tol_px: float = 0.6, tol_score: float = 2e-3) -> Dict[str, Any]:
    """按文件名对齐两套结果，逐框配对，统计位置/分数差异。

    线程数不同会让浮点归约顺序不同，score 出现 1e-6~1e-3 级抖动是正常的；
    但【框的位置】由 letterbox/NMS 的几何决定，不该有任何可见差异。
    位置对不上 = 有别的东西被改动了，必须查。
    """
    s_by = {r["name"]: r for r in slow.get("images", []) if "name" in r}
    f_by = {r["name"]: r for r in fast.get("images", []) if "name" in r}
    only_slow = sorted(set(s_by) - set(f_by))
    only_fast = sorted(set(f_by) - set(s_by))
    common = sorted(set(s_by) & set(f_by))

    n_pairs = 0
    max_dx = max_dy = 0.0
    max_ds = 0.0
    count_mismatch: List[Tuple[str, int, int]] = []
    unmatched: List[Tuple[str, str]] = []
    slow_ms: List[float] = []
    fast_ms: List[float] = []

    for name in common:
        sd = s_by[name].get("detections") or []
        fd = f_by[name].get("detections") or []
        if s_by[name].get("latency_ms"):
            slow_ms.append(float(s_by[name]["latency_ms"]))
        if f_by[name].get("latency_ms"):
            fast_ms.append(float(f_by[name]["latency_ms"]))
        if len(sd) != len(fd):
            count_mismatch.append((name, len(sd), len(fd)))
        used = set()
        for a in sd:
            best, best_iou, best_j = None, -1.0, -1
            for j, b in enumerate(fd):
                if j in used:
                    continue
                v = _iou(a["bbox"], b["bbox"])
                if v > best_iou:
                    best, best_iou, best_j = b, v, j
            if best is None or best_iou <= 0.0:
                unmatched.append((name, "只在 2 核结果里: %s" % (a["bbox"],)))
                continue
            used.add(best_j)                     # 不能用 fd.index(best)：两个完全
                                                 # 相同的框会指回同一个下标
            n_pairs += 1
            max_dx = max(max_dx, abs(a["bbox"][0] - best["bbox"][0]),
                         abs(a["bbox"][2] - best["bbox"][2]))
            max_dy = max(max_dy, abs(a["bbox"][1] - best["bbox"][1]),
                         abs(a["bbox"][3] - best["bbox"][3]))
            max_ds = max(max_ds, abs(a["score"] - best["score"]))
        for j, b in enumerate(fd):
            if j not in used:
                unmatched.append((name, "只在全速结果里: %s" % (b["bbox"],)))

    same = (not count_mismatch and not unmatched and not only_slow and not only_fast
            and max_dx <= tol_px and max_dy <= tol_px and max_ds <= tol_score)
    s_mean = sum(slow_ms) / len(slow_ms) if slow_ms else 0.0
    f_mean = sum(fast_ms) / len(fast_ms) if fast_ms else 0.0
    return {
        "identical": same,
        "compared_images": len(common),
        "paired_boxes": n_pairs,
        "only_in_slow": only_slow,
        "only_in_fast": only_fast,
        "count_mismatch": count_mismatch,
        "unmatched_boxes": unmatched,
        "max_dx_px": round(max_dx, 3),
        "max_dy_px": round(max_dy, 3),
        "max_score_delta": round(max_ds, 6),
        "tolerance": {"px": tol_px, "score": tol_score},
        "latency_ms_mean_slow_2core": round(s_mean, 1),
        "latency_ms_mean_fast": round(f_mean, 1),
        "speedup": round(s_mean / f_mean, 2) if f_mean > 0 else None,
    }


def build_slow_cmd(args, model: Path, json_path: Path,
                   cores: str, threads: int) -> List[str]:
    """把用户原始给的输入选择【原样转发】给 detect.py。

    不要在这里改成逐个 --image：几百张图会把 Windows 的 32KB 命令行上限撑爆。
    detect.py 用的是同样的 sorted(rglob) + --limit 截断规则，转发原始参数
    就能保证两边选中完全相同的一批图（这一点在 diff 里也会被 only_in_* 校验）。
    """
    cmd = [sys.executable, str(BASE_PY), "--model", str(model),
           "--cores", cores, "--threads", str(threads),
           "--warmup", str(args.warmup), "--quiet", "--json", str(json_path)]
    if args.dir:
        cmd += ["--dir", str(args.dir)]
    if args.image:
        for s in args.image:
            cmd += ["--image", str(s)]
    if args.limit and args.limit > 0:
        cmd += ["--limit", str(args.limit)]
    if args.conf is not None:
        cmd += ["--conf", str(args.conf)]
    if args.iou is not None:
        cmd += ["--iou", str(args.iou)]
    if args.max_det is not None:
        cmd += ["--max-det", str(args.max_det)]
    if args.mm_per_px is not None:
        cmd += ["--mm-per-px", str(args.mm_per_px)]
    if args.tta:
        cmd += ["--tta"]
    if args.allow_grid_mismatch:
        cmd += ["--allow-grid-mismatch"]
    return cmd


def run_compare(args, model: Path, paths: List[Path], det_kwargs: Dict[str, Any],
                pl: Dict[str, Any], out_dir: Optional[Path]) -> int:
    print("")
    print("=" * 72)
    print("--compare：同一批图，【detect.py 锁 2 核】 vs 【detect_test.py 全速】")
    print("=" * 72)
    print("要回答的问题：限核到底会不会掉精度？")
    print("理论上不会——线程数只影响浮点归约顺序（1e-6 级），不影响 letterbox、")
    print("NMS、mm 口径这些决定框位置的几何逻辑。但理论上不算数，跑一遍看。")
    print("")

    # 两份 JSON 都留在磁盘上，比对不一致时正需要拿它们对账，所以不删
    cmp_dir = (out_dir if out_dir else ROOT / "test_results") / "compare"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    slow_json = cmp_dir / "slow_2core.json"
    fast_json = cmp_dir / "fast_fullspeed.json"

    # ---- 1) 慢速：真起一个 detect.py 子进程，让它自己绑 2 核 ----
    # 用子进程而不是在本进程里再建一个 2 线程 session，是因为亲和性是【进程级】的：
    # 本进程一旦 SetProcessAffinityMask 就再也放不开（会污染后面的全速测量），
    # 而子进程退出后什么都不留。这才是干净的 A/B。
    cmd = build_slow_cmd(args, model, slow_json,
                         cores=args.compare_cores, threads=args.compare_threads)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"           # 子进程中文输出走管道时别乱码
    env["PYTHONUNBUFFERED"] = "1"
    print("[A] 跑 detect.py（--cores %s --threads %d）…"
          % (args.compare_cores, args.compare_threads))
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True, encoding="utf-8",
                              errors="replace", env=env)
    except TypeError:                            # Python 3.5 没有 encoding= 参数
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True, env=env)
    wall_slow = time.time() - t0
    if proc.returncode != 0:
        print("[A][ERROR] detect.py 退出码 %d，输出如下：" % proc.returncode)
        print((proc.stdout or "")[-3000:])
        print("[A] 命令行: %s" % " ".join(cmd))
        return 1
    if not slow_json.is_file():
        print("[A][ERROR] 没生成 %s" % slow_json)
        print((proc.stdout or "")[-2000:])
        return 1
    slow = json.loads(slow_json.read_text(encoding="utf-8"))
    print("[A] 完成，%.1f s，%d 张图 → %s"
          % (wall_slow, len(slow.get("images", [])), slow_json))

    # ---- 2) 快速：本进程全速跑同一批 ----
    print("[B] 跑 detect_test.py（%d 进程 x %d 线程）…" % (pl["workers"], pl["intra"]))
    t0 = time.time()
    det = FastDetector(verbose=True, workers=pl["workers"], **det_kwargs)
    fast_results = run_serial(det, paths, args, None)
    wall_fast = time.time() - t0
    fast = {"images": fast_results, "runtime": det.status()}
    fast_json.write_text(json.dumps(fast, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print("[B] 完成，%.1f s → %s" % (wall_fast, fast_json))

    # ---- 3) 比对 ----
    d = diff_results(slow, fast)
    print("")
    print("-" * 72)
    print("比对结果")
    print("-" * 72)
    print("  对齐的图片     : %d 张" % d["compared_images"])
    print("  配上的框       : %d 个" % d["paired_boxes"])
    print("  框数不一致的图 : %d 张 %s"
          % (len(d["count_mismatch"]),
             "" if not d["count_mismatch"] else d["count_mismatch"][:5]))
    print("  配不上的框     : %d 个 %s"
          % (len(d["unmatched_boxes"]),
             "" if not d["unmatched_boxes"] else d["unmatched_boxes"][:5]))
    print("  最大位置偏差   : dx=%.3f px  dy=%.3f px（容差 %.1f px）"
          % (d["max_dx_px"], d["max_dy_px"], d["tolerance"]["px"]))
    print("  最大分数偏差   : %.6f（容差 %.1e）"
          % (d["max_score_delta"], d["tolerance"]["score"]))
    print("  单帧均值延迟   : 2 核 %.0f ms → 全速 %.0f ms（加速 %.2fx，墙钟 %.1fs → %.1fs）"
          % (d["latency_ms_mean_slow_2core"], d["latency_ms_mean_fast"],
             d["speedup"] or 0.0, wall_slow, wall_fast))
    print("")
    if d["identical"]:
        print("  ✅ 一致。限核到 2 核【不影响识别结果】，只影响速度。")
        print("     可以放心：开发机上看到的框，就是工控机上会得到的框。")
    else:
        print("  ❌ 有差异！这不是「线程数导致的浮点抖动」能解释的，必须查：")
        if d["count_mismatch"]:
            print("     - 框数不同 → 通常是 conf/iou/mm-per-px 两边没对齐，")
            print("       或者两边用的不是同一个 best.onnx / model_meta.json")
        if d["max_dx_px"] > d["tolerance"]["px"] or d["max_dy_px"] > d["tolerance"]["px"]:
            print("     - 位置偏差 %.2f/%.2f px → 检查 detect.py 的切片常量是否被改过"
                  % (d["max_dx_px"], d["max_dy_px"]))
        if d["only_in_slow"] or d["only_in_fast"]:
            print("     - 两边图片集合不同（%d/%d）→ --limit 或目录内容变了"
                  % (len(d["only_in_slow"]), len(d["only_in_fast"])))
    print("-" * 72)

    if args.json:
        jp = Path(args.json)
        if not jp.is_absolute():
            jp = ROOT / jp
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "compare",
            "slow": {"cores": args.compare_cores, "threads": args.compare_threads,
                     "wall_seconds": round(wall_slow, 2), "runtime": slow.get("runtime"),
                     "summary": slow.get("summary")},
            "fast": {"workers": pl["workers"], "intra": pl["intra"],
                     "wall_seconds": round(wall_fast, 2), "runtime": det.status()},
            "diff": d,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[out] 比对 JSON: %s" % jp)
    print("[out] 两份原始结果保留在 %s（比对不一致时用来对账）" % cmp_dir)
    return 0 if d["identical"] else 3


# ==========================================================================
# 参数
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="detect_test.py",
        description="电池包上盖异物检测 —— 开发机全速测试端（不锁核；禁止部署到工控机）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python detect_test.py --selftest\n"
               "  python detect_test.py --calibrate --image frame.jpg\n"
               "  python detect_test.py --model deploy/best.onnx --image frame.jpg --show\n"
               "  python detect_test.py --model deploy/best.onnx --dir D:/frames "
               "--out D:/results --save-img --report D:/results/index.html\n"
               "  python detect_test.py --model deploy/best.onnx --dir D:/frames --compare\n"
               "\n"
               "和 detect.py 的区别只有速度：切片几何、letterbox、解码、NMS、mm 口径\n"
               "全部 import 自 detect.py，没有第二份实现。\n")
    ap.add_argument("--model", help="best.onnx 路径（同目录需有 model_meta.json）")
    ap.add_argument("--image", nargs="*", help="单张或多张图片路径")
    ap.add_argument("--dir", help="批量目录（递归）")
    ap.add_argument("--out", help="输出目录（配合 --save-img / --report）")
    ap.add_argument("--json", help="结果 JSON 输出路径")
    ap.add_argument("--save-img", action="store_true", help="同时输出画框图")
    ap.add_argument("--report", help="生成单文件 HTML 报告（会自动打开 --save-img）")
    ap.add_argument("--preview-side", type=int, default=0,
                    help="画框图最长边限制（0=原尺寸）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理 N 张（0=全部）")
    ap.add_argument("--warmup", type=int, default=1, help="预热帧数，默认 1")
    ap.add_argument("--show", action="store_true",
                    help="每张弹窗显示（强制单进程；按 q 或 ESC 停止浏览）")
    ap.add_argument("--show-pause", action="store_true",
                    help="配合 --show：每张等待按键而不是自动翻页")

    ap.add_argument("--conf", type=float, default=None, help="置信度阈值（默认取 meta）")
    ap.add_argument("--iou", type=float, default=None, help="NMS IoU 阈值（默认取 meta）")
    ap.add_argument("--max-det", type=int, default=None, help="单帧最多保留框数")
    ap.add_argument("--mm-per-px", type=float, default=None,
                    help="覆盖 meta 里的标定值（先用 --calibrate 实测）")
    ap.add_argument("--tta", action="store_true",
                    help="水平翻转 TTA：召回一般 +0.5~1.5pt，耗时约 x2")

    g = ap.add_argument_group("算力（本脚本的唯一卖点）")
    g.add_argument("--threads", type=int, default=None,
                   help="每条进程的 ORT intra_op 线程数，默认 = 本机核数 / 进程数")
    g.add_argument("--workers", type=int, default=0,
                   help="并发进程数，0=自动（多张图时 ≈ 核数/2，上限 8），1=单进程")
    g.add_argument("--parallel", action="store_true",
                   help="额外打开 ORT_PARALLEL 图级并行（配 --inter > 1 才有意义）")
    g.add_argument("--inter", type=int, default=1, help="ORT inter_op 线程数，默认 1")
    g.add_argument("--no-spin", action="store_true",
                   help="关闭 ORT 线程自旋等待（默认开：烧 CPU 换低延迟，测试机合适）")

    g2 = ap.add_argument_group("A/B 比对")
    g2.add_argument("--compare", action="store_true",
                    help="同一批图跑 detect.py(2 核) 与本脚本(全速)，逐个框比对")
    g2.add_argument("--compare-cores", default="0,1",
                    help="比对时给 detect.py 的核，默认 '0,1'")
    g2.add_argument("--compare-threads", type=int, default=2,
                    help="比对时给 detect.py 的 intra_op，默认 2")

    g3 = ap.add_argument_group("其它")
    g3.add_argument("--takt-ms", type=int, default=30000,
                    help="节拍预算（ms），仅用于汇总时提示，默认 30000")
    g3.add_argument("--allow-grid-mismatch", action="store_true",
                    help="grid_signature 不一致时只告警不退出（仅排查用）")
    g3.add_argument("--quiet", action="store_true", help="不逐张打印")
    g3.add_argument("--selftest", action="store_true",
                    help="转发给 detect.py：切片几何自检，不需要模型和第三方库")
    g3.add_argument("--calibrate", action="store_true",
                    help="转发给 detect.py：实测 mm/px 并给出档位建议")
    g3.add_argument("--ref-mm", type=float, default=None, help="标定用真实长度 mm")
    g3.add_argument("--ref-px", type=float, default=None, help="标定用像素长度 px")
    return ap


def print_banner(pl: Dict[str, Any]) -> None:
    print("=" * 72)
    print(" detect_test.py —— 开发/测试用【全速】推理脚本")
    print("=" * 72)
    print(" ⚠ 不要拷到工控机上跑：本脚本会吃满全部 %d 核，抢走换电服务的 CPU。"
          % pl["cpu"])
    print("   工控机上只用 detect.py（锁 2 核 + BELOW_NORMAL 优先级）。")
    print("")
    print(" 算力计划 : %d 进程 x %d 线程/进程 = %d 线程（本机 %d 核）%s"
          % (pl["workers"], pl["intra"], pl["total_threads"], pl["cpu"],
             "  ← 超订!" if pl["oversubscribed"] else ""))
    for n in pl["notes"]:
        print(" [WARN] %s" % n)
    print(" 精度保证 : 切片/letterbox/解码/NMS/mm 口径全部 import 自 detect.py，")
    print("            本文件没有第二份实现 → 这里的结果就是工控机上的结果。")
    print("=" * 72)
    print("")


# ==========================================================================
# main
# ==========================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    # 自检/标定与算力无关，直接转发给 detect.py 的同一份实现
    if args.selftest:
        return base.cmd_selftest(args)
    if args.calibrate:
        return base.cmd_calibrate(args)

    if not args.model:
        print("[ERROR] 需要 --model（或用 --selftest / --calibrate）")
        print("        python detect_test.py --help")
        return 1
    mp = Path(args.model)
    if not mp.is_absolute():
        mp = ROOT / mp
    if not mp.is_file():
        print("[ERROR] 模型不存在: %s" % mp)
        print("        先在训练机上跑: python train.py export")
        return 1

    paths = collect_paths(args)
    if not paths:
        print("[ERROR] 没有输入图片（--image / --dir 至少给一个）")
        return 1
    missing = [p for p in paths if not p.is_file()]
    if missing:
        print("[WARN] %d 个路径不存在，已忽略（例如 %s）" % (len(missing), missing[0]))
        paths = [p for p in paths if p.is_file()]
        if not paths:
            print("[ERROR] 没有可用图片")
            return 1

    pl = plan(args.workers, args.threads, len(paths))

    # ORT_PARALLEL 只有在 inter_op > 1 时才真的会并发调度算子，否则等于白开
    inter = max(1, int(args.inter))
    if args.parallel and inter <= 1:
        inter = 2
        print("[fast] --parallel 需要 inter_op > 1 才有效，已自动设为 2。")

    det_kwargs = {
        "model_path": str(mp),
        "conf_thr": args.conf,
        "iou_thr": args.iou,
        "mm_per_px_override": args.mm_per_px,
        "max_det": args.max_det,
        "intra_threads": pl["intra"],
        "inter_threads": inter,
        "tta": args.tta,
        "allow_grid_mismatch": args.allow_grid_mismatch,
        "parallel": args.parallel,
        "spin": not args.no_spin,
        "workers": pl["workers"],
    }

    # --report 需要画框图，自动补上 --save-img
    if args.report and not args.save_img:
        print("[report] HTML 报告需要缩略图，已自动开启 --save-img。")
        args.save_img = True

    # 输出目录：--compare 也要用它落盘中间 JSON，所以必须在 compare 分支之前解析
    out_dir: Optional[Path] = None
    if args.out:
        out_dir = Path(args.out)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.save_img:
            (out_dir / "annotated").mkdir(parents=True, exist_ok=True)
    elif args.save_img or args.report or args.compare:
        out_dir = ROOT / "test_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.save_img:
            (out_dir / "annotated").mkdir(parents=True, exist_ok=True)
        print("[out] 没给 --out，产物默认写到 %s" % out_dir)

    if args.compare:
        return run_compare(args, mp, paths, det_kwargs, pl, out_dir)

    print_banner(pl)

    # --show 需要窗口，多进程下窗口会在子进程里开（很多平台直接崩），强制串行
    if args.show and pl["workers"] > 1:
        print("[show] 弹窗浏览与多进程不兼容，已改为单进程（--workers 1）。")
        pl = plan(1, args.threads or CPU_COUNT, len(paths))
        det_kwargs["intra_threads"] = pl["intra"]
        det_kwargs["workers"] = pl["workers"]

    # ---- 建 detector（父进程一定要建：grid_signature 校验、banner、status 都靠它，
    #      不能把这些推迟到子进程里，否则签名不一致时报错又晚又难看）----
    t_all = time.time()
    det = FastDetector(verbose=True, **det_kwargs)
    det._cores_info = {"requested": "all", "affinity_set": False,
                       "priority_lowered": False, "cpu_count": CPU_COUNT,
                       "notes": ["detect_test.py 不做亲和性绑定，也不降优先级"]}

    if pl["workers"] <= 1:
        if args.warmup > 0:
            import numpy as np
            dummy = np.full((det.deploy_h, det.deploy_w, 3), 96, dtype=np.uint8)
            for _ in range(args.warmup):
                det.detect(dummy)
            print("[warmup] 已预热 %d 帧（%dx%d）"
                  % (args.warmup, det.deploy_w, det.deploy_h))
        results = run_serial(det, paths, args, out_dir)
    else:
        print("[workers] 预热交给各子进程自己做（每进程首帧都慢），父进程不预热。")
        results = run_parallel(det_kwargs, pl, paths, args, out_dir)
        if results is None:
            print("[workers] 回退单进程重跑。")
            pl = plan(1, args.threads or CPU_COUNT, len(paths))
            det_kwargs["intra_threads"] = pl["intra"]
            det_kwargs["workers"] = 1
            # ORT 的线程池在建 session 时就按 intra_op_num_threads 定死了，事后改
            # self.intra_threads 只是个数字，不会让已建好的 session 变快。必须重建。
            print("[workers] 重建 session：intra_op %d → %d"
                  % (det.intra_threads, pl["intra"]))
            det = FastDetector(verbose=False, **det_kwargs)
            det._cores_info = {"requested": "all", "affinity_set": False,
                               "priority_lowered": False, "cpu_count": CPU_COUNT,
                               "notes": ["多进程回退单进程"]}
            if args.warmup > 0:
                import numpy as np
                dummy = np.full((det.deploy_h, det.deploy_w, 3), 96, dtype=np.uint8)
                for _ in range(args.warmup):
                    det.detect(dummy)
            results = run_serial(det, paths, args, out_dir)

    wall = time.time() - t_all
    summ = summarize(results, det, pl, args, wall)

    # ---- JSON ----
    if args.json:
        jp = Path(args.json)
        if not jp.is_absolute():
            jp = ROOT / jp
        jp.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "script": "detect_test.py",
            "runtime": det.status(),
            "cores": getattr(det, "_cores_info", None),
            "plan": pl,
            "summary": summ,
            "images": results,
        }
        jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                      encoding="utf-8")
        print("[out] JSON: %s" % jp)

    # ---- HTML ----
    if args.report:
        rp = Path(args.report)
        if not rp.is_absolute():
            rp = ROOT / rp
        if out_dir and rp.parent.resolve() != out_dir.resolve():
            print("[report][WARN] 报告(%s)和画框图(%s)不在同一目录，缩略图会用绝对路径，"
                  % (rp.parent, out_dir))
            print("                 拷给别人看时图片会裂。建议 --report 放进 --out 里。")
        write_html_report(rp, results, det, pl, summ)

    print("")
    print("提醒：以上是【开发机全速】的数字。上线判定请用工控机上的 detect.py，")
    print("      它自己会打印 2 核条件下的延迟和节拍判定。识别结果两边一致，")
    print("      不放心就跑一次 --compare 实测。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[中断] 用户取消")
        raise SystemExit(130)
