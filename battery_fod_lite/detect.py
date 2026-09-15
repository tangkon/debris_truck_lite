#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电池包上盖异物检测 —— 工控机推理端（2 核 CPU，精度优先）

本项目只有三个文件：train.py、data.yaml、detect.py（本文件）。
本文件跑在换电站工控机上，只做推理，不需要 Docker、不需要你自己搭服务。

---------------------------------------------------------------------------
依赖（工控机上只需这两个 + opencv）
---------------------------------------------------------------------------
    pip install onnxruntime opencv-python numpy
    # 想用 OpenVINO CPU 后端加速（可选，仍是同一个 best.onnx 文件）:
    # pip install onnxruntime-openvino        # 装完后本脚本会自动优先选它

---------------------------------------------------------------------------
用法
---------------------------------------------------------------------------
# 单张图
python detect.py --model deploy/best.onnx --image D:/frames/001.jpg

# 批量目录（结果写 JSON，可选同时输出画框图）
python detect.py --model deploy/best.onnx --dir D:/frames --out D:/results ^
                 --json D:/results/result.json --save-img

# 只跑 5 张看延迟（首帧含图初始化，故默认预热 1 次）
python detect.py --model deploy/best.onnx --dir D:/frames --limit 5

# 自检：不加载模型、不依赖 cv2/onnxruntime，只打印切片网格与尺度链
python detect.py --selftest

# 物理标定：实测 mm/px（★ 上线前必做，决定这个档位够不够）
python detect.py --calibrate --image D:/frames/001.jpg
python detect.py --calibrate --ref-mm 500 --ref-px 625

# 精度优先可选开关：水平翻转 TTA（召回一般 +0.5~1.5pt，耗时 x2）
python detect.py --model deploy/best.onnx --image x.jpg --tta

---------------------------------------------------------------------------
★ 只占 2 核，不打扰工控机上的其他服务 —— 四层限流
---------------------------------------------------------------------------
1) 进程亲和性：把本进程钉在指定的 2 个核上（Windows SetProcessAffinityMask /
   Linux sched_setaffinity），其他核完全留给别的业务。默认 --cores 0,1。
2) ONNX Runtime 线程池：intra_op=2、inter_op=1、ORT_SEQUENTIAL，算子内部最多
   2 线程、算子之间不并行，杜绝 ORT 偷偷吃满所有核。
3) BLAS 线程上限：OMP/OPENBLAS/MKL/NUMEXPR/VECLIB 全部设为 2，且在 import
   numpy 之前写入环境变量（numpy 一旦导入就锁定了，之后改无效）。
4) 优先级：Windows 降到 BELOW_NORMAL，Linux nice +10。即使 2 核被占满，
   调度器也会先让别的服务跑。

   全部关掉（例如你想在训练机上全速跑验收）: --cores all
   换核: --cores 2,3

   如果工控机用 systemd 拉起本脚本，更稳妥的做法是同时在 unit 里写死
   （双保险，且重启后仍生效）:
       [Service]
       CPUAffinity=0 1
       Nice=10

---------------------------------------------------------------------------
★ 尺度一致性自检（这套方案的命门）
---------------------------------------------------------------------------
train.py export 会把切片网格的几何签名写进 model_meta.json。本脚本启动时
用【自己这份】slice_grid 重算同一个签名并比对，不一致直接拒绝运行。
这是为了防止"训练用 640x360 切片、推理被人改成 1280x720 切片"这类静默降级：
两种写法切片数可能都是 16、覆盖率都是 100%，肉眼和常规断言都发现不了，
但边缘目标在推理时的放大倍率会变成训练时从未出现过的尺度。

    640x360 切片 → letterbox 1280 → 放大 2.0x → 5mm(@0.8mm/px) = 12.5px ✅
    整帧 1920x1080 → letterbox 1280 → 缩小 0.67x → 5mm = 4.2px        ❌ 不可检

meta 缺失时（比如你只拷了 best.onnx）会退回本文件里的常量并大声告警。
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ==========================================================================
# BLAS / OpenMP 线程上限 —— 必须在 import numpy 之前设置
# ==========================================================================
# numpy 在被导入的那一刻就读取这些变量并锁定线程池，之后再改没有任何效果。
# 所以本文件【故意不写】from __future__ import annotations，也不在顶层 import
# numpy/cv2/onnxruntime —— 全部延迟到函数内部导入，保证这段代码先执行。
# 已经可用 FOD_THREADS 覆盖（例如 FOD_THREADS=4）。
_DEFAULT_THREADS = str(os.environ.get("FOD_THREADS", "2"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _DEFAULT_THREADS)

# Windows 控制台默认 GBK，中文报告会乱码
if sys.platform == "win32":
    def _enc_is_utf8(stream) -> bool:
        e = (getattr(stream, "encoding", "") or "").lower().replace("-", "").replace("_", "")
        return e in ("utf8", "utf8mb4")

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:                       # Python < 3.7 没有 reconfigure
        # 只有在当前编码确实不是 UTF-8 时才替换，并且【保留旧对象引用】：
        # 旧 wrapper 一旦被 GC，会连带关闭共享的底层 BufferedWriter，
        # 之后所有 print 都会静默消失。train.py eval 会 import 本文件，
        # 这段模块级代码因此会执行两次，不防一手就会踩到。
        if not (_enc_is_utf8(sys.stdout) and _enc_is_utf8(sys.stderr)):
            import io
            _OLD_STDOUT, _OLD_STDERR = sys.stdout, sys.stderr      # 防 GC，别删
            if not _enc_is_utf8(sys.stdout):
                sys.stdout = io.TextIOWrapper(_OLD_STDOUT.buffer, encoding="utf-8",
                                              errors="replace", line_buffering=True)
            if not _enc_is_utf8(sys.stderr):
                sys.stderr = io.TextIOWrapper(_OLD_STDERR.buffer, encoding="utf-8",
                                              errors="replace", line_buffering=True)
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent

# ==========================================================================
# 部署常量 —— 与 train.py 里的取值【逐项一致】
# ==========================================================================
# 正常情况下这些值会被 model_meta.json 覆盖（meta 才是权威来源）。
# 这里保留一份，是为了 meta 丢失时仍能跑，以及 --selftest 无依赖自检。
DEPLOY_W, DEPLOY_H = 1920, 1080      # 摄像头出图分辨率（已确认，唯一部署域）
SLICE_W, SLICE_H = 640, 360          # 切片尺寸（精度优先档：放大 2.0x）
OVERLAP_W, OVERLAP_H = 0.2, 0.2      # 切片重叠率
IMGSZ = 1280                         # 模型输入边长 = 切片 letterbox 目标边长
MM_PER_PX = 0.8                      # ★ 必须用 --calibrate 在真实帧上实测替换
MIN_MM = 5.0                         # 业务口径：只检 >5mm 异物
DETECT_FLOOR_PX = 8.0                # YOLO P3 头 stride=8 的经验检测下限
CLASS_NAMES = {0: "foreign"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
PAD_COLOR = 114                      # letterbox 填充色，必须与训练一致


# ==========================================================================
# 切片几何 —— 与 train.py 里的实现【逐字一致】
# ==========================================================================
# ★★★ 为什么两处都要有一份，以及为什么必须一模一样 ★★★
# 切片 letterbox 到 IMGSZ 的放大倍率只由切片自身尺寸决定：
#     scale = IMGSZ / max(slice_w, slice_h)
# 只要训练侧和推理侧在画面边缘产生了不同尺寸的切片，两侧的目标尺度就分叉了。
# 反面教材：用 min(W, x+sw) 夹紧边界的写法，在 1920 宽 + 640 切片上会得到
#     xs = [0, 512, 1024, 1536]，最右一片只有 384x360 → scale = 1280/384 = 3.33x
# 而贴边补齐写法得到
#     xs = [0, 512, 1024, 1280]，每片都是完整 640x360 → scale = 1280/640 = 2.00x
# 两者切片数都是 16、覆盖率都是 100%，肉眼和常规断言都发现不了，但画面右/下
# 边缘的目标在推理时被放大了 1.67 倍 —— 训练时从未出现过的尺度，症状是
# "画面中间检得好、靠边就漏检/误检"。
#
# 本项目用 grid_signature() 做强制交叉校验：train.py export 时把签名写进
# model_meta.json，detect.py 启动时用【自己那份】slice_grid 重算并比对，
# 不一致直接拒绝运行。所以即使有人手改了其中一份，也会在第一次推理时暴露。
def slice_starts(total: int, size: int, overlap: float) -> List[int]:
    """一维切片起点：等距步进网格 + 贴边补齐。

    保证每个起点 s 都满足 s + size <= total（图像比切片还小时退化为 [0]），
    因此裁出来的每一片都是完整的 size 长度，绝不会出现被夹紧的"半片"。
    最后一格若够不到边界，就把起点回退到 total - size：宁可局部重叠变大，
    也不能改变 letterbox 倍率。
    """
    if size <= 0:
        raise ValueError("slice size must be positive, got %s" % size)
    ov = max(0.0, min(float(overlap), 0.9))
    step = max(1, int(size * (1.0 - ov)))
    starts = list(range(0, max(1, total - size + 1), step))
    if starts and starts[-1] + size < total:
        starts.append(total - size)              # 贴边补齐
    return starts or [0]


def slice_grid(W: int, H: int, sw: int, sh: int,
               ow: float, oh: float) -> List[Tuple[int, int, int, int]]:
    """二维切片网格 → [(x1, y1, x2, y2), ...]，行优先（先 x 后 y）。"""
    xs = slice_starts(W, sw, ow)
    ys = slice_starts(H, sh, oh)
    return [(x, y, min(W, x + sw), min(H, y + sh))
            for y in ys for x in xs]


def letterbox_scale(sw: int, sh: int, input_size: int) -> float:
    """切片等比 letterbox 到 input_size 的放大倍率（由长边决定）。"""
    return input_size / max(1, max(sw, sh))


def grid_signature(W: int, H: int, sw: int, sh: int, ow: float, oh: float,
                   input_size: int) -> str:
    """把切片网格的全部几何特征压成一个可比对的字符串。

    detect.py 用它做"训练尺度 = 推理尺度"的强制校验。
    """
    g = slice_grid(W, H, sw, sh, ow, oh)
    xs = sorted({a for a, _, _, _ in g})
    ys = sorted({b for _, b, _, _ in g})
    sizes = sorted({(x2 - x1, y2 - y1) for x1, y1, x2, y2 in g})
    scales = sorted({round(letterbox_scale(w, h, input_size), 4)
                     for w, h in sizes})
    raw = ("{W}x{H}|{sw}x{sh}|ov{ow:.2f},{oh:.2f}|in{inp}|n{n}|"
           "xs={xs}|ys={ys}|sizes={sz}|scales={sc}").format(
        W=W, H=H, sw=sw, sh=sh, ow=ow, oh=oh, inp=input_size, n=len(g),
        xs=",".join(map(str, xs)), ys=",".join(map(str, ys)),
        sz=",".join("%dx%d" % s for s in sizes),
        sc=",".join("%.4f" % s for s in scales))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16] + "|" + raw


def min_object_px(min_mm: float, mm_per_px: float, scale: float) -> float:
    """min_mm 口径的异物在【模型输入空间】里占多少像素。"""
    if mm_per_px <= 0 or min_mm <= 0:
        return 0.0
    return (min_mm / mm_per_px) * scale


def verdict(px_model: float) -> str:
    """把模型空间像素数翻译成一句人话。"""
    if px_model <= 0:
        return "未标定"
    if px_model >= DETECT_FLOOR_PX * 1.3:
        return "OK"
    if px_model >= DETECT_FLOOR_PX:
        return "边缘"
    return "不可检"


# ==========================================================================
# 图像 I/O 工具（Unicode 路径安全）
# ==========================================================================
def imread_u(path):
    """cv2.imread 在 Windows 上遇到中文路径会返回 None，这里绕过。"""
    import cv2
    import numpy as np
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_u(path, img) -> bool:
    import cv2
    try:
        path = Path(path)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix or ".jpg"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


# ==========================================================================
# 2 核限流
# ==========================================================================
def confine_to_cores(cores: Optional[Sequence[int]] = (0, 1),
                     lower_priority: bool = True,
                     verbose: bool = True) -> Dict[str, Any]:
    """把当前进程锁在指定核上并降低优先级。

    cores=None 或 "all" → 不做亲和性绑定（仍会打印当前状态）。
    返回一个 dict，便于写进 JSON 结果里做留痕。
    注意：必须在创建 onnxruntime session 之前调用，ORT 的线程池在 session
    创建时就按当时的亲和性分配好了。
    """
    info: Dict[str, Any] = {"requested": list(cores) if cores else "all",
                            "platform": sys.platform, "affinity_set": False,
                            "affinity_actual": None, "priority_lowered": False,
                            "notes": []}
    ncpu = os.cpu_count() or 1
    info["cpu_count"] = ncpu

    if cores:
        want = [int(c) for c in cores if 0 <= int(c) < ncpu]
        bad = [c for c in cores if not (0 <= int(c) < ncpu)]
        if bad:
            info["notes"].append("忽略不存在的核: %s（本机共 %d 核，编号 0-%d）"
                                 % (bad, ncpu, ncpu - 1))
        if not want:
            info["notes"].append("没有可用核号，跳过亲和性绑定")
        elif sys.platform == "win32":
            try:
                import ctypes
                mask = 0
                for c in want:
                    mask |= (1 << int(c))
                k32 = ctypes.windll.kernel32
                k32.GetCurrentProcess.restype = ctypes.c_void_p
                k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
                k32.SetProcessAffinityMask.restype = ctypes.c_int
                proc = k32.GetCurrentProcess()
                if k32.SetProcessAffinityMask(proc, mask):
                    info["affinity_set"] = True
                    # 回读确认真的生效了
                    get_mask = ctypes.c_size_t(0)
                    sys_mask = ctypes.c_size_t(0)
                    k32.GetProcessAffinityMask.argtypes = [
                        ctypes.c_void_p,
                        ctypes.POINTER(ctypes.c_size_t),
                        ctypes.POINTER(ctypes.c_size_t)]
                    if k32.GetProcessAffinityMask(proc, ctypes.byref(get_mask),
                                                  ctypes.byref(sys_mask)):
                        info["affinity_actual"] = [
                            i for i in range(ncpu) if get_mask.value & (1 << i)]
                else:
                    info["notes"].append("SetProcessAffinityMask 返回失败")
                if lower_priority:
                    k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
                    BELOW_NORMAL = 0x00004000
                    if k32.SetPriorityClass(proc, BELOW_NORMAL):
                        info["priority_lowered"] = True
                        info["priority"] = "BELOW_NORMAL"
            except Exception as e:                        # pragma: no cover
                info["notes"].append("Windows 亲和性设置异常: %r" % (e,))
        elif hasattr(os, "sched_setaffinity"):            # Linux
            try:
                os.sched_setaffinity(0, set(want))
                info["affinity_set"] = True
                info["affinity_actual"] = sorted(os.sched_getaffinity(0))
                if lower_priority:
                    try:
                        os.nice(10)
                        info["priority_lowered"] = True
                        info["priority"] = "nice+10"
                    except OSError as e:
                        info["notes"].append("降优先级失败（需要权限）: %s" % e)
            except OSError as e:                          # pragma: no cover
                info["notes"].append("sched_setaffinity 失败: %s" % e)
        else:                                             # macOS 等
            info["notes"].append(
                "%s 不支持进程级核绑定，仅靠 BLAS/ORT 线程数上限限流" % sys.platform)

    if verbose:
        if info["affinity_set"]:
            print("[cores] 已绑定到核 %s（实际生效: %s）"
                  % (info["requested"], info.get("affinity_actual")))
        else:
            print("[cores] 未绑定核（%s），本机共 %d 核"
                  % (info["requested"], ncpu))
        if info["priority_lowered"]:
            print("[cores] 进程优先级已降低: %s" % info.get("priority"))
        print("[cores] BLAS 线程上限: %s=%s; ONNX Runtime intra_op 将在建 session 时设为 %s"
              % ("OMP/OPENBLAS/MKL/NUMEXPR/VECLIB", _DEFAULT_THREADS,
                 os.environ.get("FOD_THREADS", _DEFAULT_THREADS)))
        for n in info["notes"]:
            print("[cores][WARN] %s" % n)
    return info


def parse_cores(text: Optional[str]) -> Optional[List[int]]:
    """--cores 解析：'0,1' → [0,1]；'all'/''/None → None（不绑定）。"""
    if text is None:
        return [0, 1]
    t = str(text).strip().lower()
    if t in ("all", "none", "*", ""):
        return None
    out = []
    for part in t.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            print("[cores][WARN] 无法解析核号 %r，已忽略" % part)
    return out or None


# ==========================================================================
# meta 加载
# ==========================================================================
def find_meta(model_path: Path) -> Optional[Path]:
    """在模型同目录 / 上级目录找 model_meta.json。"""
    cands = [model_path.parent / "model_meta.json",
             model_path.parent.parent / "model_meta.json",
             ROOT / "deploy" / "model_meta.json"]
    for c in cands:
        if c.is_file():
            return c
    return None


def load_meta(model_path: Path, verbose: bool = True) -> Dict[str, Any]:
    """读取 model_meta.json；缺失时退回本文件常量并大声告警。"""
    p = find_meta(model_path)
    if p is None:
        if verbose:
            print("=" * 72)
            print("[meta][WARN] 找不到 model_meta.json（在 %s 附近都没找到）"
                  % model_path.parent)
            print("             将退回 detect.py 内置常量:")
            print("               切片 %dx%d  重叠 %.2f  输入 %d  mm/px %.2f  最小 %.1fmm"
                  % (SLICE_W, SLICE_H, OVERLAP_W, IMGSZ, MM_PER_PX, MIN_MM))
            print("             后果：无法校验【训练尺度 = 推理尺度】。如果训练时改过")
            print("             切片参数而这里没同步，边缘目标会静默漏检。")
            print("             正确做法：把 train.py export 生成的整个 deploy 目录")
            print("             （best.onnx + model_meta.json）一起拷过来。")
            print("=" * 72)
        return {"schema_version": 0, "_fallback": True,
                "imgsz": IMGSZ, "deploy_width": DEPLOY_W, "deploy_height": DEPLOY_H,
                "slice_width": SLICE_W, "slice_height": SLICE_H,
                "overlap_width": OVERLAP_W, "overlap_height": OVERLAP_H,
                "grid_signature": None,
                "letterbox_scale": round(letterbox_scale(SLICE_W, SLICE_H, IMGSZ), 4),
                "mm_per_px": MM_PER_PX, "min_object_mm": MIN_MM,
                "detect_floor_px": DETECT_FLOOR_PX,
                "conf_threshold": 0.25, "iou_threshold": 0.55,
                "max_detections": 200, "nc": len(CLASS_NAMES),
                "names": {str(k): v for k, v in CLASS_NAMES.items()}}
    try:
        meta = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception as e:
        print("[meta][ERROR] %s 解析失败: %s" % (p, e))
        raise SystemExit(1)
    if verbose:
        print("[meta] %s" % p)
    meta["_fallback"] = False
    meta["_meta_path"] = str(p)
    return meta


# ==========================================================================
# Detector
# ==========================================================================
class Detector(object):
    """切片 + ONNX Runtime + 全局 NMS + mm 口径过滤 的完整推理链路。

    与 train.py eval 的调用约定（不要改签名）:
        det = Detector("deploy/best.onnx", conf_thr=0.25, iou_thr=0.55,
                       mm_per_px_override=0.82)
        det.grid_summary()          → str
        det.detect(bgr_ndarray)     → [{"score", "bbox", ...}, ...]

    设计要点：
    * 切片【逐张串行】推理，不做 batch。16 张 1280x1280x3 float32 一起塞进去
      是 314MB，加上 ORT 的 arena，在 4GB 工控机上很危险；串行的峰值只有一张
      的 19.7MB。精度不受影响，延迟换安全。
    * 先按 conf 过滤再 NMS，且 NMS 在【整帧坐标系】上做，跨切片的重复框靠
      overlap=0.2 的重叠区 + 全局 NMS 合并。
    * 最后按物理尺寸过滤：max(w,h) * mm_per_px < min_object_mm 的直接丢掉，
      这样"小于 5mm 的石头/灰尘"不会污染结果，符合业务口径。
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
                 verbose: bool = True):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise IOError("模型不存在: %s" % self.model_path)
        self.verbose = verbose
        self.tta = bool(tta)

        # ---- meta：权威来源 ----
        self.meta = load_meta(self.model_path, verbose=verbose)
        self.imgsz = int(self.meta.get("imgsz", IMGSZ))
        self.slice_w = int(self.meta.get("slice_width", SLICE_W))
        self.slice_h = int(self.meta.get("slice_height", SLICE_H))
        self.overlap_w = float(self.meta.get("overlap_width", OVERLAP_W))
        self.overlap_h = float(self.meta.get("overlap_height", OVERLAP_H))
        self.deploy_w = int(self.meta.get("deploy_width", DEPLOY_W))
        self.deploy_h = int(self.meta.get("deploy_height", DEPLOY_H))
        self.nc = int(self.meta.get("nc", len(CLASS_NAMES)))
        names = self.meta.get("names") or {}
        self.names = {int(k): v for k, v in names.items()} if names else dict(CLASS_NAMES)
        self.floor_px = float(self.meta.get("detect_floor_px", DETECT_FLOOR_PX))

        self.conf_thr = float(conf_thr if conf_thr is not None
                              else self.meta.get("conf_threshold", 0.25))
        self.iou_thr = float(iou_thr if iou_thr is not None
                             else self.meta.get("iou_threshold", 0.55))
        self.max_det = int(max_det if max_det is not None
                           else self.meta.get("max_detections", 200))
        mm_default = float(self.meta.get("mm_per_px", MM_PER_PX) or MM_PER_PX)
        self.mm_per_px = float(mm_per_px_override) if mm_per_px_override else mm_default
        self.mm_overridden = bool(mm_per_px_override)
        self.min_mm = float(self.meta.get("min_object_mm", MIN_MM))

        self.scale = letterbox_scale(self.slice_w, self.slice_h, self.imgsz)
        self.px_model = min_object_px(self.min_mm, self.mm_per_px, self.scale)

        # ---- 尺度一致性强制校验 ----
        self.grid_sig = grid_signature(self.deploy_w, self.deploy_h,
                                       self.slice_w, self.slice_h,
                                       self.overlap_w, self.overlap_h,
                                       self.imgsz)
        self._check_signature(allow_grid_mismatch)

        # ---- 预生成部署域切片网格（每帧都用同一套，避免重复计算）----
        self._grid_cache: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = {}

        # ---- ONNX Runtime session ----
        self.intra_threads = int(intra_threads if intra_threads is not None
                                 else int(os.environ.get("FOD_THREADS", _DEFAULT_THREADS)))
        self.inter_threads = int(inter_threads)
        self.sess = None
        self.input_name = None
        self.ort_provider = None
        self._load_session()

        if verbose:
            self._print_banner()

    # ------------------------------------------------------------------
    def _check_signature(self, allow: bool) -> None:
        meta_sig = self.meta.get("grid_signature")
        if not meta_sig:
            if self.verbose and not self.meta.get("_fallback"):
                print("[grid][WARN] meta 里没有 grid_signature（旧版导出？），跳过交叉校验")
            return
        if str(meta_sig) == str(self.grid_sig):
            if self.verbose:
                print("[grid] 签名校验通过: %s" % self.grid_sig.split("|")[0])
            return
        msg = (
            "\n" + "=" * 72 + "\n"
            "[grid][FATAL] 训练尺度 ≠ 推理尺度，拒绝运行。\n"
            "  meta 里的签名 : %s\n"
            "  本脚本重算的 : %s\n\n"
            "  这意味着 detect.py 的切片几何与训练时对不上。目标在推理时的放大\n"
            "  倍率会变成训练时从未出现过的尺度，结果是【静默漏检】（尤其是画面\n"
            "  边缘），指标看着还行但上线就翻车。\n\n"
            "  常见原因:\n"
            "    1) detect.py 被人手改过 SLICE_W/SLICE_H/OVERLAP_*/IMGSZ；\n"
            "    2) best.onnx 与 model_meta.json 不是同一次 export 产出的；\n"
            "    3) 训练时用了 --slice-width/--overlap 覆盖，导出时没带同样参数。\n\n"
            "  正确做法: 用 train.py export 重新导出，并把整个 deploy 目录\n"
            "  （best.onnx + model_meta.json）一起拷到工控机。\n"
            + "=" * 72) % (meta_sig, self.grid_sig)
        if allow:
            print(msg.replace("[grid][FATAL]", "[grid][WARN]"))
            print("[grid] --allow-grid-mismatch 已指定，继续运行（结果不可信，仅供排查）")
        else:
            print(msg)
            raise SystemExit(2)

    # ------------------------------------------------------------------
    def _tune_session(self, so, ort) -> None:
        """SessionOptions 扩展点。基类什么都不做，行为与以前完全一致。

        存在的意义：detect_test.py 的 FastDetector 需要打开 ORT_PARALLEL、
        开内存模式优化等，如果它去复制整段 _load_session，两边迟早会漂移
        （而这个项目最怕的就是"两份实现悄悄不一致"）。所以只留一个钩子。
        """
        return None

    # ------------------------------------------------------------------
    def _load_session(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            print("[ERROR] 没装 onnxruntime。工控机上执行: pip install onnxruntime")
            print("        （想用 OpenVINO CPU 后端: pip install onnxruntime-openvino）")
            raise SystemExit(1)

        so = ort.SessionOptions()
        so.intra_op_num_threads = self.intra_threads      # 算子内部并行度 = 2
        so.inter_op_num_threads = self.inter_threads      # 算子之间不并行
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 精度优先：关掉可能引入数值差异的激进优化不必要，ORT_ENABLE_ALL 是
        # 无损图优化（常量折叠/节点融合），FP32 结果与 .pt 基本一致。
        self._tune_session(so, ort)          # 扩展点，基类无操作（见 detect_test.py）

        providers = []
        avail = ort.get_available_providers()
        if "OpenVINOExecutionProvider" in avail:
            # onnxruntime-openvino：同一个 best.onnx，走 OpenVINO CPU 后端
            providers.append(("OpenVINOExecutionProvider",
                              {"device_type": "CPU", "num_streams": 1}))
        providers.append("CPUExecutionProvider")
        try:
            self.sess = ort.InferenceSession(str(self.model_path), sess_options=so,
                                             providers=providers)
        except Exception:
            # 某些 ORT 版本不接受 OpenVINO 的参数字典，退回纯 CPU
            self.sess = ort.InferenceSession(str(self.model_path), sess_options=so,
                                             providers=["CPUExecutionProvider"])
        self.ort_provider = self.sess.get_providers()[0] if self.sess.get_providers() else "?"

        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        shape = list(inp.shape)
        # 静态导出应为 [1, 3, imgsz, imgsz]
        dims = [d for d in shape if isinstance(d, int) and d > 0]
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            net_h, net_w = shape[2], shape[3]
            if net_h != self.imgsz or net_w != self.imgsz:
                print("=" * 72)
                print("[onnx][FATAL] 模型输入是 %dx%d，但 meta 说 imgsz=%d。"
                      % (net_w, net_h, self.imgsz))
                print("              letterbox 边长必须等于训练/导出时的 imgsz，否则")
                print("              目标尺度全错。请重新 export 或修正 meta。")
                print("=" * 72)
                raise SystemExit(2)
            self.net_w, self.net_h = net_w, net_h
        else:
            print("[onnx][WARN] 模型输入 shape=%s（动态或非 4 维）。" % shape)
            print("             将按 meta 的 imgsz=%d 构造方形输入；CPU 上动态 shape"
                  " 更慢且无法充分优化，建议静态导出。" % self.imgsz)
            self.net_w = self.net_h = self.imgsz
        if self.verbose:
            print("[onnx] provider=%s  input=%s shape=%s  intra_op=%d inter_op=%d"
                  % (self.ort_provider, self.input_name, shape,
                     self.intra_threads, self.inter_threads))

    # ------------------------------------------------------------------
    def _print_banner(self) -> None:
        print("[model] %s (%.1f MB)"
              % (self.model_path.name, self.model_path.stat().st_size / 1e6))
        print("[cfg] %s" % self.grid_summary())
        if self.mm_overridden:
            print("[cfg] mm/px 已被 --mm-per-px 覆盖为 %.3f（meta 里是 %.3f）"
                  % (self.mm_per_px, float(self.meta.get("mm_per_px", 0) or 0)))
        if self.tta:
            print("[cfg] TTA=水平翻转（耗时约 x2，召回一般 +0.5~1.5pt）")

    def grid_summary(self) -> str:
        g = slice_grid(self.deploy_w, self.deploy_h, self.slice_w, self.slice_h,
                       self.overlap_w, self.overlap_h)
        sizes = sorted({(x2 - x1, y2 - y1) for x1, y1, x2, y2 in g})
        sz = ",".join("%dx%d" % s for s in sizes)
        return ("%dx%d → %d 片 %s（重叠 %.2f/%.2f）→ letterbox %d（%.3fx）"
                " | %.1fmm@%.2fmm/px = 原生 %.2fpx → 模型 %.2fpx（下限 %.0fpx）[%s]"
                " | sig=%s"
                % (self.deploy_w, self.deploy_h, len(g), sz,
                   self.overlap_w, self.overlap_h, self.imgsz, self.scale,
                   self.min_mm, self.mm_per_px,
                   self.min_mm / max(1e-9, self.mm_per_px), self.px_model,
                   self.floor_px, verdict(self.px_model),
                   self.grid_sig.split("|")[0]))

    # ------------------------------------------------------------------
    def grid_for(self, W: int, H: int) -> List[Tuple[int, int, int, int]]:
        key = (W, H)
        g = self._grid_cache.get(key)
        if g is None:
            g = slice_grid(W, H, self.slice_w, self.slice_h,
                           self.overlap_w, self.overlap_h)
            self._grid_cache[key] = g
            if (W, H) != (self.deploy_w, self.deploy_h) and self.verbose:
                print("[grid][WARN] 输入 %dx%d ≠ 训练域 %dx%d！切片网格与训练时不同，"
                      % (W, H, self.deploy_w, self.deploy_h))
                print("               边缘切片的 letterbox 倍率会漂移，且 mm/px 也")
                print("               不是标定值。仅可用于排查，不要拿它的指标当验收。")
        return g

    # ------------------------------------------------------------------
    def _letterbox(self, img, target: int):
        """等比缩放到 target 方形并填充 PAD_COLOR，返回 (blob_img, r, dw, dh)。"""
        import cv2
        import numpy as np
        h, w = img.shape[:2]
        r = min(target / float(h), target / float(w))
        nw = max(1, int(round(w * r)))
        nh = max(1, int(round(h * r)))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dw, dh = (target - nw) / 2.0, (target - nh) / 2.0
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT,
                                    value=(PAD_COLOR, PAD_COLOR, PAD_COLOR))
        # 补齐到精确 target（round 累积误差可能差 1 像素）
        if padded.shape[0] != target or padded.shape[1] != target:
            canvas = np.full((target, target, 3), PAD_COLOR, dtype=np.uint8)
            hh = min(target, padded.shape[0])
            ww = min(target, padded.shape[1])
            canvas[:hh, :ww] = padded[:hh, :ww]
            padded = canvas
        return padded, r, dw, dh

    def _to_blob(self, padded):
        """HWC BGR uint8 → NCHW RGB float32 /255，与 Ultralytics 预处理一致。"""
        import numpy as np
        x = padded[:, :, ::-1].transpose(2, 0, 1)          # BGR→RGB, HWC→CHW
        x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
        return x[None]

    def _infer(self, padded):
        return self.sess.run(None, {self.input_name: self._to_blob(padded)})[0]

    # ------------------------------------------------------------------
    def _decode(self, out) -> "Any":
        """把 ONNX 原始输出解码成 letterbox 空间的 [N,6] = x1,y1,x2,y2,score,cls。

        兼容三种布局：
          [1, 4+nc, A]  Ultralytics 原生导出（不含 NMS）← 本项目走这条
          [1, A, 4+nc]  转置过的
          [1, K, 6]     带 NMS 的导出（x1,y1,x2,y2,score,cls）
        """
        import numpy as np
        a = np.asarray(out)
        while a.ndim > 2 and a.shape[0] == 1:      # 压掉 batch 维
            a = a[0]
        if a.ndim != 2:
            raise ValueError("无法识别的 ONNX 输出维度: %s（期望 3 维 [1,4+nc,A]）"
                             % (np.asarray(out).shape,))

        n_cols_expect = 4 + self.nc
        if a.shape[0] == n_cols_expect and a.shape[1] != n_cols_expect:
            # [4+nc, A] —— Ultralytics 原生导出（不含 NMS），本项目走这条
            a = a.transpose(1, 0)
        elif a.shape[1] == n_cols_expect:
            pass                                   # [A, 4+nc] 已转置过的
        elif a.shape[1] == 6:
            # [K, 6] —— 带 NMS 的导出：x1,y1,x2,y2,score,cls，直接用
            keep = a[:, 4] >= self.conf_thr
            return np.asarray(a[keep][:, :6], dtype=np.float32)
        else:
            raise ValueError(
                "ONNX 输出列数 %d 与期望 4+nc=%d 不符（shape=%s）。"
                "请确认模型是本项目 train.py export 导出的。"
                % (a.shape[1], n_cols_expect, a.shape))

        boxes = a[:, :4]                                    # cx,cy,w,h（letterbox px）
        cls_scores = a[:, 4:4 + self.nc]
        if self.nc == 1:
            scores = cls_scores[:, 0]
            classes = np.zeros_like(scores, dtype=np.int32)
        else:
            classes = np.argmax(cls_scores, axis=1).astype(np.int32)
            scores = cls_scores[np.arange(cls_scores.shape[0]), classes]
        keep = scores >= self.conf_thr
        if not np.any(keep):
            return np.zeros((0, 6), dtype=np.float32)
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        out_arr = np.stack([cx - w / 2.0, cy - h / 2.0,
                            cx + w / 2.0, cy + h / 2.0,
                            scores, classes.astype(np.float32)], axis=1)
        return out_arr.astype(np.float32)

    # ------------------------------------------------------------------
    def _infer_slice(self, sl) -> List[Tuple[float, float, float, float, float, int]]:
        """单张切片 → letterbox 空间的框列表（已解码，未做 NMS）。"""
        import cv2
        padded, r, dw, dh = self._letterbox(sl, self.imgsz)
        raw = self._decode(self._infer(padded))
        res: List[Tuple[float, float, float, float, float, int]] = []
        for x1, y1, x2, y2, s, c in raw:
            # letterbox 空间 → 切片原生像素
            a = (x1 - dw) / r
            b = (y1 - dh) / r
            cc = (x2 - dw) / r
            d = (y2 - dh) / r
            res.append((float(a), float(b), float(cc), float(d),
                        float(s), int(c)))
        if self.tta:
            # 水平翻转 TTA：翻转切片本身再推一次，坐标映射回正向切片。
            # 因为 fliplr=0.5 是训练时用的增强，翻转域模型是见过的。
            flipped = cv2.flip(sl, 1)
            sw = sl.shape[1]
            padded2, r2, dw2, dh2 = self._letterbox(flipped, self.imgsz)
            raw2 = self._decode(self._infer(padded2))
            for x1, y1, x2, y2, s, c in raw2:
                a = (x1 - dw2) / r2
                b = (y1 - dh2) / r2
                cc = (x2 - dw2) / r2
                d = (y2 - dh2) / r2
                # 翻转坐标系 → 正向切片坐标系
                res.append((float(sw - 1 - cc), float(b),
                            float(sw - 1 - a), float(d), float(s), int(c)))
        return res

    # ------------------------------------------------------------------
    def _collect_candidates(self, img, grid) -> List[Tuple[float, float, float, float, float, int]]:
        """逐片推理，把 letterbox/切片坐标一路搬回【整帧坐标系】。

        抽成独立方法是为了给派生类留口子：detect_test.py 在模型是动态 batch
        导出时可以改成批量前向。但后面的全局 NMS 和 mm 口径过滤必须两边共用
        ——那才是决定精度的部分，绝不允许出现两份实现。
        """
        cands: List[Tuple[float, float, float, float, float, int]] = []
        for (x1, y1, x2, y2) in grid:
            sl = img[y1:y2, x1:x2]
            if sl.size == 0:
                continue
            sw, sh = x2 - x1, y2 - y1
            if (sw, sh) != (self.slice_w, self.slice_h):
                # 理论上不会发生（slice_starts 保证每片完整）；真发生了必须喊出来，
                # 因为 letterbox 倍率会变，这一片的结果不可信。
                print("[detect][WARN] 切片 (%d,%d)-(%d,%d) 尺寸 %dx%d ≠ %dx%d，"
                      "倍率 %.3fx ≠ %.3fx，该片区结果不可信！"
                      % (x1, y1, x2, y2, sw, sh, self.slice_w, self.slice_h,
                         letterbox_scale(sw, sh, self.imgsz), self.scale))
            for (a, b, c, d, s, cls) in self._infer_slice(sl):
                # 切片坐标 → 整帧坐标
                cands.append((a + x1, b + y1, c + x1, d + y1, s, cls))
        return cands

    # ------------------------------------------------------------------
    def detect(self, img) -> List[Dict[str, Any]]:
        """整帧 BGR ndarray → 检测结果列表（原生像素坐标）。"""
        import cv2
        import numpy as np
        if img is None:
            return []
        H, W = img.shape[:2]
        grid = self.grid_for(W, H)

        cands = self._collect_candidates(img, grid)

        if not cands:
            return []

        # ---- 全局 NMS（整帧坐标系）----
        # cv2.dnn.NMSBoxes 要的是 [x, y, w, h]（左上角 + 宽高），cands 已经是
        # x1,y1,x2,y2，直接换算即可。
        boxes_xywh: List[List[float]] = []
        scores: List[float] = []
        for (ax, ay, bx, by, s, _cls) in cands:
            w = max(1.0, bx - ax)
            h = max(1.0, by - ay)
            boxes_xywh.append([float(ax), float(ay), float(w), float(h)])
            scores.append(float(s))
        idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores, self.conf_thr, self.iou_thr)
        if idxs is None or len(idxs) == 0:
            return []
        try:
            flat = [int(i) for i in np.asarray(idxs).reshape(-1)]
        except Exception:                                     # pragma: no cover
            flat = [int(i) for i in idxs]

        # ---- 组装 + 物理尺寸过滤 ----
        out: List[Dict[str, Any]] = []
        for i in flat:
            ax, ay, bx, by, s, cls = cands[i]
            ax, ay = max(0.0, min(ax, bx)), max(0.0, min(ay, by))
            bx, by = min(float(W), max(ax, bx)), min(float(H), max(ay, by))
            w_px, h_px = bx - ax, by - ay
            if w_px <= 0 or h_px <= 0:
                continue
            size_mm = max(w_px, h_px) * self.mm_per_px
            if size_mm < self.min_mm:
                continue                                      # 小于业务口径，丢弃
            out.append({
                "bbox": (round(ax, 1), round(ay, 1), round(bx, 1), round(by, 1)),
                "score": round(float(s), 4),
                "class": int(cls),
                "name": self.names.get(int(cls), str(cls)),
                "w_px": round(float(w_px), 1),
                "h_px": round(float(h_px), 1),
                "size_mm": round(float(size_mm), 1),
            })
        out.sort(key=lambda d: -d["score"])
        if len(out) > self.max_det:
            out = out[:self.max_det]
        return out

    # ------------------------------------------------------------------
    def detect_file(self, path) -> Optional[List[Dict[str, Any]]]:
        img = imread_u(path)
        if img is None:
            return None
        return self.detect(img)

    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        """写进 JSON 结果里的运行留痕，出问题时可追溯当时的全部配置。"""
        return {
            "model": str(self.model_path),
            "onnx_provider": self.ort_provider,
            "meta_path": self.meta.get("_meta_path"),
            "meta_fallback": bool(self.meta.get("_fallback")),
            "grid_signature": self.grid_sig,
            "deploy_size": [self.deploy_w, self.deploy_h],
            "slice_size": [self.slice_w, self.slice_h],
            "overlap": [self.overlap_w, self.overlap_h],
            "imgsz": self.imgsz,
            "letterbox_scale": round(self.scale, 4),
            "mm_per_px": self.mm_per_px,
            "min_object_mm": self.min_mm,
            "min_object_px_model": round(self.px_model, 2),
            "detect_floor_px": self.floor_px,
            "verdict": verdict(self.px_model),
            "conf_thr": self.conf_thr,
            "iou_thr": self.iou_thr,
            "max_detections": self.max_det,
            "tta": self.tta,
            "intra_op_num_threads": self.intra_threads,
            "inter_op_num_threads": self.inter_threads,
            "blas_thread_env": _DEFAULT_THREADS,
            "python": sys.version.split()[0],
        }


# ==========================================================================
# 画框
# ==========================================================================
def draw(img, dets, mm_per_px: float, max_side: int = 0):
    """在整帧上画框。标签用英文，cv2 画不了中文（会显示成 ???）。"""
    import cv2
    vis = img.copy()
    # 按尺寸给颜色：越大越红（越严重）
    for d in dets:
        x1, y1, x2, y2 = [int(round(v)) for v in d["bbox"]]
        mm = d["size_mm"]
        if mm >= 20:
            color = (0, 0, 255)        # BGR 红
        elif mm >= 10:
            color = (0, 128, 255)      # 橙
        else:
            color = (0, 220, 220)      # 黄
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = "%s %.0fmm %.2f" % (d["name"], mm, d["score"])
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(0, y1 - th - base - 2)
        cv2.rectangle(vis, (x1, ty), (x1 + tw + 4, ty + th + base + 2), color, -1)
        cv2.putText(vis, label, (x1 + 2, ty + th), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # 状态角标
    tag = "%d obj | mm/px=%.2f" % (len(dets), mm_per_px)
    cv2.rectangle(vis, (0, 0), (300, 26), (40, 40, 40), -1)
    cv2.putText(vis, tag, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    if max_side and max(vis.shape[:2]) > max_side:
        r = max_side / float(max(vis.shape[:2]))
        vis = cv2.resize(vis, (int(vis.shape[1] * r), int(vis.shape[0] * r)),
                         interpolation=cv2.INTER_AREA)
    return vis


# ==========================================================================
# 单张 / 批量
# ==========================================================================
def run_paths(args, det) -> int:
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
            return 1
        found = [p for p in sorted(d.rglob("*")) if p.suffix.lower() in IMG_EXTS]
        paths.extend(found)
    if not paths:
        print("[ERROR] 没有输入图片（--image / --dir 至少给一个）")
        return 1
    if args.limit and args.limit > 0:
        paths = paths[:args.limit]

    # 预热：首帧要建线程池、跑图优化，延迟远高于稳态，别把它算进指标
    if args.warmup > 0:
        import numpy as np
        dummy = np.full((det.deploy_h, det.deploy_w, 3), 96, dtype=np.uint8)
        for _ in range(args.warmup):
            det.detect(dummy)
        print("[warmup] 已预热 %d 帧（%dx%d）" % (args.warmup, det.deploy_w, det.deploy_h))

    out_dir = Path(args.out) if args.out else None
    if out_dir and not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.save_img:
            (out_dir / "annotated").mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    total_obj = 0
    lat: List[float] = []
    t_all = time.time()
    for i, p in enumerate(paths, 1):
        img = imread_u(p)
        if img is None:
            print("  [WARN] 读不了 %s（跳过）" % p)
            results.append({"path": str(p), "error": "unreadable"})
            continue
        t0 = time.time()
        dets = det.detect(img)
        ms = (time.time() - t0) * 1000.0
        lat.append(ms)
        total_obj += len(dets)
        rec = {
            "path": str(p),
            "name": p.name,
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "latency_ms": round(ms, 1),
            "count": len(dets),
            "detections": dets,
        }
        results.append(rec)
        if not args.quiet:
            flag = "有异物" if dets else "  干净"
            sizes = ",".join("%.0fmm" % d["size_mm"] for d in dets[:6])
            print("  [%d/%d] %s  %s  n=%d  %s  %.0fms"
                  % (i, len(paths), flag, p.name, len(dets), sizes, ms))
        if out_dir and args.save_img:
            vis = draw(img, dets, det.mm_per_px, max_side=args.preview_side)
            imwrite_u(out_dir / "annotated" / (p.stem + "_det.jpg"), vis)

    # ---- 汇总 ----
    n_ok = len(lat)
    lat_sorted = sorted(lat)

    def pct(q: float) -> float:
        if not lat_sorted:
            return 0.0
        k = max(0, min(len(lat_sorted) - 1, int(round(q * (len(lat_sorted) - 1)))))
        return lat_sorted[k]

    print("")
    print("=" * 72)
    print("推理汇总（%d 张，其中 %d 张成功）" % (len(paths), n_ok))
    print("=" * 72)
    if n_ok:
        print("  检出异物的图 : %d 张（%.1f%%）"
              % (sum(1 for r in results if r.get("count", 0) > 0),
                 100.0 * sum(1 for r in results if r.get("count", 0) > 0) / n_ok))
        print("  目标总数     : %d" % total_obj)
        print("  单帧延迟     : 均值 %.0f ms | P50 %.0f | P90 %.0f | P99 %.0f | 最大 %.0f"
              % (sum(lat) / n_ok, pct(0.50), pct(0.90), pct(0.99), lat_sorted[-1]))
        print("  总耗时       : %.1f s" % (time.time() - t_all))
        print("  ORT 后端     : %s（intra_op=%d, inter_op=%d）"
              % (det.ort_provider, det.intra_threads, det.inter_threads))
        budget = args.takt_ms
        if budget > 0:
            p99 = pct(0.99)
            print("  节拍判定     : P99 %.0f ms vs 预算 %d ms → %s"
                  % (p99, budget, "满足" if p99 <= budget else "超预算"))
            print("                 换电站单次换电 3~5 min，单帧 30s 内都可上线；")
            print("                 别为了追求低延迟去降 imgsz 或改切片尺寸——那会")
            print("                 直接把 5mm 目标打到检测下限以下。")
    print("=" * 72)

    # ---- JSON ----
    if args.json:
        jp = Path(args.json)
        if not jp.is_absolute():
            jp = ROOT / jp
        if jp.parent and not jp.parent.exists():
            jp.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime": det.status(),
            "cores": getattr(det, "_cores_info", None),
            "summary": {
                "images": len(paths),
                "ok": n_ok,
                "images_with_object": sum(1 for r in results if r.get("count", 0) > 0),
                "objects": total_obj,
                "latency_ms_mean": round(sum(lat) / n_ok, 1) if n_ok else 0.0,
                "latency_ms_p50": round(pct(0.50), 1),
                "latency_ms_p90": round(pct(0.90), 1),
                "latency_ms_p99": round(pct(0.99), 1),
                "latency_ms_max": round(lat_sorted[-1], 1) if lat_sorted else 0.0,
            },
            "images": results,
        }
        jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                      encoding="utf-8")
        print("[out] JSON: %s" % jp)
    if out_dir and args.save_img:
        print("[out] 画框图: %s" % (out_dir / "annotated"))
    return 0


# ==========================================================================
# --selftest：零依赖自检
# ==========================================================================
def cmd_selftest(args) -> int:
    print("=" * 72)
    print("切片几何自检（不加载模型，不依赖 cv2 / numpy / onnxruntime）")
    print("=" * 72)
    failures = []
    for (W, H) in [(DEPLOY_W, DEPLOY_H)]:
        g = slice_grid(W, H, SLICE_W, SLICE_H, OVERLAP_W, OVERLAP_H)
        xs = slice_starts(W, SLICE_W, OVERLAP_W)
        ys = slice_starts(H, SLICE_H, OVERLAP_H)
        sizes = sorted({(x2 - x1, y2 - y1) for x1, y1, x2, y2 in g})
        scales = sorted({letterbox_scale(w, h, IMGSZ) for w, h in sizes})
        print("  帧尺寸        : %dx%d" % (W, H))
        print("  切片尺寸      : %dx%d  重叠 %.2f/%.2f" % (SLICE_W, SLICE_H, OVERLAP_W, OVERLAP_H))
        print("  X 起点        : %s" % xs)
        print("  Y 起点        : %s" % ys)
        print("  切片数        : %d" % len(g))
        print("  实际切片尺寸  : %s" % ["%dx%d" % s for s in sizes])
        print("  letterbox 倍率: %s → 输入边长 %d" % (["%.3fx" % s for s in scales], IMGSZ))

        # 断言 1：每一片都必须是完整尺寸
        if sizes != [(SLICE_W, SLICE_H)]:
            failures.append("出现了非 %dx%d 的切片: %s" % (SLICE_W, SLICE_H, sizes))
        # 断言 2：只有一个倍率
        if len(scales) != 1:
            failures.append("切片倍率不唯一: %s" % scales)
        # 断言 3：起点不得越界
        for s in xs:
            if s + SLICE_W > W:
                failures.append("X 起点 %d 越界（%d+%d > %d）" % (s, s, SLICE_W, W))
        for s in ys:
            if s + SLICE_H > H:
                failures.append("Y 起点 %d 越界（%d+%d > %d）" % (s, s, SLICE_H, H))
        # 断言 4：100% 覆盖
        covered = set()
        for (x1, y1, x2, y2) in g:
            for xx in range(x1, x2, 8):
                for yy in range(y1, y2, 8):
                    covered.add((xx, yy))
        missed = 0
        for xx in range(0, W, 8):
            for yy in range(0, H, 8):
                if (xx, yy) not in covered:
                    missed += 1
        if missed:
            failures.append("有 %d 个采样点未被任何切片覆盖" % missed)

        # 尺度链
        print("")
        print("  尺度链（mm/px=%.2f，业务口径 %.1fmm）:" % (MM_PER_PX, MIN_MM))
        for name, sc in [("整帧 letterbox %d" % IMGSZ,
                          IMGSZ / float(max(W, H))),
                         ("切片 %dx%d → %d" % (SLICE_W, SLICE_H, IMGSZ),
                          letterbox_scale(SLICE_W, SLICE_H, IMGSZ))]:
            px_native = MIN_MM / MM_PER_PX
            px_model = px_native * sc
            mark = "✅" if verdict(px_model) == "OK" else ("⚠️" if verdict(px_model) == "边缘" else "❌")
            print("    %-24s %5.3fx → %5.2fpx → %6.2fpx  [%s] %s"
                  % (name, sc, px_native, px_model, verdict(px_model), mark))
        print("")
        limit = MIN_MM * letterbox_scale(SLICE_W, SLICE_H, IMGSZ) / DETECT_FLOOR_PX
        print("  本档位支持的 mm/px 上限: %.2f" % limit)
        print("    实测 mm/px <= %.2f → 640x360 切片够用" % limit)
        print("    实测 mm/px >  %.2f → 必须换更小的切片（480x270 → 2.67x，上限 %.2f）"
              % (limit, MIN_MM * letterbox_scale(480, 270, IMGSZ) / DETECT_FLOOR_PX))
        print("")
        print("  grid_signature = %s" % grid_signature(
            W, H, SLICE_W, SLICE_H, OVERLAP_W, OVERLAP_H, IMGSZ))

    print("=" * 72)
    if failures:
        print("自检失败 %d 项:" % len(failures))
        for f in failures:
            print("  ✗ %s" % f)
        return 1
    print("自检通过：切片完整、倍率唯一、覆盖 100%、尺度链在检测下限之上。")
    return 0


# ==========================================================================
# --calibrate：实测 mm/px
# ==========================================================================
def cmd_calibrate(args) -> int:
    print("=" * 72)
    print("物理标定：实测 mm/px")
    print("=" * 72)
    print("为什么要做：整套方案是【切片放大 + 像素口径过滤】，mm/px 是把像素换成")
    print("毫米的唯一桥梁。它错了会有两种后果：")
    print("  * 标小了（比如按近端 0.6 标，实际远端 1.2）→ 远端 5mm 异物只有 4.2px，")
    print("    模型空间 8.3px，贴着下限，表现为【远处总漏】；")
    print("  * 标大了 → 3mm 灰尘也被判成 >5mm 异物，误检暴涨。")
    print("")
    print("★ 关键：要在电池包面的【最远端】测，不是近端。斜视机位下近端约 0.6、")
    print("  远端约 1.2，差近一倍。按最远端标定才能保证全包面都达标。")
    print("")

    img_w = img_h = 0
    # --image 是 nargs="*"，拿到的是列表（哪怕只给了一张）。直接 Path(list) 会
    # TypeError，而 --help 的示例里恰好就写着 --calibrate --image frame.jpg。
    cal_img = args.image[0] if isinstance(args.image, (list, tuple)) and args.image \
        else (args.image if isinstance(args.image, str) else None)
    if cal_img:
        p = Path(cal_img)
        if not p.is_absolute():
            p = ROOT / p
        if not p.is_file():
            print("[ERROR] 图片不存在: %s" % p)
            return 1
        # 优先用 cv2 读；读不到就退化到只解析文件头（--calibrate 不该强依赖 cv2）
        try:
            img = imread_u(p)
            if img is not None:
                img_h, img_w = img.shape[:2]
        except Exception:
            img = None
        if img_w == 0:
            img_w, img_h = _image_size_from_header(p)
        print("标定图: %s  (%dx%d)" % (p.name, img_w, img_h))
        if (img_w, img_h) != (DEPLOY_W, DEPLOY_H):
            print("[WARN] 这张图不是部署域 %dx%d！标定必须在摄像头正式出图的"
                  % (DEPLOY_W, DEPLOY_H))
            print("       分辨率上做，否则 mm/px 对不上。")
        print("")

    ref_mm, ref_px = args.ref_mm, args.ref_px
    if (ref_mm is None or ref_px is None) and img_w and sys.stdin.isatty():
        print("请按下面步骤量一次（任意看图/画图画廊工具都能量像素距离）：")
        print("  1) 在电池包面上放一把钢尺，或者量一段已知长度的特征（如加强筋间距）；")
        print("  2) 记下这段的真实长度 L（mm）；")
        print("  3) 在这张 1920x1080 图上量出它对应的像素长度 P（水平方向量，")
        print("     且要在【包面最远端】量）；")
        print("  4) 把 L 和 P 输进来。")
        print("")
        try:
            if ref_mm is None:
                ref_mm = float(input("  真实长度 L (mm) > ").strip())
            if ref_px is None:
                ref_px = float(input("  像素长度 P (px) > ").strip())
        except (EOFError, KeyboardInterrupt, ValueError):
            print("")
            print("[ERROR] 输入无效。也可以直接给参数：")
            print("        python detect.py --calibrate --ref-mm 500 --ref-px 625")
            return 1

    if not ref_mm or not ref_px:
        print("[ERROR] 需要 --ref-mm 和 --ref-px（或在交互终端里输入）")
        print("        例: python detect.py --calibrate --ref-mm 500 --ref-px 625")
        return 1
    if ref_px <= 0 or ref_mm <= 0:
        print("[ERROR] 长度和像素数都必须为正")
        return 1

    mm_px = ref_mm / ref_px
    print("")
    print("实测 mm/px = %.2f mm / %.2f px = %.4f" % (ref_mm, ref_px, mm_px))
    print("")

    _report_tiers(mm_px, img_w or DEPLOY_W, img_h or DEPLOY_H)
    return 0


def _report_tiers(mm_px: float, W: int, H: int) -> None:
    """对若干候选档位算一遍尺度链，告诉用户当前配置够不够。"""
    scale_now = letterbox_scale(SLICE_W, SLICE_H, IMGSZ)
    px_native = MIN_MM / mm_px
    px_model = px_native * scale_now
    v = verdict(px_model)
    limit_now = MIN_MM * scale_now / DETECT_FLOOR_PX

    print("-" * 72)
    print("当前配置判定（切片 %dx%d → letterbox %d，%.3fx）" % (SLICE_W, SLICE_H, IMGSZ, scale_now))
    print("-" * 72)
    print("  %.1fmm 异物 → 原生 %.2fpx → 模型 %.2fpx（检测下限 %.0fpx）→ [%s]"
          % (MIN_MM, px_native, px_model, DETECT_FLOOR_PX, v))
    print("  该档位支持的 mm/px 上限: %.2f" % limit_now)
    print("")
    if mm_px <= limit_now and v == "OK":
        print("  ✅ 够用。训练和推理都用这个档位，命令行加 --mm-per-px %.3f：" % mm_px)
        print("     python train.py prepare --src <原始数据> --mm-per-px %.3f" % mm_px)
        print("     python train.py all      --src <原始数据> --mm-per-px %.3f" % mm_px)
        print("     python detect.py --model deploy/best.onnx --image x.jpg --mm-per-px %.3f" % mm_px)
        print("")
        print("     更彻底的做法：把 train.py 里的 MM_PER_PX 常量改成 %.3f，" % mm_px)
        print("     这样 meta 会记录真实标定值，detect.py 无需每次传参。")
    elif v == "边缘":
        print("  ⚠️ 贴边（%.2fpx 只比下限 %.0fpx 高一点点）。能跑，但召回会明显"
              % (px_model, DETECT_FLOOR_PX))
        print("     低于理想值，且对标注误差很敏感。建议换更小切片。")
        _suggest_finer(mm_px)
    else:
        print("  ❌ 不够。%.2fpx 低于检测下限 %.0fpx，%.1fmm 异物在模型空间里"
              % (px_model, DETECT_FLOOR_PX, MIN_MM))
        print("     学不到也检不出，必须换更小切片（更高放大倍率）。")
        _suggest_finer(mm_px)
    print("")
    print("-" * 72)
    print("各候选档位对照（mm/px=%.3f，口径 %.1fmm，下限 %.0fpx）" % (mm_px, MIN_MM, DETECT_FLOOR_PX))
    print("-" * 72)
    print("  %-18s %-9s %-9s %-11s %s" % ("切片", "倍率", "模型px", "mm/px上限", "判定"))
    for (sw, sh) in [(1280, 720), (960, 540), (SLICE_W, SLICE_H), (480, 270), (320, 180)]:
        sc = letterbox_scale(sw, sh, IMGSZ)
        g = slice_grid(W, H, sw, sh, OVERLAP_W, OVERLAP_H)
        pxm = px_native * sc
        lim = MIN_MM * sc / DETECT_FLOOR_PX
        mark = "← 当前" if (sw, sh) == (SLICE_W, SLICE_H) else ""
        print("  %-18s %-9.3f %-9.2f %-11.2f [%s] %d 片 %s"
              % ("%dx%d" % (sw, sh), sc, pxm, lim, verdict(pxm), len(g), mark))
    print("")
    print("  注意：切片越小，片数越多，延迟线性上升（4GB 内存下串行推理仍安全）。")
    print("        换电站节拍 3~5 min，单帧 30s 内都能上线，所以【精度优先】时")
    print("        宁可多切几片，也不要为了快把倍率降下来。")


def _suggest_finer(mm_px: float) -> None:
    need_scale = DETECT_FLOOR_PX * 1.3 * mm_px / MIN_MM      # 留 30% 余量
    print("")
    print("  建议：需要放大倍率 >= %.2fx（含 30%% 余量），候选:" % need_scale)
    for (sw, sh) in [(640, 360), (480, 270), (320, 180), (256, 144)]:
        sc = letterbox_scale(sw, sh, IMGSZ)
        n = len(slice_grid(DEPLOY_W, DEPLOY_H, sw, sh, OVERLAP_W, OVERLAP_H))
        ok = "✅" if sc >= need_scale else "  "
        print("    %s %dx%d → %.2fx，%d 片，上限 mm/px %.2f"
              % (ok, sw, sh, sc, n, MIN_MM * sc / DETECT_FLOOR_PX))
    print("")
    print("  换档位必须【训练和推理一起换】：train.py 和 detect.py 的")
    print("  SLICE_W/SLICE_H 是逐字一致的两份，改完要重跑 prepare → train →")
    print("  export，否则 grid_signature 校验会直接拒绝启动（这是故意的）。")


def _image_size_from_header(path: Path) -> Tuple[int, int]:
    """不依赖 cv2，直接从文件头读宽高（PNG IHDR / BMP / JPEG SOF）。"""
    try:
        import struct
        with open(str(path), "rb") as f:
            head = f.read(32)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", head[16:24])
                return int(w), int(h)
            if head[:2] == b"BM":
                w, h = struct.unpack("<ii", head[18:26])
                return int(w), abs(int(h))
            if head[:2] == b"\xff\xd8":
                f.seek(2)
                while True:
                    b = f.read(1)
                    if not b:
                        break
                    if b != b"\xff":
                        continue
                    marker = f.read(1)
                    if not marker:
                        break
                    m = marker[0]
                    if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                        f.read(3)
                        h, w = struct.unpack(">HH", f.read(4))
                        return int(w), int(h)
                    ln = struct.unpack(">H", f.read(2))[0]
                    f.seek(ln - 2, 1)
    except Exception:
        pass
    return 0, 0


# ==========================================================================
# CLI
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="detect.py",
        description="电池包上盖异物检测 —— 工控机推理端（2 核 CPU，精度优先）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python detect.py --selftest\n"
               "  python detect.py --calibrate --image frame.jpg\n"
               "  python detect.py --model deploy/best.onnx --image frame.jpg\n"
               "  python detect.py --model deploy/best.onnx --dir D:/frames "
               "--out D:/results --json D:/results/r.json --save-img\n")
    ap.add_argument("--model", help="best.onnx 路径（同目录需有 model_meta.json）")
    ap.add_argument("--image", nargs="*", help="单张或多张图片路径")
    ap.add_argument("--dir", help="批量目录（递归）")
    ap.add_argument("--out", help="输出目录（配合 --save-img / --json）")
    ap.add_argument("--json", help="结果 JSON 输出路径")
    ap.add_argument("--save-img", action="store_true", help="同时输出画框图")
    ap.add_argument("--preview-side", type=int, default=0,
                    help="画框图最长边限制（0=原尺寸）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理 N 张（0=全部）")
    ap.add_argument("--warmup", type=int, default=1, help="预热帧数，默认 1")

    ap.add_argument("--conf", type=float, default=None, help="置信度阈值（默认取 meta）")
    ap.add_argument("--iou", type=float, default=None, help="NMS IoU 阈值（默认取 meta）")
    ap.add_argument("--max-det", type=int, default=None, help="单帧最多保留框数")
    ap.add_argument("--mm-per-px", type=float, default=None,
                    help="覆盖 meta 里的标定值（先用 --calibrate 实测）")
    ap.add_argument("--tta", action="store_true",
                    help="水平翻转 TTA：召回一般 +0.5~1.5pt，耗时约 x2")

    ap.add_argument("--cores", default="0,1",
                    help="绑定的 CPU 核，默认 '0,1'；'all' 表示不绑定")
    ap.add_argument("--threads", type=int, default=None,
                    help="ONNX Runtime intra_op 线程数，默认取 FOD_THREADS 或 2")
    ap.add_argument("--no-nice", action="store_true", help="不降低进程优先级")
    ap.add_argument("--takt-ms", type=int, default=30000,
                    help="节拍预算（ms），仅用于汇总时判定，默认 30000")
    ap.add_argument("--allow-grid-mismatch", action="store_true",
                    help="grid_signature 不一致时只告警不退出（仅排查用）")
    ap.add_argument("--quiet", action="store_true", help="不逐张打印")

    ap.add_argument("--selftest", action="store_true",
                    help="只做切片几何自检，不需要模型和任何第三方库")
    ap.add_argument("--calibrate", action="store_true", help="实测 mm/px 并给出档位建议")
    ap.add_argument("--ref-mm", type=float, default=None, help="标定用真实长度 mm")
    ap.add_argument("--ref-px", type=float, default=None, help="标定用像素长度 px")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    if args.selftest:
        return cmd_selftest(args)
    if args.calibrate:
        return cmd_calibrate(args)

    if not args.model:
        print("[ERROR] 需要 --model（或用 --selftest / --calibrate）")
        print("        python detect.py --help")
        return 1
    mp = Path(args.model)
    if not mp.is_absolute():
        mp = ROOT / mp
    if not mp.is_file():
        print("[ERROR] 模型不存在: %s" % mp)
        print("        先在训练机上跑: python train.py export")
        return 1
    if not args.image and not args.dir:
        print("[ERROR] 需要 --image 或 --dir")
        return 1

    # ★ 先限流，再建 session：ORT 的线程池按建 session 时的亲和性分配
    cores = parse_cores(args.cores)
    cores_info = confine_to_cores(cores, lower_priority=not args.no_nice)
    try:
        import cv2
        cv2.setNumThreads(1)      # 解码/resize 别偷偷吃满核
    except Exception:
        pass

    intra = args.threads if args.threads else int(os.environ.get("FOD_THREADS", _DEFAULT_THREADS))
    det = Detector(str(mp), conf_thr=args.conf, iou_thr=args.iou,
                   mm_per_px_override=args.mm_per_px, max_det=args.max_det,
                   intra_threads=intra, tta=args.tta,
                   allow_grid_mismatch=args.allow_grid_mismatch)
    det._cores_info = cores_info
    return run_paths(args, det)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[中断] 用户取消")
        raise SystemExit(130)
