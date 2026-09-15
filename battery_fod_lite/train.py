#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电池包上盖异物检测 —— 训练端一体化脚本（数据准备 / 训练 / 导出 ONNX / 端到端验收）

本项目只有三个文件：train.py（本文件）、data.yaml、detect.py。
train.py 在带 GPU 的训练机上跑，detect.py 在 2 核 CPU 工控机上跑。

---------------------------------------------------------------------------
用法
---------------------------------------------------------------------------
一条龙（推荐，从原始标注直接到可部署的 ONNX）:
    python train.py all --src /path/to/raw

分步:
    # 1) 数据准备：统一分辨率 → 清洗标签 → 切 640x360 片
    python train.py prepare --src /path/to/raw --mm-per-px 0.8

    # 2) 训练
    python train.py train --epochs 300 --batch 16 --device 0

    # 3) 导出 ONNX + model_meta.json（detect.py 靠这个 meta 校验尺度一致性）
    python train.py export

    # 4) 端到端验收：用 detect.py 的真实推理链路在【整帧】测试集上算 P/R/mAP50
    python train.py eval

断点续跑（改了 mm_per_px 只需重跑清洗+切片，不用重新缩放图片）:
    python train.py prepare --src /path/to/raw --mm-per-px 0.62 --from-stage 2

---------------------------------------------------------------------------
★ 三个必须知道的硬约束
---------------------------------------------------------------------------
1) 训练尺度 = 推理尺度。
   部署域 1920x1080、mm_per_px≈0.8 → 5mm 异物原生只有 6.25px，低于 YOLO P3
   检测头 ≈8px 的可靠下限。整帧 letterbox 到 1280 是 0.67x 缩小 → 4.2px，
   模型学不到。切成 640x360 再 letterbox 到 1280 是 2.0x 放大 → 12.5px。
   所以【必须用切片集训练】，且 detect.py 必须用完全相同的切片参数推理。

2) multi_scale 必须关闭。
   它把网络输入在 imgsz 的 ±50% 之间抖动，而推理固定在 1280；12.5px 的目标
   抖到 0.5x 只剩 6px，等于用推理时永远见不到的尺度训练。尺度鲁棒性交给
   scale=0.4（只缩放图像内容、不改网络输入）。

3) mm_per_px 必须在真实 1920x1080 帧上实测，且取【包面最远端】。
   斜视机位下近端约 0.6、远端约 1.2，差近一倍。按近端标定会让远端的 5mm
   目标实际不足口径要求，表现为"远处总漏"。实测方法见 detect.py --calibrate。
"""

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import struct
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
        # 之后所有 print 都会静默消失。eval 子命令会 import detect.py，
        # 那边也有一份同样的代码，不防一手就会踩到。
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
# 部署常量 —— 改这里等于改整个部署域，train.py 与 detect.py 必须同步
# ==========================================================================
DEPLOY_W, DEPLOY_H = 1920, 1080      # 摄像头出图分辨率（已确认，唯一部署域）
SLICE_W, SLICE_H = 640, 360          # 切片尺寸（精度优先档：放大 2.0x）
OVERLAP_W, OVERLAP_H = 0.2, 0.2      # 切片重叠率
IMGSZ = 1280                         # 模型输入边长 = 切片 letterbox 目标边长
MM_PER_PX = 0.8                      # ★ 必须在真实帧上实测替换
MIN_MM = 5.0                         # 业务口径：只检 >5mm 异物
DETECT_FLOOR_PX = 8.0                # YOLO P3 头 stride=8 的经验检测下限
BASE_MODEL = "yolo11s.pt"            # 预训练权重
CLASS_NAMES = {0: "foreign"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ==========================================================================
# 切片几何 —— 与 detect.py 里的实现【逐字一致】
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


# ==========================================================================
# 图像 I/O 工具
# ==========================================================================
def imread_u(path: Path):
    """Unicode 路径安全的 imread（Windows 上 cv2.imread 遇到中文路径会返回 None）。"""
    import cv2
    import numpy as np
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_u(path: Path, img) -> bool:
    """Unicode 路径安全的 imwrite。"""
    import cv2
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix or ".jpg"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def image_size(path: Path) -> Optional[Tuple[int, int]]:
    """只读文件头拿 (W, H)，比整图解码快一个数量级。失败回退 cv2。"""
    try:
        with open(str(path), "rb") as f:
            head = f.read(32)
            if len(head) < 24:
                return None
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", head[16:24])
                return int(w), int(h)
            if head[:2] == b"BM":
                w, h = struct.unpack("<ii", head[18:26])
                return abs(int(w)), abs(int(h))
            if head[:2] == b"\xff\xd8":
                with open(str(path), "rb") as f2:
                    data = f2.read()
                i = 2
                n = len(data)
                while i + 9 < n:
                    if data[i] != 0xFF:
                        i += 1
                        continue
                    marker = data[i + 1]
                    if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                                  0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                        h, w = struct.unpack(">HH", data[i + 5:i + 9])
                        return int(w), int(h)
                    if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                        i += 2
                        continue
                    seg = struct.unpack(">H", data[i + 2:i + 4])[0]
                    i += 2 + seg
    except Exception:
        pass
    img = imread_u(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    return int(w), int(h)


def find_label(img_path: Path, src_root: Path) -> Optional[Path]:
    """按 YOLO 约定找标签：把路径里的 /images/ 换成 /labels/，后缀换 .txt。"""
    parts = list(img_path.relative_to(src_root).parts)
    for i, p in enumerate(parts):
        if p == "images":
            parts[i] = "labels"
            cand = src_root.joinpath(*parts).with_suffix(".txt")
            if cand.exists():
                return cand
            # 标签目录平铺的情况
            flat = src_root / "labels" / (img_path.stem + ".txt")
            return flat if flat.exists() else None
    flat = img_path.parent / (img_path.stem + ".txt")
    if flat.exists():
        return flat
    cand = src_root / "labels" / (img_path.stem + ".txt")
    return cand if cand.exists() else None


def read_label(path: Optional[Path]) -> List[Tuple[int, float, float, float, float]]:
    """读 YOLO txt → [(cls, cx, cy, w, h), ...]，坐标已归一化。"""
    out: List[Tuple[int, float, float, float, float]] = []
    if path is None or not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            c = int(float(parts[0]))
            vals = [float(v) for v in parts[1:5]]
        except ValueError:
            continue
        if any(v < -0.001 or v > 1.001 for v in vals):
            continue                        # 越界标签直接丢
        out.append((c, vals[0], vals[1], vals[2], vals[3]))
    return out


def write_label(path: Path, boxes: Sequence[Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for b in boxes:
        c = int(b[0])
        lines.append("%d %.6f %.6f %.6f %.6f" % (c, b[1], b[2], b[3], b[4]))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ==========================================================================
# 阶段 1：统一分辨率 + 划分 train/val/test
# ==========================================================================
def stage_unify(args) -> Dict[str, int]:
    src = Path(args.src).resolve()
    if not src.is_dir():
        die("--src 目录不存在: %s" % src)
    out = Path(args.work) / "01_unified"
    tw, th = args.target_size

    imgs = sorted(p for p in src.rglob("*")
                  if p.suffix.lower() in IMG_EXTS and p.is_file())
    if not imgs:
        die("在 %s 下没找到任何图片（支持 %s）" % (src, sorted(IMG_EXTS)))
    print("[1/3] 统一分辨率: 找到 %d 张图 → %dx%d" % (len(imgs), tw, th))

    # 是否尊重已有的 train/val/test 划分
    rel_split: Dict[Path, str] = {}
    has_split = any(re.search(r"[\\/]images[\\/](train|val|test)[\\/]", str(p))
                    for p in imgs)
    rng = random.Random(args.seed)
    order = imgs[:]
    rng.shuffle(order)
    n = len(order)
    r_tr, r_va, _ = args.ratios
    cut_tr = int(n * r_tr)
    cut_va = int(n * (r_tr + r_va))
    auto_split = {}
    for i, p in enumerate(order):
        auto_split[p] = "train" if i < cut_tr else ("val" if i < cut_va else "test")

    res_hist = Counter()
    stat = {"resized": 0, "kept": 0, "unreadable": 0, "no_label": 0, "boxes": 0}
    for p in imgs:
        size = image_size(p)
        if size is None:
            stat["unreadable"] += 1
            print("  [WARN] 读不出尺寸，跳过: %s" % p)
            continue
        res_hist["%dx%d" % size] += 1

        if has_split:
            m = re.search(r"[\\/]images[\\/](train|val|test)[\\/]", str(p))
            split = m.group(1) if m else auto_split[p]
        else:
            split = auto_split[p]

        dst_img = out / "images" / split / p.name
        dst_lbl = out / "labels" / split / (p.stem + ".txt")
        # 两个目标目录必须在这里就建好：原样拷贝走 shutil.copy2，它不像
        # imwrite_u / write_label 那样自带 mkdir。源图本来就是 1920x1080
        # 是常态（部署域就是它），不建目录会直接 FileNotFoundError。
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        if dst_img.exists() and dst_lbl.exists() and not args.force:
            stat["kept"] += 1
            continue

        lbl_src = find_label(p, src)
        boxes = read_label(lbl_src)
        if lbl_src is None:
            stat["no_label"] += 1                # 当作负样本（干净上盖）

        if size != (tw, th):
            img = imread_u(p)
            if img is None:
                stat["unreadable"] += 1
                continue
            import cv2
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
            if not imwrite_u(dst_img, img):
                stat["unreadable"] += 1
                continue
            stat["resized"] += 1
            # 归一化标签在等比缩放下数值不变；这里是非等比（16:9→16:9 除外），
            # 但 2560x1440 与 1920x1080 同为 16:9，等比 → 标签直接沿用。
            # 若源图宽高比与目标不同，归一化坐标依然正确（各自按自己的 W/H 归一）。
        else:
            shutil.copy2(str(p), str(dst_img))
            stat["kept"] += 1

        write_label(dst_lbl, [(b[0], b[1], b[2], b[3], b[4]) for b in boxes])
        stat["boxes"] += len(boxes)

    print("  源图分辨率分布: %s" % dict(res_hist))
    print("  缩放 %d 张 / 原样 %d 张 / 读失败 %d 张 / 无标签(负样本) %d 张 / 框 %d 个"
          % (stat["resized"], stat["kept"], stat["unreadable"],
             stat["no_label"], stat["boxes"]))
    if len(res_hist) > 1:
        print("  [INFO] 源数据存在多种分辨率，已统一到 %dx%d。" % (tw, th))
        print("         2560x1440 与 1920x1080 同为 16:9 且来自同一相机同一机位，")
        print("         等比缩放后 mm/px 与部署域一致，可以 100% 混用。")
    return stat


# ==========================================================================
# 阶段 2：清洗标签（单类化 + mm 口径过滤 + 负样本补全）→ dataset_full
# ==========================================================================
def stage_clean(args) -> Dict[str, object]:
    work = Path(args.work) / "01_unified"
    if not work.is_dir():
        die("找不到 %s，请先跑阶段 1（或去掉 --from-stage）" % work)
    dst = Path(args.dst_full)
    tw, th = args.target_size
    remap = parse_remap(args.remap)

    print("[2/3] 清洗标签 → %s" % dst)
    print("  类别映射: %s   mm/px=%.4f   最小口径=%.1fmm"
          % (remap, args.mm_per_px, args.min_mm))

    stat = Counter()
    cls_hist_before: Counter = Counter()
    cls_hist_after: Counter = Counter()
    dropped_mm = 0
    per_split: Dict[str, Counter] = defaultdict(Counter)

    for split in ("train", "val", "test"):
        img_dir = work / "images" / split
        if not img_dir.is_dir():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            boxes = read_label(work / "labels" / split / (p.stem + ".txt"))
            stat["images"] += 1
            per_split[split]["images"] += 1

            kept: List[Tuple[int, float, float, float, float]] = []
            for c, cx, cy, w, h in boxes:
                cls_hist_before[c] += 1
                nc = remap.get(c, None)
                if nc is None:
                    stat["dropped_class"] += 1
                    continue
                if not args.no_min_mm_filter:
                    px = max(w * tw, h * th)
                    if px * args.mm_per_px < args.min_mm:
                        dropped_mm += 1
                        continue
                # 裁剪到 [0,1] 并丢弃退化框
                cx = min(max(cx, 0.0), 1.0); cy = min(max(cy, 0.0), 1.0)
                w = min(max(w, 0.0), 1.0);   h = min(max(h, 0.0), 1.0)
                if w <= 0 or h <= 0:
                    continue
                kept.append((nc, cx, cy, w, h))
                cls_hist_after[nc] += 1

            dst_img = dst / "images" / split / p.name
            if not dst_img.exists():
                dst_img.parent.mkdir(parents=True, exist_ok=True)   # copy2 不建目录
                shutil.copy2(str(p), str(dst_img))
            write_label(dst / "labels" / split / (p.stem + ".txt"), kept)

            per_split[split]["boxes"] += len(kept)
            if kept:
                stat["pos"] += 1
                per_split[split]["pos"] += 1
            else:
                stat["neg"] += 1
                per_split[split]["neg"] += 1

    total = stat["pos"] + stat["neg"]
    neg_ratio = (stat["neg"] / total) if total else 0.0
    print("  图片 %d 张（正 %d / 负 %d，负样本占比 %.1f%%）"
          % (total, stat["pos"], stat["neg"], neg_ratio * 100))
    print("  清洗前类别分布: %s" % dict(cls_hist_before))
    print("  清洗后类别分布: %s   （必须只有 {0: ...}）" % dict(cls_hist_after))
    print("  因 <%.1fmm 剔除的框: %d 个；因类别不在映射表剔除: %d 个"
          % (args.min_mm, dropped_mm, stat["dropped_class"]))
    for split in ("train", "val", "test"):
        s = per_split[split]
        print("    %-5s 图 %4d  正 %4d  负 %4d  框 %6d"
              % (split, s["images"], s["pos"], s["neg"], s["boxes"]))

    if any(c != 0 for c in cls_hist_after):
        die("清洗后仍有非 0 类别: %s（检查 --remap）" % dict(cls_hist_after))
    if neg_ratio < 0.10:
        print("  [WARN] 负样本占比仅 %.1f%%，低于 15%% 容易让模型把灰尘/污渍误检成异物。"
              % (neg_ratio * 100))
        print("         建议补一批【干净上盖】图（有灰尘但无异物的也要），无标签即可。")
    if neg_ratio > 0.50:
        print("  [WARN] 负样本占比 %.1f%%，过高会让模型倾向不报，Recall 会掉。"
              % (neg_ratio * 100))
    return {"stat": dict(stat), "neg_ratio": neg_ratio,
            "cls_after": dict(cls_hist_after), "dropped_mm": dropped_mm}


def parse_remap(specs: Sequence[str]) -> Dict[int, int]:
    """['1:0', '2:0'] → {1:0, 2:0}。特殊值 'all:0' 表示所有类别都映射到 0。"""
    out: Dict[int, int] = {}
    for s in specs:
        s = s.strip()
        if not s:
            continue
        if s.lower() == "all:0":
            return AllToZero()
        m = re.match(r"^(\d+)\s*:\s*(\d+)$", s)
        if not m:
            die("--remap 格式应为 '源类:目标类'，收到: %s" % s)
        out[int(m.group(1))] = int(m.group(2))
    if not out:
        out = {0: 0, 1: 0}
    return out


class AllToZero(dict):
    """所有类别都映射到 0（用户标注工具可能输出任意 label 值）。"""
    def get(self, k, default=None):
        return 0


# ==========================================================================
# 阶段 3：切片 → dataset_sliced（最终训练集）
# ==========================================================================
def stage_slice(args) -> Dict[str, object]:
    src = Path(args.dst_full)
    if not src.is_dir():
        die("找不到整帧集 %s，请先跑阶段 2" % src)
    dst = Path(args.dst_sliced)
    sw, sh = args.slice_width, args.slice_height
    tw, th = args.target_size

    if sw > tw or sh > th:
        die("切片 %dx%d 大于整帧 %dx%d" % (sw, sh, tw, th))

    scale = letterbox_scale(sw, sh, IMGSZ)
    px_native = args.min_mm / args.mm_per_px if args.mm_per_px > 0 else 0
    px_model = px_native * scale
    grid = slice_grid(tw, th, sw, sh, args.overlap, args.overlap)
    sizes = sorted({(x2 - x1, y2 - y1) for x1, y1, x2, y2 in grid})

    print("[3/3] 切片 → %s" % dst)
    print("  切片 %dx%d  overlap=%.2f  网格 %d 片  尺寸集合=%s"
          % (sw, sh, args.overlap, len(grid), sizes))
    print("  letterbox 到 %d → 放大 %.2fx；%.1fmm 原生 %.1fpx → 模型 %.1fpx（下限 %.0fpx）"
          % (IMGSZ, scale, args.min_mm, px_native, px_model, DETECT_FLOOR_PX))
    if sizes != [(sw, sh)]:
        die("切片网格产生了非完整尺寸的切片 %s —— 会破坏训练/推理尺度一致性" % sizes)
    if px_model < DETECT_FLOOR_PX:
        print("  [WARN] 当前档位下 %.1fmm 只有 %.1fpx，低于 %.0fpx 检测下限，会静默漏检！"
              % (args.min_mm, px_model, DETECT_FLOOR_PX))
        print("         你的 mm/px=%.3f 偏大（标定在包面远端了？）。可选：" % args.mm_per_px)
        print("           - 切片改 480x270（放大 %.2fx）" % letterbox_scale(480, 270, IMGSZ))
        print("           - 或把 IMGSZ 提到 1600（放大 %.2fx，需重训+重新导出）"
              % (1600 / max(sw, sh)))

    import numpy as np
    rng = random.Random(args.seed)
    stat = Counter()
    box_hist: List[float] = []

    for split in ("train", "val", "test"):
        img_dir = src / "images" / split
        if not img_dir.is_dir():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            img = imread_u(p)
            if img is None:
                stat["unreadable"] += 1
                continue
            H, W = img.shape[:2]
            if (W, H) != (tw, th):
                import cv2
                img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
                H, W = th, tw
            boxes = read_label(src / "labels" / split / (p.stem + ".txt"))
            # 归一化 → 像素 xyxy
            pboxes = []
            for c, cx, cy, w, h in boxes:
                pboxes.append([c, (cx - w / 2) * W, (cy - h / 2) * H,
                               (cx + w / 2) * W, (cy + h / 2) * H])

            xs = slice_starts(W, sw, args.overlap)
            ys = slice_starts(H, sh, args.overlap)
            idx = 0
            for y in ys:
                for x in xs:
                    x2s, y2s = min(W, x + sw), min(H, y + sh)
                    crop = img[y:y2s, x:x2s]
                    cw, ch = x2s - x, y2s - y
                    new_boxes = []
                    for c, bx1, by1, bx2, by2 in pboxes:
                        ix1, iy1 = max(bx1, x), max(by1, y)
                        ix2, iy2 = min(bx2, x2s), min(by2, y2s)
                        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                        area = max(1e-6, (bx2 - bx1) * (by2 - by1))
                        if inter / area < args.min_visibility:
                            continue
                        ncx = ((ix1 + ix2) / 2 - x) / cw
                        ncy = ((iy1 + iy2) / 2 - y) / ch
                        nw = (ix2 - ix1) / cw
                        nh = (iy2 - iy1) / ch
                        new_boxes.append((c, ncx, ncy, nw, nh))
                        box_hist.append(max(nw * cw, nh * ch))

                    name = "%s_s%03d" % (p.stem, idx)
                    idx += 1
                    if new_boxes:
                        stat["pos_slices"] += 1
                        stat["boxes"] += len(new_boxes)
                    elif rng.random() < args.keep_empty_ratio:
                        # 保留一部分纯背景切片：教会模型"干净上盖长什么样"，
                        # 这是抑制灰尘误检的关键，不能全丢
                        stat["neg_slices"] += 1
                    else:
                        stat["skipped_empty"] += 1
                        continue
                    imwrite_u(dst / "images" / split / (name + p.suffix), crop)
                    write_label(dst / "labels" / split / (name + ".txt"), new_boxes)
        print("    %s 完成，累计切片 %d" % (split, stat["pos_slices"] + stat["neg_slices"]))

    total_slices = stat["pos_slices"] + stat["neg_slices"]
    print("  切片总数 %d（含目标 %d / 纯背景 %d，背景保留率 %.2f），丢弃空片 %d，框 %d 个"
          % (total_slices, stat["pos_slices"], stat["neg_slices"],
             args.keep_empty_ratio, stat["skipped_empty"], stat["boxes"]))
    if total_slices < 500:
        print("  [WARN] 切片总数只有 %d，YOLO11s 从零学小目标偏少，" % total_slices)
        print("         建议原始整帧图至少 200 张（切完约 3200 片）。")

    if box_hist:
        box_hist.sort()
        def pct(q):
            return box_hist[min(len(box_hist) - 1, int(len(box_hist) * q))]
        tiny = sum(1 for b in box_hist if b < DETECT_FLOOR_PX)
        print("  切片内框长边像素: p10=%.1f p50=%.1f p90=%.1f；<%.0fpx 占 %.1f%%"
              % (pct(0.10), pct(0.50), pct(0.90), DETECT_FLOOR_PX,
                 100.0 * tiny / len(box_hist)))
    return {"slices": total_slices, "boxes": stat["boxes"],
            "scale": scale, "px_model": px_model}


# ==========================================================================
# 训练
# ==========================================================================
def cmd_train(args) -> None:
    data_yaml = Path(args.data)
    if not data_yaml.is_absolute():
        data_yaml = ROOT / data_yaml
    if not data_yaml.exists():
        die("找不到 %s" % data_yaml)

    # ---- 起飞前检查：data.yaml 指向的必须是切片集 ----
    txt = data_yaml.read_text(encoding="utf-8")
    m = re.search(r"^\s*path:\s*(.+?)\s*(?:#.*)?$", txt, re.M)
    dpath = Path(m.group(1).strip().strip("'\"")) if m else None
    if dpath is not None and not dpath.is_absolute():
        dpath = (data_yaml.parent / dpath).resolve()
    if dpath is None or not dpath.is_dir():
        die("data.yaml 的 path 指向的目录不存在: %s\n"
            "    先跑: python train.py prepare --src <原始数据目录>" % dpath)

    exp_w, exp_h = args.slice_width, args.slice_height
    probe = list((dpath / "images" / "train").glob("*"))[:20] if (dpath / "images" / "train").is_dir() else []
    if not probe:
        die("%s/images/train 是空的，先跑 prepare" % dpath)
    bad = []
    for p in probe:
        s = image_size(p)
        if s and s != (exp_w, exp_h):
            bad.append((p.name, s))
    if bad:
        die("训练集不是切片域！期望 %dx%d，实际 %s\n"
            "    data.yaml 现在指着整帧集，用它训练的话 5mm 只有 4.2px，模型学不到。\n"
            "    先跑: python train.py prepare --src <原始数据目录>"
            % (exp_w, exp_h, bad[:3]))
    print("[train] 起飞前检查通过: %s 是 %dx%d 切片域" % (dpath, exp_w, exp_h))

    scale = letterbox_scale(exp_w, exp_h, args.imgsz)
    px = (args.min_mm / args.mm_per_px) * scale if args.mm_per_px > 0 else 0
    print("[train] 尺度链: %.1fmm → 原生 %.1fpx → 模型 %.1fpx（下限 %.0fpx）"
          % (args.min_mm, args.min_mm / max(1e-9, args.mm_per_px), px, DETECT_FLOOR_PX))
    if 0 < px < DETECT_FLOOR_PX:
        die("当前档位下 %.1fmm 只有 %.1fpx，低于检测下限，训了也白训。" % (args.min_mm, px))
    if args.imgsz != IMGSZ:
        print("[train][WARN] --imgsz=%d 与常量 IMGSZ=%d 不一致，"
              "导出的 ONNX 会与 detect.py 的预期不符！" % (args.imgsz, IMGSZ))

    try:
        from ultralytics import YOLO
    except ImportError:
        die("没装 ultralytics。训练机执行: pip install ultralytics>=8.3.0")

    print("[train] 加载预训练权重 %s" % args.model)
    model = YOLO(args.resume) if args.resume else YOLO(args.model)

    train_kw = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=args.exist_ok,
        patience=args.patience,
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,
        # ---- 损失权重：偏向小目标 ----
        box=10.0,               # 默认 7.5，小目标定位误差占比大，调高
        cls=0.5,
        dfl=1.5,
        # ---- 几何增强 ----
        degrees=5.0,            # 工位相机固定，不做大幅旋转
        translate=0.1,
        scale=0.4,              # ★ 尺度抖动 ±40%。物理依据：斜视机位下包面
                                #   近端 mm/px≈0.6、远端≈1.2，本身就有约 2 倍跨度。
                                #   不要调到 0.5 以上，会造出部署域里没有的尺度。
        shear=0.0,
        perspective=0.0,
        flipud=0.0,             # 上盖不上下翻转
        fliplr=0.5,
        mosaic=0.5,             # 减弱 Mosaic，避免小目标被再缩小
        mixup=0.0,
        copy_paste=0.3,         # ★ Copy-Paste 对小目标涨点最有效
        close_mosaic=30,        # 最后 30 epoch 关掉 Mosaic，让模型见真实分布
        # ---- 色彩增强 ----
        hsv_h=0.01, hsv_s=0.5, hsv_v=0.4,
        # ---- 关键：尺度一致性 ----
        multi_scale=False,      # ★★ 必须 False。它把网络输入在 imgsz 的 ±50% 间抖动，
                                #   而 detect.py 推理固定在 1280。12.5px 的目标抖到
                                #   0.5x 只剩 6px —— 用推理时永远见不到的尺度训练。
        rect=False,
        cache=args.cache,
        seed=args.seed,
        deterministic=True,
        single_cls=False,       # 数据集本身已经是单类，不需要 ultralytics 再合类
        val=True, save=True, save_period=20, plots=True,
        verbose=True,
    )
    if args.override:
        for kv in args.override:
            if "=" not in kv:
                die("--override 格式应为 key=value，收到 %s" % kv)
            k, v = kv.split("=", 1)
            train_kw[k.strip()] = _coerce(v.strip())
            print("[train] override %s = %s" % (k.strip(), v.strip()))

    print("[train] 开始训练: epochs=%d imgsz=%d batch=%s device=%s"
          % (train_kw["epochs"], train_kw["imgsz"], train_kw["batch"], train_kw["device"]))
    t0 = time.time()
    results = model.train(**train_kw)
    print("[train] 完成，耗时 %.1f min" % ((time.time() - t0) / 60))

    best = Path(getattr(results, "save_dir", ROOT / "runs" / args.name)) / "weights" / "best.pt"
    if not best.exists():
        best = ROOT / "runs" / args.name / "weights" / "best.pt"
    print("[train] 最优权重: %s" % best)

    # 提醒：切片域的 val 指标是"乐观值"
    print("")
    print("=" * 70)
    print("★ 上面这组 mAP 是【切片域】指标，比真实部署表现乐观。")
    print("  因为切片把目标放大到 12.5px，且每张切片只含局部区域。")
    print("  真实的部署域指标必须跑端到端验收:")
    print("      python train.py export")
    print("      python train.py eval")
    print("  eval 会用 detect.py 的真实推理链路（切片+letterbox+NMS+mm过滤）")
    print("  在整帧 1920x1080 测试集上算 Precision / Recall / mAP50。")
    print("=" * 70)


def _coerce(v: str):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() in ("none", "null"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


# ==========================================================================
# 导出 ONNX
# ==========================================================================
def cmd_export(args) -> Path:
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = ROOT / weights
    if not weights.exists():
        die("找不到权重 %s\n    先跑: python train.py train" % weights)
    try:
        from ultralytics import YOLO
    except ImportError:
        die("没装 ultralytics: pip install ultralytics>=8.3.0")

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[export] %s → ONNX (imgsz=%d, opset=%d, simplify=%s, dynamic=%s, half=%s)"
          % (weights, args.imgsz, args.opset, args.simplify, args.dynamic, args.half))
    if args.half:
        print("[export][WARN] half=True 会导出 FP16 ONNX。CPU 上 onnxruntime 对 FP16")
        print("               支持不完整且往往更慢，你要的是精度最高，建议 --no-half。")
    if args.dynamic:
        print("[export][WARN] dynamic=True 会导出动态 shape，CPU 上比静态 shape 慢，")
        print("               且无法做图优化。本项目输入固定，建议保持静态。")

    model = YOLO(str(weights))
    exported = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset,
                            simplify=args.simplify, dynamic=args.dynamic,
                            half=args.half)
    exported = Path(exported)
    dst_onnx = out_dir / "best.onnx"
    shutil.copy2(str(exported), str(dst_onnx))
    print("[export] ONNX: %s (%.1f MB)" % (dst_onnx, dst_onnx.stat().st_size / 1e6))

    # ---- 写 model_meta.json：detect.py 靠它强制校验尺度一致性 ----
    sig = grid_signature(DEPLOY_W, DEPLOY_H, args.slice_width, args.slice_height,
                         args.overlap, args.overlap, args.imgsz)
    scale = letterbox_scale(args.slice_width, args.slice_height, args.imgsz)
    px_model = (args.min_mm / args.mm_per_px) * scale if args.mm_per_px > 0 else 0.0
    meta = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_weights": str(weights),
        "base_model": args.model_tag,
        "onnx_opset": args.opset,
        "onnx_simplify": bool(args.simplify),
        "onnx_half": bool(args.half),
        "imgsz": int(args.imgsz),
        "deploy_width": DEPLOY_W,
        "deploy_height": DEPLOY_H,
        "slice_width": int(args.slice_width),
        "slice_height": int(args.slice_height),
        "overlap_width": float(args.overlap),
        "overlap_height": float(args.overlap),
        "grid_signature": sig,
        "letterbox_scale": round(scale, 4),
        "mm_per_px": float(args.mm_per_px),
        "min_object_mm": float(args.min_mm),
        "min_object_px_model": round(px_model, 2),
        "detect_floor_px": DETECT_FLOOR_PX,
        "conf_threshold": float(args.conf),
        "iou_threshold": float(args.iou),
        "max_detections": int(args.max_det),
        "nc": len(CLASS_NAMES),
        "names": {str(k): v for k, v in CLASS_NAMES.items()},
    }
    meta_path = out_dir / "model_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print("[export] meta: %s" % meta_path)
    print("[export] grid_signature = %s" % sig)
    print("[export] 尺度链: %.1fmm → 原生 %.2fpx → 切片 letterbox %.2fx → 模型 %.2fpx"
          % (args.min_mm, args.min_mm / max(1e-9, args.mm_per_px), scale, px_model))
    if 0 < px_model < DETECT_FLOOR_PX:
        print("[export][WARN] 模型空间里 %.1fmm 只有 %.2fpx，低于 %.0fpx 下限！"
              % (args.min_mm, px_model, DETECT_FLOOR_PX))
    print("")
    print("[export] 把整个 %s 目录拷到工控机，然后:" % out_dir)
    print("           python detect.py --model %s --image <图片>" % dst_onnx)
    return dst_onnx


# ==========================================================================
# 端到端验收
# ==========================================================================
def cmd_eval(args) -> None:
    """用 detect.py 的真实推理链路在【整帧】测试集上算指标。

    这一步很关键：训练时打印的 mAP 是切片域的（目标被放大到 12.5px、每张切片
    只含局部），比真实部署表现乐观。只有跑完整帧端到端，数字才代表上线效果。
    """
    model = Path(args.model)
    if not model.is_absolute():
        model = ROOT / model
    if not model.exists():
        die("找不到 %s，先跑: python train.py export" % model)

    img_dir = Path(args.images) if args.images else ROOT / "dataset_full" / "images" / "test"
    lbl_dir = Path(args.labels) if args.labels else ROOT / "dataset_full" / "labels" / "test"
    if not img_dir.is_dir():
        die("验收图片目录不存在: %s" % img_dir)

    detect_mod = _import_detect()
    det = detect_mod.Detector(str(model), conf_thr=args.conf, iou_thr=args.iou,
                              mm_per_px_override=args.mm_per_px or None)
    print("[eval] 端到端验收: %d 张整帧图，conf=%.2f iou=%.2f"
          % (len(list(img_dir.glob("*"))), args.conf, args.iou))
    print("[eval] 切片网格: %s" % det.grid_summary())

    preds: Dict[str, List[Tuple[float, Tuple[float, float, float, float]]]] = {}
    gts: Dict[str, List[Tuple[float, float, float, float]]] = {}
    t0 = time.time()
    lat: List[float] = []
    imgs = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
    for i, p in enumerate(imgs, 1):
        img = imread_u(p)
        if img is None:
            print("  [WARN] 读不了 %s" % p.name)
            continue
        t1 = time.time()
        dets = det.detect(img)
        lat.append((time.time() - t1) * 1000.0)
        preds[p.stem] = [(d["score"], tuple(d["bbox"])) for d in dets]
        H, W = img.shape[:2]
        gb = []
        for c, cx, cy, w, h in read_label(lbl_dir / (p.stem + ".txt")):
            gb.append(((cx - w / 2) * W, (cy - h / 2) * H,
                       (cx + w / 2) * W, (cy + h / 2) * H))
        gts[p.stem] = gb
        if i % 20 == 0 or i == len(imgs):
            print("  进度 %d/%d  累计 %.1fs（%.2fs/帧）"
                  % (i, len(imgs), time.time() - t0, (time.time() - t0) / i))

    # 首帧含 ORT 线程池/图优化的冷启动开销，远高于稳态，算均值时剔掉
    steady = lat[1:] if len(lat) > 3 else lat
    avg_ms = sum(steady) / len(steady) if steady else 0.0
    metrics = prf_map50(preds, gts, iou_thr=0.5, avg_ms=avg_ms)
    print("")
    print("=" * 70)
    print("端到端验收结果（整帧 1920x1080，%d 张）" % len(preds))
    print("=" * 70)
    print("  Precision @IoU0.5 : %.4f" % metrics["precision"])
    print("  Recall    @IoU0.5 : %.4f" % metrics["recall"])
    print("  F1                : %.4f" % metrics["f1"])
    print("  mAP50             : %.4f" % metrics["map50"])
    print("  TP=%d  FP=%d  FN=%d" % (metrics["tp"], metrics["fp"], metrics["fn"]))
    print("  平均单帧延迟      : %.0f ms（本机、%d 帧稳态均值，已剔除首帧冷启动）"
          % (metrics["avg_ms"], len(steady)))
    print("                        ★ 这是【当前这台机器】的数字，不是 2 核工控机的。")
    print("                          工控机上的真实延迟要在现场跑:")
    print("                            python detect.py --model deploy/best.onnx "
          "--dir <图目录> --limit 20")
    print("")
    print("  判读:")
    print("    Recall 是本项目的第一指标（漏检异物 = 装车风险，误检只是多看一眼）。")
    print("    Recall >= 0.90 且 Precision >= 0.70 基本可用；")
    print("    Recall <  0.80 说明还差得远，按优先级补：数据量 → Copy-Paste →")
    print("    实测 mm/px 确认档位 → 换 YOLO11m。")
    print("    FP 高且都落在灰尘/污渍上 → 补【有灰尘但无异物】的负样本重训。")
    print("    FN 集中在画面右/下边缘 → 切片网格分叉了，查 detect.py 启动时的")
    print("    grid_signature 校验有没有报警。")
    print("=" * 70)


def _import_detect():
    """按文件路径导入同目录的 detect.py（避免 sys.path 污染）。"""
    import importlib.util
    p = ROOT / "detect.py"
    if not p.exists():
        die("找不到 %s" % p)
    spec = importlib.util.spec_from_file_location("fod_detect", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prf_map50(preds, gts, iou_thr=0.5, fbest=None, avg_ms=0.0):
    """VOC 风格 all-point AP@0.5 + P/R/F1（取 F1 最优阈值处的 P/R）。"""
    all_scores: List[Tuple[float, str, Tuple[float, float, float, float]]] = []
    for stem, ds in preds.items():
        for s, b in ds:
            all_scores.append((s, stem, b))
    all_scores.sort(key=lambda t: -t[0])

    gt_remain = {k: len(v) for k, v in gts.items()}
    total_gt = sum(gt_remain.values())
    matched: Dict[Tuple[str, int], bool] = {}
    tp_list, fp_list = [], []
    for s, stem, b in all_scores:
        gl = gts.get(stem, [])
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gl):
            iou = _iou(b, g)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and not matched.get((stem, best_j)):
            matched[(stem, best_j)] = True
            tp_list.append(1); fp_list.append(0)
        else:
            tp_list.append(0); fp_list.append(1)

    tp_cum, fp_cum = 0, 0
    recalls, precisions, scores = [], [], []
    f1s = []
    for tp, fp, (s, _, _) in zip(tp_list, fp_list, all_scores):
        tp_cum += tp; fp_cum += fp
        r = tp_cum / total_gt if total_gt else 0.0
        p = tp_cum / max(1, tp_cum + fp_cum)
        recalls.append(r); precisions.append(p); scores.append(s)
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)

    # all-point interpolation
    ap = 0.0
    if recalls:
        mrec = [0.0] + recalls + [1.0]
        mpre = [0.0] + precisions + [0.0]
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    if f1s:
        bi = max(range(len(f1s)), key=lambda i: f1s[i])
        best_p, best_r, best_f1 = precisions[bi], recalls[bi], f1s[bi]
    else:
        best_p = best_r = best_f1 = 0.0
    return {"precision": best_p, "recall": best_r, "f1": best_f1, "map50": ap,
            "tp": sum(tp_list), "fp": sum(fp_list),
            "fn": total_gt - sum(tp_list),
            "avg_ms": float(avg_ms),
            "best_conf": scores[max(range(len(f1s)), key=lambda i: f1s[i])] if f1s else 0.0}


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = ua + ub - inter
    return inter / denom if denom > 0 else 0.0


# ==========================================================================
# 一条龙
# ==========================================================================
def cmd_all(args) -> None:
    cmd_prepare(args)
    cmd_train(args)
    args.weights = str(ROOT / "runs" / args.name / "weights" / "best.pt")
    cmd_export(args)
    print("")
    print("[all] 全流程完成。下一步跑端到端验收:")
    print("        python train.py eval")


def cmd_prepare(args) -> None:
    if args.from_stage <= 1:
        stage_unify(args)
    else:
        print("[1/3] 跳过（--from-stage %d）" % args.from_stage)
    if args.from_stage <= 2:
        stage_clean(args)
    else:
        print("[2/3] 跳过（--from-stage %d）" % args.from_stage)
    if args.from_stage <= 3:
        stage_slice(args)
    print("")
    print("[prepare] 完成。")
    print("  整帧验收集 : %s   ← train.py eval 用这个" % args.dst_full)
    print("  切片训练集 : %s   ← data.yaml 指向这个" % args.dst_sliced)
    print("  下一步     : python train.py train")


def die(msg: str) -> None:
    print("[ERROR] %s" % msg)
    raise SystemExit(1)


# ==========================================================================
# CLI
# ==========================================================================
def add_common(p):
    p.add_argument("--target-size", nargs=2, type=int, default=[DEPLOY_W, DEPLOY_H],
                   metavar=("W", "H"), help="部署域分辨率，默认 1920 1080")
    p.add_argument("--slice-width", type=int, default=SLICE_W,
                   help="切片宽，默认 %d（精度优先档，放大 %.1fx）"
                        % (SLICE_W, letterbox_scale(SLICE_W, SLICE_H, IMGSZ)))
    p.add_argument("--slice-height", type=int, default=SLICE_H, help="切片高，默认 %d" % SLICE_H)
    p.add_argument("--overlap", type=float, default=OVERLAP_W, help="切片重叠率，默认 0.2")
    p.add_argument("--mm-per-px", type=float, default=MM_PER_PX,
                   help="★ 物理标定值 mm/像素，必须按包面【最远端】实测。默认 %.2f 是估计值" % MM_PER_PX)
    p.add_argument("--min-mm", type=float, default=MIN_MM, help="最小检测口径 mm，默认 5.0")


def build_parser():
    ap = argparse.ArgumentParser(
        description="电池包上盖异物检测 —— 训练端（数据准备 / 训练 / 导出 ONNX / 验收）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("---------------------------------------------------------------------------")[1]
        if "---" in __doc__ else None)
    sub = ap.add_subparsers(dest="cmd")

    # prepare
    p = sub.add_parser("prepare", help="数据准备：统一分辨率 → 清洗标签 → 切片")
    p.add_argument("--src", help="原始标注数据目录（images/ + labels/，YOLO txt）")
    p.add_argument("--work", default=str(ROOT / "work"), help="中间产物目录")
    p.add_argument("--dst-full", default=str(ROOT / "dataset_full"),
                   help="整帧验收集输出目录（1920x1080）")
    p.add_argument("--dst-sliced", default=str(ROOT / "dataset_sliced"),
                   help="切片训练集输出目录（data.yaml 指向这里）")
    p.add_argument("--remap", nargs="*", default=["1:0"],
                   help="类别映射，默认 1:0（你的标注工具输出 label=1）。"
                        "所有类都并成一类用 all:0")
    p.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                   metavar=("TRAIN", "VAL", "TEST"), help="划分比例，默认 8:1:1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-visibility", type=float, default=0.3,
                   help="目标与切片的交并比低于此值则丢弃该切片内的这个目标")
    p.add_argument("--keep-empty-ratio", type=float, default=0.15,
                   help="保留多少比例的纯背景切片（抑制灰尘误检的关键，别设 0）")
    p.add_argument("--no-min-mm-filter", action="store_true",
                   help="不按 mm 口径过滤小框（不建议，会把学不到的目标喂给模型）")
    p.add_argument("--from-stage", type=int, choices=[1, 2, 3], default=1,
                   help="从第 N 阶段开始（改了 mm-per-px 用 2，只改切片参数用 3）")
    p.add_argument("--force", action="store_true", help="阶段 1 强制重新缩放已存在的图")
    add_common(p)
    p.set_defaults(func=cmd_prepare)

    # train
    p = sub.add_parser("train", help="训练 YOLO11s（切片域，imgsz=1280）")
    p.add_argument("--data", default="data.yaml")
    p.add_argument("--model", default=BASE_MODEL, help="预训练权重，默认 %s" % BASE_MODEL)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=16, help="切片图小，24G 显存可到 32")
    p.add_argument("--imgsz", type=int, default=IMGSZ,
                   help="★ 必须等于 detect.py 的 letterbox 边长（%d）" % IMGSZ)
    p.add_argument("--device", default="0", help="GPU 序号；CPU 训练填 cpu（不推荐）")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=60, help="早停轮数")
    p.add_argument("--name", default="yolo11s_slice640x360_v1", help="run 名")
    p.add_argument("--exist-ok", action="store_true", help="覆盖同名 run 目录")
    p.add_argument("--resume", default=None, help="断点续训的 last.pt 路径")
    p.add_argument("--cache", action="store_true", help="内存充足时缓存图片加速 IO")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--override", nargs="*", default=[],
                   help="临时覆盖任意超参，如 epochs=500 mosaic=0.0")
    add_common(p)
    p.set_defaults(func=cmd_train)

    # export
    p = sub.add_parser("export", help="导出 ONNX + model_meta.json")
    p.add_argument("--weights", default=str(ROOT / "runs" / "yolo11s_slice640x360_v1" / "weights" / "best.pt"))
    p.add_argument("--out", default=str(ROOT / "deploy"), help="输出目录（拷到工控机）")
    p.add_argument("--imgsz", type=int, default=IMGSZ, help="★ 必须与训练时一致")
    p.add_argument("--opset", type=int, default=12, help="ONNX opset，12 兼容性最好")
    p.add_argument("--simplify", dest="simplify", action="store_true", default=True,
                   help="onnxslim 化简图（默认开）")
    p.add_argument("--no-simplify", dest="simplify", action="store_false")
    p.add_argument("--dynamic", action="store_true", help="动态 shape（不建议，CPU 上更慢）")
    p.add_argument("--half", action="store_true", help="FP16（CPU 上不建议）")
    p.add_argument("--conf", type=float, default=0.25, help="写入 meta 的默认置信度阈值")
    p.add_argument("--iou", type=float, default=0.55, help="写入 meta 的默认 NMS IoU 阈值")
    p.add_argument("--max-det", type=int, default=200)
    p.add_argument("--model-tag", default="yolo11s", help="写进 meta 的模型标识")
    add_common(p)
    p.set_defaults(func=cmd_export)

    # eval
    p = sub.add_parser("eval", help="端到端验收：用 detect.py 在整帧测试集上算 P/R/mAP50")
    p.add_argument("--model", default=str(ROOT / "deploy" / "best.onnx"))
    p.add_argument("--images", default=None, help="整帧测试图目录，默认 dataset_full/images/test")
    p.add_argument("--labels", default=None, help="对应标签目录，默认 dataset_full/labels/test")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.55)
    p.add_argument("--mm-per-px", type=float, default=None, help="覆盖 meta 里的标定值")
    p.set_defaults(func=cmd_eval)

    # all
    p = sub.add_parser("all", help="一条龙：prepare → train → export")
    p.add_argument("--src", help="原始标注数据目录")
    p.add_argument("--work", default=str(ROOT / "work"))
    p.add_argument("--dst-full", default=str(ROOT / "dataset_full"))
    p.add_argument("--dst-sliced", default=str(ROOT / "dataset_sliced"))
    p.add_argument("--remap", nargs="*", default=["1:0"])
    p.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-visibility", type=float, default=0.3)
    p.add_argument("--keep-empty-ratio", type=float, default=0.15)
    p.add_argument("--no-min-mm-filter", action="store_true")
    p.add_argument("--from-stage", type=int, choices=[1, 2, 3], default=1)
    p.add_argument("--force", action="store_true")
    p.add_argument("--data", default="data.yaml")
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=IMGSZ)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--name", default="yolo11s_slice640x360_v1")
    p.add_argument("--exist-ok", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--cache", action="store_true")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--out", default=str(ROOT / "deploy"))
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--simplify", dest="simplify", action="store_true", default=True)
    p.add_argument("--no-simplify", dest="simplify", action="store_false")
    p.add_argument("--dynamic", action="store_true")
    p.add_argument("--half", action="store_true")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.55)
    p.add_argument("--max-det", type=int, default=200)
    p.add_argument("--model-tag", default="yolo11s")
    add_common(p)
    p.set_defaults(func=cmd_all)

    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    if not getattr(args, "cmd", None):
        ap.print_help()
        print("")
        print("最常用的第一条命令:")
        print("    python train.py all --src /path/to/raw")
        raise SystemExit(1)

    # 归一化 target_size
    if hasattr(args, "target_size"):
        tw, th = args.target_size
        if (tw, th) != (DEPLOY_W, DEPLOY_H):
            print("[WARN] --target-size %dx%d 与部署域 %dx%d 不一致。"
                  % (tw, th, DEPLOY_W, DEPLOY_H))
            print("       摄像头出图已确认恒为 1920x1080，除非你换了相机否则别改。")

    # prepare / all 必须有 --src
    if args.cmd in ("prepare", "all") and args.from_stage == 1 and not getattr(args, "src", None):
        die("prepare 需要 --src <原始数据目录>")

    print("=" * 70)
    print("电池包上盖异物检测 · 训练端")
    if hasattr(args, "target_size"):
        print("  部署域   : %dx%d（摄像头出图，已确认）"
              % (args.target_size[0], args.target_size[1]))
    if hasattr(args, "slice_width"):
        print("  切片     : %dx%d overlap=%.2f → %d 片，letterbox %d = 放大 %.2fx"
              % (args.slice_width, args.slice_height, args.overlap,
                 len(slice_grid(args.target_size[0], args.target_size[1],
                                args.slice_width, args.slice_height,
                                args.overlap, args.overlap)),
                 IMGSZ, letterbox_scale(args.slice_width, args.slice_height, IMGSZ)))
        print("  标定     : mm/px=%.3f  最小口径=%.1fmm" % (args.mm_per_px, args.min_mm))
    print("=" * 70)
    args.func(args)


if __name__ == "__main__":
    main()
