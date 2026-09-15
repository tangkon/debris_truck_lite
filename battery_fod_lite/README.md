# battery_fod_lite —— 电池包上盖异物检测（精简三文件工程 + 文档）

换电站工况：拆卸下来的电池包上盖过站时，由固定工业相机（1920×1080）拍一帧，
检测上盖表面 **≥5mm** 的异物（石子、螺栓、螺母、煤块等，统一一类 `foreign`）。
部署端是 **2 核 CPU、≤4GB 内存、无 GPU** 的工控机，且不能影响同机其它服务。

本目录是"精度优先"的精简工程：训练端只暴露 `train.py`，部署端只有一个
`detect.py` + 一个 ONNX；`detect_test.py` 是开发机全速测试端，**禁止上工控机**。

---

## 文件清单

| 文件 | 角色 | 运行位置 |
|---|---|---|
| `train.py` | 数据准备 / 训练 / 导出 ONNX+meta / 端到端验收 | 训练机（有 GPU） |
| `data.yaml` | Ultralytics 数据集描述（指向切片训练集） | 训练机 |
| `detect.py` | 部署推理：16 切片串行 + 2 核 confinement | **工控机** |
| `detect_test.py` | 开发机全速测试：多进程/弹窗/HTML 报告/A-B 比对 | 开发机（**禁上工控机**） |
| `README.md` | 本文件：导航 + 速览 | — |
| `技术方案.md` | 完整技术方案：架构、尺度链、阈值体系、标定、ROI 分析、验收 | — |
| `标注规范.md` | 标注口径与质检清单（给标注员/数据负责人） | — |
| `部署与操作手册.md` | 命令示例、标定操作、上线 checklist、故障排查 | — |

---

## 三分钟速览

```bash
# 训练机
python train.py prepare --src <原始标注目录> --mm-per-px <实测值> --min-mm 4.0
python train.py train                      # YOLO11s，切片域 imgsz=1280
python train.py export --mm-per-px <实测值> --min-mm 5.0 --out deploy
python train.py eval                       # 整帧端到端 P/R/mAP50

# 工控机（把整个 deploy/ 拷过去：best.onnx + model_meta.json 必须同目录）
python detect.py --model deploy/best.onnx --dir <图片目录> --cores 0,1

# 开发机快速看效果（不锁核、多进程、可出 HTML 报告）
python detect_test.py --model deploy/best.onnx --dir <图片目录> --report report.html
```

---

## 核心结论速览（细节见《技术方案.md》）

- **尺度链**：mm/px=0.8 时，5mm = 原生 6.25px = 切片 letterbox 后 **12.5px**（模型输入空间）。
  整帧直接送 1280 只有 4.17px，不可检；所以必须切片 640×360 → 1280（放大 2.0×，16 片）。
- **三个尺寸阈值不能混**：
  - 物理检测下限（本档位）：**3.2mm**（模型空间 8px = YOLO P3 stride 下限）。
  - 训练/标注截断（`prepare --min-mm`）：建议 **4.0mm**，这是"有利于训练"的最小值。
  - 业务门控（`export --min-mm` → meta → detect 丢弃小于它的框）：保持 **5.0mm**。
- **mm/px 必须实测**：默认 0.8 只是估计值。用 `detect.py --calibrate --ref-mm --ref-px`
  在真实帧上测，参照物放在**上盖表面**（异物停留的那个平面），中心+四角各测一次。
  本档位（5mm 口径）mm/px 上限 1.25；超过就要换更小切片档。
- **训练尺度 = 推理尺度**是本项目第一硬约束：`model_meta.json` 里的 `grid_signature`
  与 detect.py 本地重算不一致时直接拒绝运行（exit 2），防止"静默漏检"。
- **detect_test.py 的数字不代表工控机**：它不锁核、吃满 CPU，只用于开发期快速看效果
  和做 `--compare`（2 核 vs 全速的逐框 A/B 比对）。

---

## 文档导航

- 想知道"为什么这么设计"（切片、ONNX、2 核、ROI 要不要加）→ 《技术方案.md》
- 想知道"标注怎么标、标到多小"→ 《标注规范.md》
- 想知道"每条命令怎么敲、上线要检查什么、报错了怎么办"→ 《部署与操作手册.md》
  （附录 A/B/C 是 train.py / detect.py / detect_test.py 的**完整 CLI 参考**：每个参数、默认值、退出码）
