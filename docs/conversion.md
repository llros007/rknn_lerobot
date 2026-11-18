# 从 PyTorch 重新转换 RKNN

这是可选流程。只想运行板端服务时，直接下载已编译的 RKNN，无需安装 LeRobot、PyTorch 或转换工具。

## 已验证环境

PC：Linux x86_64、Python 3.12.12、LeRobot 0.6.1、PyTorch 2.10.0、torchvision 0.25.0；
ONNX 1.16.1、ONNX Runtime 1.20.1、onnxsim 0.4.36、RKNN Toolkit2 2.3.2。

本项目验证的 LeRobot 上游 commit：`7e241bd630a3719a56157a497ce5d08f244784f1`。
LeRobot 在仓库外单独安装，不复制到本仓库。

RKNN Toolkit2 2.3.2 声明的 PyTorch 依赖范围与本环境的 2.10 不一致。本项目使用 `--no-deps` 保留已有训练环境，实际只走 **ONNX 导入 RKNN** 路径，并完成数值与闭环校验；这不表示其 PyTorch 导入器支持 2.10。

## 1. 准备独立转换环境

若已有训练时的 conda `le312` 环境（包含上述 LeRobot、PyTorch、torchvision 及数据解码依赖），推荐基于它创建独立转换环境：

```bash
conda activate le312
# 在 rknn_lerobot 根目录执行，python 指向已安装 LeRobot 的 conda 环境
uv venv --python "$(command -v python)" --system-site-packages .venv-export
uv pip install --python .venv-export/bin/python --no-deps -r requirements-conversion.txt
.venv-export/bin/python -c 'import torch, onnx; from rknn.api import RKNN; print(torch.__version__)'
```

没有 LeRobot 环境时，可在仓库外克隆指定版本，然后安装到独立 Python 3.12 环境中：

```bash
# 以下从 rknn_lerobot 根目录开始；LeRobot 克隆在相邻目录，不进入本 Git 仓库
cd ..
git clone https://github.com/huggingface/lerobot.git lerobot-upstream
git -C lerobot-upstream checkout 7e241bd630a3719a56157a497ce5d08f244784f1
cd rknn_lerobot
uv venv --python 3.12 .venv-export
uv pip install --python .venv-export/bin/python 'torch==2.10.0' 'torchvision==0.25.0' -e ../lerobot-upstream
uv pip install --python .venv-export/bin/python --no-deps -r requirements-conversion.txt
```

LeRobot 的 TorchCodec 数据解码需要与 PyTorch 和系统 FFmpeg 版本相匹配。导出脚本会实际解码训练数据来校验，开始前先确认当前 LeRobot 环境可以读取数据。运行库冲突时应按自身平台创建兼容环境，不要混装板端 aarch64 wheel。

## 2. 下载检查点与准备数据

```bash
.venv-export/bin/python scripts/download_models.py --format pytorch
```

下载后 `checkpoint/` 含 safetensors 权重、config、preprocessor、postprocessor 和对应统计量。
导出验证使用公开 LeRobot 数据集 `aractingi/franka_sim_pick_lift_6`，脚本默认由 LeRobot 下载并缓存。
如果已有本地数据集，传入 `--dataset-root /PATH/TO/DATASET`。

本仓库不包含训练数据。脚本选择 12 个示范样本，生成本地 `samples/*.npz` 与 PyTorch 参考输出，不会自动上传。

## 3. 导出静态 ONNX 并验证

```bash
mkdir -p tmp cache logs reports onnx samples models
OMP_NUM_THREADS=4 .venv-export/bin/python scripts/export_act.py
# 或显式指定已有数据根目录：
# .venv-export/bin/python scripts/export_act.py --dataset-root /PATH/TO/DATASET
.venv-export/bin/python scripts/verify_reference.py
```

生成 `onnx/act_raw.onnx`、简化后的 `onnx/act.onnx` 和验证报告。
导出图使用 ONNX opset 13，batch=1、双图 `[1,3,128,128]`、状态 `[1,18]`、输出 `[1,50,4]`。

包装器读取检查点保存的统计量，将输入标准化和动作反标准化嵌入图中，推理潜变量固定为零。
`verify_reference.py` 另用原始 LeRobot preprocessor → ACTPolicy → postprocessor 流水线对比 ONNX。

## 4. 编译 RK3588 FP16

```bash
.venv-export/bin/python scripts/convert_rknn.py --target rk3588 --verify
```

生成：

```text
models/act_franka_rk3588_fp16.rknn
models/act_franka_rk3588_fp16.abi.json
reports/rknn_rk3588_validation.json
```

`target_platform="rk3588"`，`float_dtype="float16"`，`optimization_level=3`，`build(do_quantization=False)`。
该目标用于本次 RK3588S 测试。其他芯片需要重新编译与板端验证。

`--verify` 使用 Toolkit2 的 PC 模拟器做精度比对，**不是板端真实 NPU 实测**。
如需在 PC 模拟器运行 MuJoCo，可安装 `requirements-pc.txt` 后使用：

```bash
uv pip install --python .venv-export/bin/python -r requirements-pc.txt
MUJOCO_GL=egl .venv-export/bin/python scripts/convert_rknn.py \
  --target rk3588 --verify --eval-episodes 3
```

## 5. 真实板端验证

按主 README 安装服务，将自己的 RKNN 文件放入板端 `models/`，启动服务。
PC 执行 `scripts/smoke_client.py` 与 `scripts/eval_mujoco.py --backend http`。

RKNNLite 的图像原生布局可能由编译器调整。`runtime.py` 显式传入三个 `data_format="nchw"`，
让运行库按模型元数据处理布局；不要仅根据 ONNX 的形状假设 RKNN 内部布局。

## 使用原始 PyTorch 模型

模型库 `pytorch/` 是原始 LeRobot 检查点，未包含 RKNN 导出包装器。
通过 `ACTPolicy.from_pretrained()` 加载时，同时使用保存的 pre/postprocessor；模型卡提供 CPU 加载示例。
不要将 RKNN 已嵌入归一化的调用方式直接套用到 PyTorch 权重。

## 产物管理

`checkpoint/ models/ onnx/ samples/ reports/ logs/ cache/ tmp/ .venv*/` 都被 `.gitignore` 排除。
编译模型与 PyTorch 权重发布到 Hugging Face，代码仓库只保留可复现脚本和说明。
