# CtrlVQ

多模态知识图谱实体量化与两阶段 LLM 训练的独立代码整理版本。

本版以现有项目的核心源码为基础，保留原始算法与实验口径，增加统一配置、命令入口、运行记录和恢复管理。不是此前 `revision/` 中重写训练或评价逻辑的版本。

## 包含内容

- 原始文本、图像、结构嵌入读取和多模态融合。
- 原始 FSQ / adapter / 重构、多样性与对比损失实现。
- 原始实体编码、碰撞 EID、候选提示和 LLM 数据生成。
- 原始 Stage 1 embedding / lm_head 训练与 Stage 2 LoRA 微调。
- 原始 `test_metrics_fast.py` 评分及 MRR、MR、Hits@1/3/10。
- 三个数据集的 JSON 配置、统一命令行、运行 manifest 和源码来源记录。

不附带数据、模型权重、日志、旧实验输出或后台调度程序。没有自动启动训练的脚本。

## 目录

```text
CtrlVQ/
├── ctrlvq/
│   ├── cli.py                  # 统一命令行
│   ├── configuration.py        # 配置、参数与路径
│   ├── pipeline.py             # 前台流水线、运行记录、断点管理
│   ├── entry.py                # 配置桥接与固定随机种子
│   └── core/                   # 从原项目保留的核心实现
│       ├── fsq_quantization/
│       ├── multimodal_embedding/
│       ├── gen_dataset4LLM/
│       ├── configs/
│       ├── finetuning_token.py
│       ├── finetuning_LoRA.py
│       └── test_metrics_fast.py
├── configs/
│   ├── mkg-w.json
│   ├── mkg-y.json
│   ├── mmkb-db15k.json
│   └── local/                  # 当前机器可用的路径配置
├── docs/
│   ├── PROTOCOL.md
│   ├── PROVENANCE.json
│   └── ENVIRONMENT.txt
├── pyproject.toml
├── requirements.txt
└── MANIFEST.sha256
```

## 使用现有环境

可以直接在源码目录运行，不需要重新安装依赖或复制大文件：

```bash
cd /home/daci3090/data3090/PersonalFile/shifobo/code/CtrlVQ
conda activate shifobo_moe

# 仅展示命令，不创建训练任务，不导入 PyTorch，不使用 GPU。
python -m ctrlvq plan --config configs/local/mkg-w.json
```

`configs/local/` 引用当前机器已有的数据、基础模型和结构嵌入。新输出写入本目录的 `outputs/`，不会覆盖原项目。

以下命令只有手动执行时才启动对应任务：

```bash
# 执行完整流水线；默认只使用配置中的一张卡。
python -m ctrlvq run --config configs/local/mkg-w.json

# 分阶段执行。
python -m ctrlvq run --config configs/local/mkg-w.json --stage quantize
python -m ctrlvq run --config configs/local/mkg-w.json --stage data
python -m ctrlvq run --config configs/local/mkg-w.json --stage stage1
python -m ctrlvq run --config configs/local/mkg-w.json --stage stage2
python -m ctrlvq run --config configs/local/mkg-w.json --stage evaluate

# 指定物理 GPU；多张卡用于原生 DDP。
python -m ctrlvq run --config configs/local/mkg-w.json --stage stage1 --gpus 0,1

# 查看本次整理版本的输出状态。
python -m ctrlvq status --config configs/local/mkg-w.json

# 仅对已有完整 training_state.pt 的 Stage1 / Stage2 断点恢复。
python -m ctrlvq run --config configs/local/mkg-w.json --stage stage1 --resume
```

如使用可安装命令入口：

```bash
pip install -e . --no-deps
ctrlvq plan --config configs/local/mkg-w.json
```

此操作不安装或升级现有依赖。需要新建环境时，依赖见 `requirements.txt`；该文件是原项目的最低版本要求，不是已完整验证的版本锁。

## 其他机器

编辑 JSON 中的路径，或者为通用配置设置三个环境变量：

```bash
export CTRLVQ_DATA_ROOT=/path/to/data
export CTRLVQ_STRUCTURE_ROOT=/path/to/KnowledgeGraphEmbedding/models
export CTRLVQ_MODEL_PATH=/path/to/Qwen3-1.7B
python -m ctrlvq plan --config configs/mkg-w.json
```

路径相对于 JSON 文件所在目录解析，支持 `${ENV_VAR}`。`model_path` 使用已下载的本地模型，不自动下载模型。

数据目录至少需要：

```text
MKG-W/
├── train.txt                   # head<TAB>relation<TAB>tail
├── valid.txt
├── test.txt
├── one_hop.json                # 原项目准备的一阶邻居文件
├── relation_names.json         # 可选
├── ent_links
├── MKG_W_description_sentences.h5
└── MKG_W_img_BEIT_16-224.h5
```

另外两个数据集的 HDF5 和实体链接文件名见各自 JSON。结构嵌入单独配置 `structural_emb_path`。保留原有 ID 映射规则；不重新生成或重排一阶邻居和结构嵌入。

## 输出与中断

```text
outputs/<dataset>/seed_42/
├── quantizer/
├── datasets/
├── stage1/
├── stage2/
├── evaluation/metrics.json
├── logs/
└── metadata/
    ├── resolved_config.json
    ├── run.json
    ├── status.json
    └── <stage>_complete.json
```

运行保持在前台。`Ctrl+C` / `SIGTERM` 中断时保留已保存的 checkpoint；尚未保存的训练步数不会保留。不会自动重启或恢复。

同一输出目录使用运行锁。异常强制退出留下锁时，先检查 `.run.lock` 中 PID 是否仍在运行，再手动删除失效锁。改变模型、数据、训练参数、DDP 卡数或核心代码时，使用新的 `output_dir`。恢复时可以改变物理卡号，但卡数应保持一致。

原生训练脚本可能捕获异常后返回退出码 0；外层同时检查错误日志与关键输出文件，避免将这种情况记录为训练成功。

## 实验口径

**默认评价保持原始脚本口径，不是 filtered evaluation，也不是完整候选自回归似然评价。** 训练、编码和评分的具体兼容性事项见 [docs/PROTOCOL.md](docs/PROTOCOL.md)。

本版没有替换成“只训练新增 token 行”“仅答案监督”“头尾双向数据”或新评价器。只整理目录、配置及执行接口。普通 FSQ / VQ 对比与模态、融合消融没有混入主流程。

历史日志恢复的参数已写入三个配置；历史 DDP 卡数和部分量化设置没有完整归档，因此不声称能够直接复现论文表格数值。尤其 MKG-W Stage2 原日志预算为 200000 步，运行前请查看配置和计划。

## 交付状态

本次没有启动训练、没有使用 GPU，也没有运行实现测试。本包已去除 Python 注释和 docstring；已进行静态语法与可执行逻辑结构核对。核心来源文件及去注释后文件的 SHA-256 记录在 `docs/PROVENANCE.json`；本目录文件校验和见 `MANIFEST.sha256`。源项目保持原样。
