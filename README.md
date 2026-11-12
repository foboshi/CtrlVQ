# CtrlVQ

CtrlVQ quantizes multimodal knowledge graph entities and trains an LLM in two stages. This standalone package preserves the core algorithms and evaluation procedure from the original project and adds a shared configuration, a command-line interface, run metadata, and checkpoint management.

## Contents

- Load text, image, and structural embeddings and fuse them into multimodal representations.
- Quantize representations with FSQ and an adapter, using reconstruction, diversity, and contrastive losses.
- Encode entities, assign entity IDs to resolve collisions, construct candidate prompts, and generate LLM training data.
- Train the embeddings and language model head in Stage 1 and fine-tune with LoRA in Stage 2.
- Score predictions with `test_metrics_fast.py` and report MRR, MR, and Hits@1/3/10.

## Directory Structure

```text
CtrlVQ/
├── ctrlvq/
│   ├── cli.py                  # Command-line interface
│   ├── configuration.py        # Configuration, arguments, and paths
│   ├── pipeline.py             # Pipeline, run metadata, and checkpoint management
│   ├── entry.py                # Configuration bridge and random seed setup
│   └── core/                   # Core implementation
│       ├── fsq_quantization/
│       ├── multimodal_embedding/
│       ├── gen_dataset4LLM/
│       ├── configs/
│       ├── finetuning_token.py
│       ├── finetuning_LoRA.py
│       └── test_metrics_fast.py
├── pyproject.toml
└── requirements.txt
```

## Using an Existing Environment

Run the commands from the repository root with your existing Python environment activated. The examples assume that you have a JSON configuration at `configs/local/mkg-w.json` pointing to your dataset, base model, and structural embeddings.

```bash
# Preview the commands without starting training or using a GPU.
python -m ctrlvq plan --config configs/local/mkg-w.json

# Run the full pipeline. By default, it uses one GPU from the configuration.
python -m ctrlvq run --config configs/local/mkg-w.json

# Run individual stages.
python -m ctrlvq run --config configs/local/mkg-w.json --stage quantize
python -m ctrlvq run --config configs/local/mkg-w.json --stage data
python -m ctrlvq run --config configs/local/mkg-w.json --stage stage1
python -m ctrlvq run --config configs/local/mkg-w.json --stage stage2
python -m ctrlvq run --config configs/local/mkg-w.json --stage evaluate

# Select physical GPUs. Multiple GPUs enable native DDP.
python -m ctrlvq run --config configs/local/mkg-w.json --stage stage1 --gpus 0,1

# Inspect run status.
python -m ctrlvq status --config configs/local/mkg-w.json

# Resume Stage 1 or Stage 2 from a complete training_state.pt checkpoint.
python -m ctrlvq run --config configs/local/mkg-w.json --stage stage1 --resume
```

To install the command-line entry point in editable mode without changing your installed dependencies:

```bash
pip install -e . --no-deps
ctrlvq plan --config configs/local/mkg-w.json
```

For a new environment, see `requirements.txt` for minimum dependency versions.
