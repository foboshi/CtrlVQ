




import os
import sys
import argparse
import json
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from glob import glob

logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_logging(log_file=None, rank=0):
    log_format = '%(asctime)s %(levelname)-8s %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO, format=log_format, datefmt=date_format,
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        if log_file:
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setFormatter(logging.Formatter(log_format, date_format))
            logging.getLogger().addHandler(file_handler)
    else:
        logging.basicConfig(level=logging.CRITICAL, format='%(message)s')

def setup_ddp():
    if not dist.is_available():
        raise RuntimeError("PyTorch 分布式训练不可用")
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    logger.info(f"DDP 初始化成功: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    return rank, world_size, local_rank

def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
        logger.info("DDP 清理完成")

def is_main_process(rank):
    return rank == 0

class TokenDataset(Dataset):
    def __init__(self, data_path, max_length=512, tokenizer=None):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        if isinstance(item, str):
            input_ids = self.tokenizer(item, truncation=True, max_length=self.max_length)['input_ids']
        elif isinstance(item, dict):
            input_ids = item['input_ids']
        else:
            input_ids = item
        input_ids = input_ids[:self.max_length]
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(input_ids, dtype=torch.long)
        }

@torch.no_grad()
def evaluate(model, dataloader, device, use_amp=True):
    model.eval()
    total_loss = 0
    count = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        with autocast(enabled=use_amp):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
        total_loss += loss.item() * input_ids.size(0)
        count += input_ids.size(0)
    model.train()
    return total_loss / count if count > 0 else 0

def main():
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B", help="")
    parser.add_argument("--model_source", type=str, default="modelscope", choices=["modelscope", "huggingface"], help="")
    parser.add_argument("--data_path", type=str, required=True, help="")
    parser.add_argument("--valid_data_path", type=str, default=None, help="")
    parser.add_argument("--tokens_path", type=str, required=True, help="")
    parser.add_argument("--output_dir", type=str, required=True, help="")
    
    parser.add_argument("--batch_size", type=int, default=4, help="")
    parser.add_argument("--grad_accum", type=int, default=4, help="")
    parser.add_argument("--lr", type=float, default=1e-4, help="")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="")
    parser.add_argument("--warmup_steps", type=int, default=500, help="")
    parser.add_argument("--max_steps", type=int, default=100000, help="")
    parser.add_argument("--save_steps", type=int, default=5000, help="")
    parser.add_argument("--eval_steps", type=int, default=1000, help="")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="")
    parser.add_argument("--max_length", type=int, default=256, help="")
    
    parser.add_argument("--local-rank", type=int, dest="local_rank", default=0, help="")
    parser.add_argument("--no_amp", action="store_true", default=False, help="")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=True, help="")
    
    parser.add_argument("--auto_resume", action="store_true", help="")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="")
    parser.add_argument("--log_file", type=str, default="training_stage1.log", help="")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    log_path = os.path.join(args.output_dir, args.log_file)
    setup_logging(log_path, rank)

    if is_main_process(rank):
        logger.info("="*30 + "\nStage 1: 扩展词表微调\n" + "="*30)
        logger.info(f"训练参数: {vars(args)}")

    try:
        resume_path = args.resume_from_checkpoint
        if not resume_path and args.auto_resume and os.path.isdir(args.output_dir):
            checkpoints = sorted(glob(os.path.join(args.output_dir, "checkpoint-*")))
            if checkpoints: resume_path = checkpoints[-1]

        load_path = resume_path if resume_path else args.model_name
        source_type = "huggingface" if resume_path else args.model_source
        if is_main_process(rank): logger.info(f"从 '{load_path}' 加载模型和 tokenizer...")

        if source_type == "modelscope":
            from modelscope import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(load_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(load_path)
            tokenizer = AutoTokenizer.from_pretrained(load_path)

        if not resume_path:
            with open(args.tokens_path, 'r') as f: new_tokens = json.load(f)
            if isinstance(new_tokens, dict): new_tokens = new_tokens.get('tokens', [])
            if is_main_process(rank): logger.info(f"添加 {len(new_tokens)} 个新 token 到词表")
            tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))

        for param in model.parameters(): param.requires_grad = False
        model.get_input_embeddings().weight.requires_grad = True
        if hasattr(model, 'lm_head'): model.lm_head.weight.requires_grad = True
        if args.use_gradient_checkpointing: model.gradient_checkpointing_enable()

        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

        dataset = TokenDataset(args.data_path, args.max_length, tokenizer)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)

        valid_loader = None
        if args.valid_data_path and is_main_process(rank):
            valid_dataset = TokenDataset(args.valid_data_path, args.max_length, tokenizer)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=4)

        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        scaler = GradScaler() if not args.no_amp else None

        start_step, start_epoch, best_valid_loss, patience = 0, 0, float('inf'), 0
        if resume_path:
            state_path = os.path.join(resume_path, 'training_state.pt')
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location=device)
                start_step, start_epoch = state['step'], state['epoch']
                best_valid_loss, patience = state.get('best_valid_loss', float('inf')), state.get('patience', 0)
                optimizer.load_state_dict(state['optimizer'])
                if scaler and 'scaler' in state: scaler.load_state_dict(state['scaler'])
                if is_main_process(rank): logger.info(f"恢复成功: 从 step {start_step}, epoch {start_epoch} 开始")

        step = start_step
        for epoch in range(start_epoch, 1000):
            sampler.set_epoch(epoch)
            for batch in train_loader:
                if step >= args.max_steps: break
                input_ids, labels = batch['input_ids'].to(device), batch['labels'].to(device)
                with autocast(enabled=not args.no_amp):
                    loss = model(input_ids=input_ids, labels=labels).loss
                optimizer.zero_grad()
                if scaler: scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else: loss.backward(); optimizer.step()
                step += 1

                if is_main_process(rank) and step % 100 == 0:
                    grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
                    logger.info(f"Epoch: {epoch+1}, Step: {step}/{args.max_steps}, Loss: {loss.item():.4f}, Grad Norm: {grad_norm:.4f}")

                if is_main_process(rank) and step % args.eval_steps == 0 and valid_loader:
                    valid_loss = evaluate(model, valid_loader, device, not args.no_amp)
                    logger.info(f"Validation Loss at step {step}: {valid_loss:.4f}")
                    if valid_loss < best_valid_loss:
                        best_valid_loss, patience = valid_loss, 0
                        best_model_path = os.path.join(args.output_dir, "best_model")
                        os.makedirs(best_model_path, exist_ok=True)
                        model.module.save_pretrained(best_model_path)
                        tokenizer.save_pretrained(best_model_path)
                        logger.info(f"新最佳模型已保存到: {best_model_path}")
                    else:
                        patience += 1
                        if patience >= args.early_stopping_patience:
                            logger.info(f"早停触发! 验证损失在 {patience} 次评估中未改善。")
                            step = args.max_steps 
                            break

                if is_main_process(rank) and step % args.save_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.module.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    torch.save({
                        'step': step, 'epoch': epoch, 'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict() if scaler else None,
                        'best_valid_loss': best_valid_loss, 'patience': patience
                    }, os.path.join(checkpoint_dir, 'training_state.pt'))
                    logger.info(f"Checkpoint 已保存到: {checkpoint_dir}")
            if step >= args.max_steps: break

        if is_main_process(rank):
            final_path = os.path.join(args.output_dir, "final")
            os.makedirs(final_path, exist_ok=True)
            model.module.save_pretrained(final_path)
            tokenizer.save_pretrained(final_path)
            logger.info(f"Stage 1 训练完成! 最终模型已保存到: {final_path}")

    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}", exc_info=True)
    finally:
        cleanup_ddp()

if __name__ == "__main__":
    main()

































