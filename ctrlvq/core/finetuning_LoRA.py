




import os
import sys
import argparse
import json
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler, ConcatDataset
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel

import torch.amp
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

class KGCPromptDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, last_k=32, only_answer_loss=False):
        with open(data_path, 'r') as f:
            raw_data = json.load(f)

        
        
        
        if isinstance(raw_data, dict) and 'prompts' in raw_data:
            self.data = raw_data['prompts']
        else:
            self.data = raw_data

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.last_k = last_k
        self.only_answer_loss = only_answer_loss
        
        self.answer_prefix = "Please generate the quantized representation of the answer entity:"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(
            text, return_tensors='pt', max_length=self.max_length,
            padding='max_length', truncation=True
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels = input_ids.clone()

        if self.only_answer_loss:
            
            answer_prefix_ids = self.tokenizer(self.answer_prefix, return_tensors='pt')['input_ids'].squeeze(0)
            
            input_ids_list = input_ids.tolist()
            answer_prefix_list = answer_prefix_ids.tolist()

            
            answer_start_pos = -1
            prefix_len = len(answer_prefix_list)
            for i in range(len(input_ids_list) - prefix_len + 1):
                if input_ids_list[i:i+prefix_len] == answer_prefix_list:
                    answer_start_pos = i + prefix_len
                    break

            if answer_start_pos > 0:
                
                labels[:answer_start_pos] = -100
            else:
                
                seq_len = attention_mask.sum().item()
                if seq_len > self.last_k:
                    labels[:seq_len - self.last_k] = -100
        else:
            
            seq_len = attention_mask.sum().item()
            if seq_len > self.last_k:
                labels[:seq_len - self.last_k] = -100

        labels[attention_mask == 0] = -100
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

@torch.no_grad()
def evaluate(model, dataloader, device, use_amp=True):
    model.eval()
    total_loss = 0
    count = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16 if use_amp else torch.float32):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        total_loss += loss.item() * input_ids.size(0)
        count += input_ids.size(0)
    model.train()
    return total_loss / count if count > 0 else 0


def evaluate_metrics(model, tokenizer, test_data, entity_codes_path, device, rank, logger, batch_size=16):
    
    import numpy as np
    from tqdm import tqdm

    logger.info("开始评估模型指标...")

    
    with open(entity_codes_path) as f:
        entity_codes = json.load(f)

    code_to_entity, entity_to_code = {}, {}
    for eid_str, info in entity_codes.items():
        eid = int(eid_str)
        token_seq = info['token_sequence']
        code_to_entity[token_seq] = eid
        entity_to_code[eid] = token_seq

    
    prompts = test_data['prompts']
    meta = test_data['meta']

    
    entity_ids = list(entity_to_code.keys())
    entity_id_to_idx = {eid: idx for idx, eid in enumerate(entity_ids)}

    
    code_token_ids = []
    code_lengths = []
    for eid in entity_ids:
        token_seq = entity_to_code[eid]
        tokens = tokenizer.encode(token_seq, add_special_tokens=False)
        code_token_ids.append(tokens)
        code_lengths.append(len(tokens))

    max_len = max(code_lengths)

    
    padded_codes = []
    for tokens in code_token_ids:
        padded = tokens + [tokenizer.pad_token_id] * (max_len - len(tokens))
        padded_codes.append(padded)

    code_tensor = torch.tensor(padded_codes, dtype=torch.long, device=device)
    lengths_tensor = torch.tensor(code_lengths, dtype=torch.long, device=device)

    
    length_to_indices = {}
    unique_lengths = torch.unique(lengths_tensor).cpu().numpy()
    for length in unique_lengths:
        mask = lengths_tensor == length
        indices = torch.where(mask)[0]
        length_to_indices[int(length)] = indices

    
    model_dtype = next(model.parameters()).dtype
    use_amp = (device.type == 'cuda') and (model_dtype in (torch.float16, torch.bfloat16))
    amp_dtype = model_dtype if use_amp else torch.float32

    logs = []
    num_entities = len(entity_ids)

    
    def compute_batch_scores_vectorized(batch_logits, seq_lengths, code_tensor, lengths_tensor, length_to_indices, device):
        
        batch_size = batch_logits.shape[0]
        num_entities = code_tensor.shape[0]
        dtype = batch_logits.dtype
        batch_scores = torch.full((batch_size, num_entities), -float('inf'), device=device, dtype=dtype)

        lengths = list(length_to_indices.keys())

        for sample_idx in range(batch_size):
            seq_len = seq_lengths[sample_idx].item()
            for L in lengths:
                if L <= seq_len:
                    indices = length_to_indices[L]
                    codes = code_tensor[indices, :L]
                    start_pos = seq_len - L
                    logits_block = batch_logits[sample_idx, start_pos:seq_len, :]
                    log_probs = F.log_softmax(logits_block, dim=-1)

                    idx_range = torch.arange(L, device=device).view(-1, 1)
                    token_log_probs = log_probs[idx_range, codes.T]
                    scores = token_log_probs.sum(dim=0)

                    batch_scores[sample_idx, indices] = scores

        return batch_scores

    for i in tqdm(range(0, len(prompts), batch_size), desc="评估中"):
        batch_prompts = prompts[i:i + batch_size]
        batch_meta = meta[i:i + batch_size]

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            prompt_encodings = tokenizer(
                batch_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=2048
            ).to(device)
            prompt_logits = model(**prompt_encodings).logits

        seq_lengths = prompt_encodings.attention_mask.sum(dim=1)

        
        batch_scores = compute_batch_scores_vectorized(
            prompt_logits, seq_lengths, code_tensor, lengths_tensor, length_to_indices, device
        )

        batch_size_actual = len(batch_prompts)
        for j in range(batch_size_actual):
            target_eid = batch_meta[j]['answer_entity_id']
            if target_eid not in entity_id_to_idx:
                continue
            target_idx = entity_id_to_idx[target_eid]
            target_score = batch_scores[j, target_idx]

            
            rank = (batch_scores[j] > target_score).sum().item() + 1

            logs.append({
                'MRR': 1.0 / rank,
                'MR': float(rank),
                'Hits@1': 1.0 if rank <= 1 else 0.0,
                'Hits@3': 1.0 if rank <= 3 else 0.0,
                'Hits@10': 1.0 if rank <= 10 else 0.0,
            })

    
    if logs:
        metrics = {
            'MRR': np.mean([l['MRR'] for l in logs]),
            'MR': np.mean([l['MR'] for l in logs]),
            'Hits@1': np.mean([l['Hits@1'] for l in logs]),
            'Hits@3': np.mean([l['Hits@3'] for l in logs]),
            'Hits@10': np.mean([l['Hits@10'] for l in logs]),
        }
    else:
        metrics = {'MRR': 0, 'MR': 0, 'Hits@1': 0, 'Hits@3': 0, 'Hits@10': 0}

    logger.info(f"评估指标: MRR={metrics['MRR']:.4f}, MR={metrics['MR']:.2f}, Hits@1={metrics['Hits@1']:.4f}, Hits@3={metrics['Hits@3']:.4f}, Hits@10={metrics['Hits@10']:.4f}")
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Stage 2: LLM LoRA 推理微调 (DDP 版本)")
    
    parser.add_argument("--stage1_model_path", type=str, required=True, help="Stage 1 微调后的模型路径")
    parser.add_argument("--train_data_path", type=str, required=True, help="Stage 2 训练数据路径")
    parser.add_argument("--valid_data_path", type=str, required=True, help="Stage 2 验证数据路径")
    parser.add_argument("--test_data_path", type=str, default=None, help="测试数据路径，用于作为验证集")
    parser.add_argument("--entity_codes_path", type=str, default=None, help="实体码字路径，用于评估")
    parser.add_argument("--merge_train_valid", action="store_true", default=False, help="将训练集和验证集合并")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    
    parser.add_argument("--batch_size", type=int, default=4, help="每个 GPU 的批处理大小")
    parser.add_argument("--grad_accum", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup_steps", type=int, default=200, help="预热步数")
    parser.add_argument("--use_cosine_annealing", action="store_true", default=False, help="使用余弦退火学习率调度")
    parser.add_argument("--max_steps", type=int, default=30000, help="最大训练步数")
    parser.add_argument("--save_steps", type=int, default=1000, help="保存 checkpoint 的步数间隔")
    parser.add_argument("--eval_steps", type=int, default=1000, help="评估模型的步数间隔")
    parser.add_argument("--early_stopping_patience", type=int, default=20, help="早停耐心值")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--last_k", type=int, default=32, help="只计算最后 k 个 token 的 loss")
    parser.add_argument("--only_answer_loss", action="store_true", default=False, help="只对 answer 部分计算 loss，忽略 prompt")
    
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    parser.add_argument("--local-rank", type=int, dest="local_rank", default=0, help="本地排名")
    parser.add_argument("--use_amp", action="store_true", default=True, help="使用混合精度训练")
    parser.add_argument("--no_amp", action="store_true", default=False, help="禁用混合精度训练")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=False, help="使用梯度检查点")  
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help="梯度裁剪的最大范数")  
    
    parser.add_argument("--auto_resume", action="store_true", help="自动从最新的 checkpoint 恢复")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="指定 checkpoint 目录进行恢复")
    parser.add_argument("--log_file", type=str, default="training_stage2.log", help="训练日志文件路径")
    args = parser.parse_args()

    
    if args.no_amp:
        args.use_amp = False

    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    log_path = os.path.join(args.output_dir, args.log_file)
    setup_logging(log_path, rank)

    if is_main_process(rank):
        logger.info("="*30 + "\nStage 2: LoRA 推理微调\n" + "="*30)
        logger.info(f"训练参数: {vars(args)}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if is_main_process(rank): logger.info(f"从 '{args.stage1_model_path}' 加载 Stage 1 模型...")
        model = AutoModelForCausalLM.from_pretrained(args.stage1_model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.stage1_model_path, trust_remote_code=True)

        
        if is_main_process(rank): logger.info("冻结所有基础参数...")
        for param in model.parameters():
            param.requires_grad = False

        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], bias="none"
        )
        model = get_peft_model(model, lora_config)

        
        resume_path = args.resume_from_checkpoint
        if not resume_path and args.auto_resume and os.path.isdir(args.output_dir):
            checkpoints = sorted(glob(os.path.join(args.output_dir, "checkpoint-*")), key=lambda x: int(x.split('-')[-1]))
            if checkpoints: resume_path = checkpoints[-1]

        if resume_path and os.path.isdir(resume_path):
            adapter_path = os.path.join(resume_path, 'adapter_model.safetensors')
            if os.path.exists(adapter_path):
                from safetensors.torch import load_file
                if is_main_process(rank): logger.info(f"从 '{adapter_path}' 加载 LoRA 权重...")
                adapter_state = load_file(adapter_path, device=str(device))
                
                model.load_state_dict(adapter_state, strict=False)
                if is_main_process(rank): logger.info(f"✓ LoRA 权重已恢复")

        
        if is_main_process(rank):
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            
            if trainable_params == 0:
                raise RuntimeError(
                    f"❌ 错误: 没有可训练的参数!\n"
                    f"   可训练参数: {trainable_params}\n"
                    f"   总参数数: {total_params}\n"
                    f"   请检查 LoRA 配置是否正确"
                )
            
            logger.info(f"✓ LoRA 参数配置成功")
            logger.info(f"  可训练参数: {trainable_params:,}")
            logger.info(f"  总参数数: {total_params:,}")
            logger.info(f"  可训练比例: {100 * trainable_params / total_params:.4f}%")
            model.print_trainable_parameters()
        
        
        if args.use_gradient_checkpointing:
            if is_main_process(rank): logger.info("启用梯度检查点...")
            model.gradient_checkpointing_enable()

        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        
        train_dataset = KGCPromptDataset(args.train_data_path, tokenizer, args.max_length, args.last_k, args.only_answer_loss)

        
        if args.merge_train_valid:
            if is_main_process(rank):
                logger.info(f"合并训练集和验证集...")
            valid_dataset = KGCPromptDataset(args.valid_data_path, tokenizer, args.max_length, args.last_k, args.only_answer_loss)
            
            train_dataset = ConcatDataset([train_dataset, valid_dataset])
            if is_main_process(rank):
                logger.info(f"合并后训练集大小: {len(train_dataset)}")

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=0)

        
        test_data_for_eval = None
        if args.test_data_path:
            if is_main_process(rank):
                logger.info(f"使用测试集作为验证集: {args.test_data_path}")
            
            valid_loader = DataLoader(KGCPromptDataset(args.test_data_path, tokenizer, args.max_length, args.last_k, only_answer_loss=False), batch_size=args.batch_size, num_workers=0)
            
            if args.entity_codes_path and is_main_process(rank):
                with open(args.test_data_path, 'r') as f:
                    test_data_for_eval = json.load(f)
        else:
            valid_loader = DataLoader(KGCPromptDataset(args.valid_data_path, tokenizer, args.max_length, args.last_k, only_answer_loss=False), batch_size=args.batch_size, num_workers=0)

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

        
        if args.use_cosine_annealing:
            def lr_lambda(step):
                if step < args.warmup_steps:
                    
                    return 0.1 + 0.9 * (step / args.warmup_steps)
                else:
                    
                    progress = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
                    return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            if is_main_process(rank):
                logger.info(f"使用自定义学习率调度: Warmup({args.warmup_steps}步) + CosineAnnealing({args.max_steps - args.warmup_steps}步)")
        else:
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_steps)
        
        scaler = torch.amp.GradScaler(device='cuda') if args.use_amp else None

        start_step, start_epoch, best_valid_loss, best_mrr, patience = 0, 0, float('inf'), 0.0, 0
        resume_path = args.resume_from_checkpoint
        if not resume_path and args.auto_resume and os.path.isdir(args.output_dir):
            checkpoints = sorted(glob(os.path.join(args.output_dir, "checkpoint-*")), key=lambda x: int(x.split('-')[-1]))
            if checkpoints: resume_path = checkpoints[-1]

        if resume_path:
            state_path = os.path.join(resume_path, 'training_state.pt')
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location=device)
                start_step, start_epoch = state['step'], state['epoch']
                best_valid_loss, best_mrr, patience = state.get('best_valid_loss', float('inf')), state.get('best_mrr', 0.0), state.get('patience', 0)
                optimizer.load_state_dict(state['optimizer'])
                scheduler.load_state_dict(state['scheduler'])
                if scaler and 'scaler' in state: scaler.load_state_dict(state['scaler'])
                if is_main_process(rank): logger.info(f"恢复成功: 从 step {start_step}, epoch {start_epoch} 开始")

        step = start_step
        for epoch in range(start_epoch, 1000):
            train_sampler.set_epoch(epoch)
            model.train()  
            for batch in train_loader:
                if step >= args.max_steps: break
                input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
                
                
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                if not trainable_params:
                    raise RuntimeError("❌ 没有可训练的参数!")
                
                
                if args.use_amp:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
                    
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    
                    
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
                    optimizer.zero_grad()
                    loss.backward()
                    
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    optimizer.step()
                
                scheduler.step()
                step += 1

                if is_main_process(rank) and step % 100 == 0:
                    grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(f"Epoch: {epoch+1}, Step: {step}/{args.max_steps}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}, Grad Norm: {grad_norm:.4f}")

                
                if step % args.save_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.module.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    torch.save({
                        'step': step, 'epoch': epoch, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                        'scaler': scaler.state_dict() if scaler else None, 'best_valid_loss': best_valid_loss, 'best_mrr': best_mrr, 'patience': patience
                    }, os.path.join(checkpoint_dir, 'training_state.pt'))
                    if is_main_process(rank):
                        logger.info(f"Checkpoint 已保存到: {checkpoint_dir}")

                    
                    if args.entity_codes_path and test_data_for_eval is not None and is_main_process(rank):
                        metrics = evaluate_metrics(model, tokenizer, test_data_for_eval, args.entity_codes_path, device, rank, logger, batch_size=16)
                        current_mrr = metrics.get('MRR', 0)
                        logger.info(f"Step {step} - MRR: {current_mrr:.4f}")

                        
                        if current_mrr > best_mrr:
                            best_mrr = current_mrr
                            best_model_dir = os.path.join(args.output_dir, "best_model")
                            os.makedirs(best_model_dir, exist_ok=True)
                            model.module.save_pretrained(best_model_dir)
                            tokenizer.save_pretrained(best_model_dir)
                            logger.info(f"新的最佳MRR: {best_mrr:.4f}, 最佳模型已保存到: {best_model_dir}")

                if args.eval_steps > 0 and step % args.eval_steps == 0:
                    avg_valid_loss = evaluate(model, valid_loader, device, args.use_amp)
                    if is_main_process(rank):
                        logger.info(f"Validation Loss at step {step}: {avg_valid_loss:.4f}")

                        is_best = avg_valid_loss < best_valid_loss
                        if is_best:
                            best_valid_loss, patience = avg_valid_loss, 0
                            logger.info(f"新的最佳验证损失: {best_valid_loss:.4f}")
                        else:
                            patience += 1
                            if patience >= args.early_stopping_patience:
                                logger.info(f"早停触发! 验证损失在 {patience} 次评估中未改善。")
                                step = args.max_steps 
                                break

            if step >= args.max_steps: break

        if is_main_process(rank):
            final_path = os.path.join(args.output_dir, "final")
            logger.info(f"训练完成! 最终模型已保存到: {final_path}")
            model.module.save_pretrained(final_path)
            tokenizer.save_pretrained(final_path)

    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}", exc_info=True)
    finally:
        cleanup_ddp()

if __name__ == "__main__":
    main()