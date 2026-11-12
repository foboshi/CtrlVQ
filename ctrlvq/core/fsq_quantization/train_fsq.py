




import os
import sys
import argparse
import logging
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default_config import DATASET_CONFIG, FUSION_CONFIG
from multimodal_embedding.load_embeddings import build_entity_relation_ids, load_multimodal_embeddings
from multimodal_embedding.fusion import MultiModalFusionWithReconstruction
from fsq_quantization.fsq import FSQAutoEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def setup_ddp():
    
    if os.environ.get('RANK') is None:
        logger.info("单卡模式运行，不使用 DDP")
        return 0, 1, 0
    if not dist.is_available():
        raise RuntimeError("Requires distributed package to be available")
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    logger.info(f"DDP 初始化完成: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    return rank, world_size, local_rank

def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank):
    return rank == 0

def parse_args():
    parser = argparse.ArgumentParser(description="训练 FSQ 量化模型 (DDP 版本)")
    parser.add_argument("--dataset", type=str, default="MKG-W", choices=["MKG-W", "MKG-Y", "MMKB-DB15K"], help="数据集名称")
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 8, 8, 8, 8, 8], help="FSQ 量化级别")
    parser.add_argument("--num_codebooks", type=int, default=1, help="FSQ 码本数量")
    parser.add_argument("--fused_emb_dim", type=int, default=512, help="融合嵌入维度")
    parser.add_argument("--latent_dim", type=int, default=256, help="FSQ 潜在空间维度")
    parser.add_argument("--fusion_method", type=str, default="concat_proj", choices=["concat_proj", "attention", "gate"], help="多模态融合方式")
    parser.add_argument("--batch_size", type=int, default=512, help="训练批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--recon_weight", type=float, default=0.5, help="多模态重建损失权重")
    parser.add_argument("--diversity_weight", type=float, default=1.0, help="码本多样性损失权重")
    parser.add_argument("--contrastive_weight", type=float, default=0.1, help="对比学习损失权重")
    parser.add_argument("--use_adapter", action="store_true", default=True, help="使用 Adapter 层")
    parser.add_argument("--use_gumbel", action="store_true", default=False, help="使用 Gumbel Softmax 量化")
    parser.add_argument("--temperature", type=float, default=1.0, help="Gumbel Softmax 温度")
    parser.add_argument("--save_dir", type=str, default="./outputs/fsq_model", help="模型保存目录")
    parser.add_argument("--local-rank", type=int, dest="local_rank", default=0, help="本地排名")
    return parser.parse_args()

def train_epoch(fusion_model, fsq_model, dataloader, optimizer, device, recon_weight=0.5, diversity_weight=1.0, contrastive_weight=0.1, use_adapter=True, use_gumbel=False, temperature=1.0, rank=0, has_struct_emb=False):
    
    fusion_model.train()
    fsq_model.train()

    total_loss, total_recon_loss, total_mm_recon_loss, total_diversity_loss, total_contrastive_loss = 0, 0, 0, 0, 0
    all_indices = []
    num_batches = 0

    for batch in dataloader:
        text_emb, img_emb = batch[0].to(device), batch[1].to(device)
        struct_emb = batch[2].to(device) if has_struct_emb else None
        batch_size = text_emb.shape[0]

        
        fused_emb, recon_text, recon_img = fusion_model(text_emb, img_emb, struct_emb)

        
        x_recon, indices, z, z_q = fsq_model(fused_emb, use_adapter=use_adapter, use_gumbel=use_gumbel, temperature=temperature)

        
        fsq_recon_loss = F.mse_loss(x_recon, fused_emb)

        
        fusion_model_actual = fusion_model.module if isinstance(fusion_model, nn.parallel.DistributedDataParallel) else fusion_model
        mm_recon_loss = fusion_model_actual.reconstruction_loss(text_emb, img_emb, recon_text, recon_img)

        
        fsq_model_actual = fsq_model.module if isinstance(fsq_model, nn.parallel.DistributedDataParallel) else fsq_model
        codebook_size = fsq_model_actual.fsq.codebook_size

        
        code_counts = torch.bincount(indices.flatten(), minlength=codebook_size).float()

        
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(code_counts, op=dist.ReduceOp.SUM)

        
        code_probs = code_counts / (code_counts.sum() + 1e-9)

        
        diversity_loss = (code_probs * torch.log(code_probs + 1e-9)).sum()

        
        
        if batch_size >= 4:  
            mid = batch_size // 2
            z1 = z[:mid]  
            z2 = z[mid:]  

            
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            sim_matrix = torch.mm(z1, z2.t()) / 0.07  

            
            labels = torch.arange(min(z1.shape[0], z2.shape[0]), device=z.device)
            contrastive_loss = F.cross_entropy(sim_matrix, labels)
        else:
            contrastive_loss = torch.tensor(0.0, device=z.device)
        

        
        loss = fsq_recon_loss + recon_weight * mm_recon_loss + diversity_weight * diversity_loss + contrastive_weight * contrastive_loss

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += fsq_recon_loss.item()
        total_mm_recon_loss += mm_recon_loss.item()
        total_diversity_loss += diversity_loss.item()
        total_contrastive_loss += contrastive_loss.item()
        all_indices.append(indices.detach().cpu())
        num_batches += 1

    
    all_indices = torch.cat(all_indices)
    codebook_usage = fsq_model_actual.fsq.get_codebook_usage(all_indices)

    return (
        total_loss / num_batches,
        total_recon_loss / num_batches,
        total_mm_recon_loss / num_batches,
        total_diversity_loss / num_batches,
        total_contrastive_loss / num_batches,
        codebook_usage
    )

def main():
    args = parse_args()

    
    rank, world_size, local_rank = setup_ddp()
    args.local_rank = local_rank
    torch.manual_seed(42 + rank)

    device = torch.device(f"cuda:{local_rank}")
    if is_main_process(rank):
        logger.info(f"使用设备: {device}")

    dataset_cfg = DATASET_CONFIG[args.dataset]
    if is_main_process(rank):
        logger.info(f"数据集: {args.dataset}")
        logger.info(f"多样性损失权重: {args.diversity_weight}")

    try:
        
        entity2id, relation2id, id2entity, id2relation = build_entity_relation_ids(dataset_cfg["data_path"])

        
        if is_main_process(rank):
            logger.info("加载多模态嵌入...")

        
        structural_emb_path = dataset_cfg.get("structural_emb_path")

        ent_text_emb, ent_img_emb, ent_struct_emb = load_multimodal_embeddings(
            args.dataset, dataset_cfg, entity2id, device='cpu',
            structural_emb_path=structural_emb_path
        )
        if is_main_process(rank):
            logger.info(f"文本嵌入形状: {ent_text_emb.shape}")
            logger.info(f"图像嵌入形状: {ent_img_emb.shape}")
            if ent_struct_emb is not None:
                logger.info(f"结构嵌入形状: {ent_struct_emb.shape}")

        
        if ent_struct_emb is not None:
            dataset = TensorDataset(ent_text_emb, ent_img_emb, ent_struct_emb)
        else:
            dataset = TensorDataset(ent_text_emb, ent_img_emb)

        
        use_sampler = False
        if world_size == 1:
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        else:
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
            )
            dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
            use_sampler = True

        
        fusion_cfg = FUSION_CONFIG.copy()
        fusion_cfg["fused_emb_dim"] = args.fused_emb_dim
        fusion_cfg["fusion_method"] = args.fusion_method

        fusion_model = MultiModalFusionWithReconstruction(fusion_cfg).to(device)
        fsq_model = FSQAutoEncoder(
            in_dim=args.fused_emb_dim,
            latent_dim=args.latent_dim,
            levels=args.levels,
            num_codebooks=args.num_codebooks,
        ).to(device)

        
        if world_size > 1:
            fusion_model = nn.parallel.DistributedDataParallel(
                fusion_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
            )
            fsq_model = nn.parallel.DistributedDataParallel(
                fsq_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
            )
            fsq_codebook_size = fsq_model.module.fsq.codebook_size
            
            fusion_model_actual = fusion_model.module
            fsq_model_actual = fsq_model.module
        else:
            fsq_codebook_size = fsq_model.fsq.codebook_size
            fusion_model_actual = fusion_model
            fsq_model_actual = fsq_model

        if is_main_process(rank):
            logger.info(f"融合模型参数量: {sum(p.numel() for p in fusion_model.parameters()):,}")
            logger.info(f"FSQ模型参数量: {sum(p.numel() for p in fsq_model.parameters()):,}")
            logger.info(f"FSQ 码本大小: {fsq_codebook_size}")

        
        optimizer = torch.optim.Adam(
            list(fusion_model.parameters()) + list(fsq_model.parameters()),
            lr=args.lr
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs
        )

        
        os.makedirs(args.save_dir, exist_ok=True)
        best_loss = float('inf')

        
        has_struct_emb = ent_struct_emb is not None and ent_struct_emb.shape[1] > 0

        for epoch in range(args.num_epochs):
            if use_sampler:
                sampler.set_epoch(epoch)
            
            avg_loss, avg_recon, avg_mm_recon, avg_diversity, avg_contrastive, usage = train_epoch(
                fusion_model, fsq_model, dataloader, optimizer, device,
                recon_weight=args.recon_weight, diversity_weight=args.diversity_weight,
                contrastive_weight=args.contrastive_weight,
                use_adapter=args.use_adapter, use_gumbel=args.use_gumbel,
                temperature=args.temperature, rank=rank,
                has_struct_emb=has_struct_emb
            )
            scheduler.step()

            if is_main_process(rank):
                logger.info(
                    f"Epoch {epoch+1}/{args.num_epochs}: "
                    f"loss={avg_loss:.4f}, fsq_recon={avg_recon:.4f}, "
                    f"mm_recon={avg_mm_recon:.4f}, diversity={avg_diversity:.4f}, "
                    f"contrastive={avg_contrastive:.4f}, codebook_usage={usage:.2%}"
                )

                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save({
                        'fusion_model': fusion_model_actual.state_dict(),
                        'fsq_model': fsq_model_actual.state_dict(),
                        'epoch': epoch,
                        'loss': avg_loss,
                    }, os.path.join(args.save_dir, 'best_model.pt'))
                    logger.info(f"  → 保存最佳模型 (loss={avg_loss:.4f})")

        
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        
        if is_main_process(rank):
            logger.info("生成所有实体的 FSQ 码字...")

        fusion_model.eval()
        fsq_model.eval()

        all_indices = []
        all_fused = []

        
        has_struct_emb = ent_struct_emb is not None and ent_struct_emb.shape[1] > 0

        with torch.no_grad():
            for i in range(0, len(ent_text_emb), args.batch_size):
                text_batch = ent_text_emb[i:i+args.batch_size].to(device)
                img_batch = ent_img_emb[i:i+args.batch_size].to(device)
                struct_batch = ent_struct_emb[i:i+args.batch_size].to(device) if has_struct_emb else None

                fused_emb, _, _ = fusion_model(text_batch, img_batch, struct_batch)
                indices = fsq_model_actual.get_codes(fused_emb)

                all_indices.append(indices.cpu())
                all_fused.append(fused_emb.cpu())

        all_indices = torch.cat(all_indices)
        all_fused = torch.cat(all_fused)

        
        if is_main_process(rank):
            
            np.save(os.path.join(args.save_dir, 'entity_fsq_indices.npy'),
                    all_indices.numpy())
            np.save(os.path.join(args.save_dir, 'entity_fused_embeddings.npy'),
                    all_fused.numpy())

            
            with open(os.path.join(args.save_dir, 'entity2id.json'), 'w') as f:
                json.dump(entity2id, f, ensure_ascii=False)
            with open(os.path.join(args.save_dir, 'relation2id.json'), 'w') as f:
                json.dump(relation2id, f, ensure_ascii=False)
            with open(os.path.join(args.save_dir, 'id2entity.json'), 'w') as f:
                json.dump(id2entity, f, ensure_ascii=False)
            with open(os.path.join(args.save_dir, 'id2relation.json'), 'w') as f:
                json.dump(id2relation, f, ensure_ascii=False)

            
            config = {
                'dataset': args.dataset,
                'levels': args.levels,
                'num_codebooks': args.num_codebooks,
                'codebook_size': fsq_model_actual.fsq.codebook_size,
                'fused_emb_dim': args.fused_emb_dim,
                'latent_dim': args.latent_dim,
                'fusion_method': args.fusion_method,
                'num_entities': len(entity2id),
                'num_relations': len(relation2id),
                'diversity_weight': args.diversity_weight,
            }
            with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=2)

            final_usage = fsq_model_actual.fsq.get_codebook_usage(all_indices)
            logger.info(f"训练完成! 码本利用率: {final_usage:.2%}")
            logger.info(f"码字索引已保存到: {args.save_dir}/entity_fsq_indices.npy")

    finally:
        cleanup_ddp()

if __name__ == "__main__":
    main()
