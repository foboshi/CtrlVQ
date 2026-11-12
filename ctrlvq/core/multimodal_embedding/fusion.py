

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class MultiModalFusion(nn.Module):
    

    def __init__(self, text_emb_dim=384, img_emb_dim=383,
                 text_max_sentences=4, img_max_patches=24,
                 text_proj_dim=256, img_proj_dim=256,
                 fused_emb_dim=512, fusion_method="concat_proj",
                 dropout=0.1):
        super().__init__()

        self.text_emb_dim = text_emb_dim
        self.img_emb_dim = img_emb_dim
        self.text_max_sentences = text_max_sentences
        self.img_max_patches = img_max_patches
        self.text_proj_dim = text_proj_dim
        self.img_proj_dim = img_proj_dim
        self.fused_emb_dim = fused_emb_dim
        self.fusion_method = fusion_method

        
        
        self.text_proj = nn.Linear(text_emb_dim, text_proj_dim)
        self.text_norm = nn.LayerNorm(text_proj_dim)

        
        
        self.img_proj = nn.Linear(img_emb_dim, img_proj_dim)
        self.img_norm = nn.LayerNorm(img_proj_dim)

        
        
        if fusion_method == "attention":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=text_proj_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.attn_norm = nn.LayerNorm(text_proj_dim)

        
        if fusion_method == "gate":
            self.gate_linear = nn.Linear(text_proj_dim + img_proj_dim,
                                          text_proj_dim + img_proj_dim)

        
        if fusion_method == "concat_proj":
            self.fusion_proj = nn.Linear(text_proj_dim + img_proj_dim,
                                          fused_emb_dim)
        elif fusion_method == "attention":
            
            self.fusion_proj = nn.Linear(text_proj_dim * 2, fused_emb_dim)
        elif fusion_method == "gate":
            self.fusion_proj = nn.Linear(text_proj_dim + img_proj_dim,
                                          fused_emb_dim)
        else:
            raise ValueError(f"不支持的融合方式: {fusion_method}")

        self.fusion_norm = nn.LayerNorm(fused_emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU(0.1)

        
        self._init_weights()

    def _init_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_text(self, text_emb):
        
        
        text_seq = self.relu(self.text_proj(text_emb))
        text_seq = self.text_norm(text_seq)

        
        text_feat = text_seq.mean(dim=1)

        return text_feat, text_seq

    def encode_image(self, img_emb):
        
        
        img_seq = self.relu(self.img_proj(img_emb))
        img_seq = self.img_norm(img_seq)

        
        img_feat = img_seq.mean(dim=1)

        return img_feat, img_seq

    def fuse_concat_proj(self, text_feat, img_feat):
        
        
        concat = torch.cat([text_feat, img_feat], dim=-1)
        
        fused = self.fusion_proj(concat)
        fused = self.fusion_norm(fused)
        return fused

    def fuse_attention(self, text_feat, text_seq, img_feat, img_seq):
        
        
        text_attended, _ = self.cross_attn(text_seq, img_seq, img_seq)
        text_attended = self.attn_norm(text_attended + text_seq)
        text_out = text_attended.mean(dim=1)  

        
        img_attended, _ = self.cross_attn(img_seq, text_seq, text_seq)
        img_attended = self.attn_norm(img_attended + img_seq)
        img_out = img_attended.mean(dim=1)  

        
        concat = torch.cat([text_out, img_out], dim=-1)
        fused = self.fusion_proj(concat)
        fused = self.fusion_norm(fused)
        return fused

    def fuse_gate(self, text_feat, img_feat):
        
        concat = torch.cat([text_feat, img_feat], dim=-1)
        gate = torch.sigmoid(self.gate_linear(concat))
        gated = concat * gate
        fused = self.fusion_proj(gated)
        fused = self.fusion_norm(fused)
        return fused

    def forward(self, text_emb, img_emb, struct_emb=None):
        
        
        text_feat, text_seq = self.encode_text(text_emb)
        img_feat, img_seq = self.encode_image(img_emb)

        
        if struct_emb is not None and struct_emb.shape[1] > 0:
            
            struct_dim = struct_emb.shape[1]
            if not hasattr(self, 'struct_proj') or self.struct_proj.in_features != struct_dim:
                
                self.struct_proj = nn.Linear(struct_dim, self.text_proj_dim)
                self.struct_norm = nn.LayerNorm(self.text_proj_dim)
                
                self.struct_proj = self.struct_proj.to(struct_emb.device)
                self.struct_norm = self.struct_norm.to(struct_emb.device)

                
                if self.fusion_method == "concat_proj":
                    self.fusion_proj = nn.Linear(
                        self.text_proj_dim + self.img_proj_dim + self.text_proj_dim,
                        self.fused_emb_dim
                    ).to(struct_emb.device)
                elif self.fusion_method == "gate":
                    self.gate_linear = nn.Linear(
                        self.text_proj_dim + self.img_proj_dim + self.text_proj_dim,
                        self.text_proj_dim + self.img_proj_dim + self.text_proj_dim
                    ).to(struct_emb.device)

            struct_feat = self.relu(self.struct_proj(struct_emb))
            struct_feat = self.struct_norm(struct_feat)

            
            if self.fusion_method == "concat_proj":
                concat = torch.cat([text_feat, img_feat, struct_feat], dim=-1)
                fused_emb = self.fusion_proj(concat)
            elif self.fusion_method == "attention":
                
                text_feat = text_feat + struct_feat
                img_feat = img_feat + struct_feat
                fused_emb = self.fuse_attention(text_feat, text_seq,
                                                 img_feat, img_seq)
            elif self.fusion_method == "gate":
                concat = torch.cat([text_feat, img_feat, struct_feat], dim=-1)
                gate = torch.sigmoid(self.gate_linear(concat))
                gated = concat * gate
                
                if not hasattr(self, 'gate_fusion_proj') or self.gate_fusion_proj.in_features != concat.shape[1]:
                    self.gate_fusion_proj = nn.Linear(
                        concat.shape[1], self.fused_emb_dim
                    ).to(concat.device)
                fused_emb = self.gate_fusion_proj(gated)
            else:
                raise ValueError(f"不支持的融合方式: {self.fusion_method}")
        else:
            
            if self.fusion_method == "concat_proj":
                fused_emb = self.fuse_concat_proj(text_feat, img_feat)
            elif self.fusion_method == "attention":
                fused_emb = self.fuse_attention(text_feat, text_seq,
                                                 img_feat, img_seq)
            elif self.fusion_method == "gate":
                fused_emb = self.fuse_gate(text_feat, img_feat)
            else:
                raise ValueError(f"不支持的融合方式: {self.fusion_method}")

        fused_emb = self.dropout(fused_emb)
        return fused_emb


class MultiModalFusionWithReconstruction(nn.Module):
    

    def __init__(self, fusion_config):
        super().__init__()

        cfg = fusion_config
        self.fusion = MultiModalFusion(
            text_emb_dim=cfg["text_emb_dim"],
            img_emb_dim=cfg["img_emb_dim"],
            text_max_sentences=cfg["text_max_sentences"],
            img_max_patches=cfg["img_max_patches"],
            text_proj_dim=cfg["text_proj_dim"],
            img_proj_dim=cfg["img_proj_dim"],
            fused_emb_dim=cfg["fused_emb_dim"],
            fusion_method=cfg["fusion_method"],
            dropout=cfg["dropout"],
        )

        fused_dim = cfg["fused_emb_dim"]
        text_flat_dim = cfg["text_max_sentences"] * cfg["text_emb_dim"]
        img_flat_dim = cfg["img_max_patches"] * cfg["img_emb_dim"]

        
        self.text_decoder = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(fused_dim, text_flat_dim),
        )

        self.img_decoder = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(fused_dim, img_flat_dim),
        )

        self.text_max_sentences = cfg["text_max_sentences"]
        self.text_emb_dim = cfg["text_emb_dim"]
        self.img_max_patches = cfg["img_max_patches"]
        self.img_emb_dim = cfg["img_emb_dim"]

    def forward(self, text_emb, img_emb, struct_emb=None):
        
        B = text_emb.shape[0]

        
        fused_emb = self.fusion(text_emb, img_emb, struct_emb)

        
        recon_text = self.text_decoder(fused_emb)
        recon_text = recon_text.view(B, self.text_max_sentences,
                                      self.text_emb_dim)

        
        recon_img = self.img_decoder(fused_emb)
        recon_img = recon_img.view(B, self.img_max_patches,
                                    self.img_emb_dim)

        return fused_emb, recon_text, recon_img

    def reconstruction_loss(self, text_emb, img_emb, recon_text, recon_img):
        
        text_loss = F.mse_loss(recon_text, text_emb)
        img_loss = F.mse_loss(recon_img, img_emb)
        return text_loss + img_loss
