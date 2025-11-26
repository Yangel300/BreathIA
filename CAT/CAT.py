import torch
import torch.nn as nn
from dataclasses import dataclass, field
from MLP import NeuralNetwork, MLPConfig


@dataclass
class AcousticAttentionBlockConfig:
    d_model: int = 256
    heads_mel: int = 8
    heads_raw: int = 8
    dropout: float = 0.1

class AcousticAttentionBlock(nn.Module):
    """
    Entradas:
      - x_mel: Tensor [B, T, P, D]  (features para filtro mel)
      - x_raw: Tensor [B, T, P, D]  (features para filtro crudo)
      - pe3d : Tensor [B, T, P, D]  (3D attributes embedding ya calculado: tiempo+frecuencia+resolución)

        * B = Batch size
        * T = frames en tiempo
        * P = patches por frame 
        * D = d_model

    Salida:
      - y: Tensor [B, T, P, D]

    Detalles:
      - Suma PE3D a cada rama.
      - Aplica MHA separada por filtro
      - Fusiona salidas (concat -> proyección)
    """
    def __init__(self, cfg: AcousticAttentionBlockConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.heads_mel = cfg.heads_mel
        self.heads_raw = cfg.heads_raw

        # MHA separadas (batch_first=True => [B, L, D])
        self.mha_mel = nn.MultiheadAttention(cfg.d_model, self.heads_mel, dropout=cfg.dropout, batch_first=True)
        self.mha_raw = nn.MultiheadAttention(cfg.d_model, self.heads_raw, dropout=cfg.dropout, batch_first=True)

        # Fusión + salida
        self.out_proj = nn.Linear(2 * cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self,
                x_mel: torch.Tensor,      # [B, T, P, D]
                x_raw: torch.Tensor,      # [B, T, P, D]
                pe3d: torch.Tensor,       # [B, T, P, D]
                attn_mask=None,
                key_padding_mask=None):
        B, T, P, D = x_mel.shape
        assert x_raw.shape == (B, T, P, D)
        assert pe3d.shape == (B, T, P, D)

        # 1) Suma de 3D attributes embedding (tiempo + freq + resolución) en cada rama
        h_mel = x_mel + pe3d
        h_raw = x_raw + pe3d

        # 2) Reacomodar a secuencia para MHA: [B, L, D] con L = T*P
        L = T * P
        h_mel_seq = h_mel.reshape(B, L, D)
        h_raw_seq = h_raw.reshape(B, L, D)

        # 3) Atención separada por filtro
        #    Usamos self-attention por rama (Q=K=V = rama), pero puedes cruzarlas si lo necesitas.
        out_mel, _ = self.mha_mel(h_mel_seq, h_mel_seq, h_mel_seq,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=False)
        out_raw, _ = self.mha_raw(h_raw_seq, h_raw_seq, h_raw_seq,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=False)

        fused = torch.cat([out_mel, out_raw], dim=-1)  # [B, L, 2D]
        y_seq = self.out_proj(self.dropout(fused))                     # [B, L, D]

        # 6) Volver a [B, T, P, D]
        y = y_seq.reshape(B, T, P, D)
        return y
    


@dataclass
class CATConfig:
    d_model: int = 256
    dropout: float = 0.1
    pooling: str = "flatten"
    acoustic_cfg: AcousticAttentionBlockConfig = field(default_factory=AcousticAttentionBlockConfig)
    mlp_config: MLPConfig = field(default_factory=MLPConfig)



class CAT(nn.Module):
    """
    Bloque 'Acoustic attention block':
        LN -> AcousticAttention -> (+) -> LN -> MLP -> (+)
    Entradas:
      x_mel, x_raw, pe3d : [B, T, P, D]
    Salida:
      y : [B, T, P, D]
    """
    def __init__(self, cfg: CATConfig):
        super().__init__()

        self.cfg = cfg
        self.acoustic_attn = AcousticAttentionBlock(cfg.acoustic_cfg)
        # Pre-norm antes de la atención
        self.ln_attn = nn.LayerNorm(cfg.d_model)
        # Proyección para formar el "skip" de entrada (concat -> D) que se usa en el residual tras la atención
        self.res_in_proj = nn.Linear(2 * cfg.d_model, cfg.d_model)
        # Post-attention: LN y MLP
        self.ln_mlp = nn.LayerNorm(cfg.d_model)
        # MLP
        self.mlp = NeuralNetwork(cfg.mlp_config)

    def forward(self, x_mel: torch.Tensor, x_raw: torch.Tensor, pe3d: torch.Tensor):
        B, T, P, D = x_mel.shape
        # LN previo (pre-norm) sobre cada rama
        xm = self.ln_attn(x_mel)
        xr = self.ln_attn(x_raw)

        # Skip de entrada (concat -> proyección a D) para el residual tras la atención
        skip0 = self.res_in_proj(torch.cat([xm, xr], dim=-1))   # [B, T, P, D]

        # Acoustic Attention
        y_attention = self.acoustic_attn(xm, xr, pe3d) # [B, T, P, D]

        # Residual después de la atención 
        y_attention = y_attention + skip0

        # LN + MLP + Residual 
        ln_skip = self.ln_mlp(y_attention)

        mlp_out = self.mlp(ln_skip)

        y_out = y_attention + mlp_out 

        return y_out
    
    def pool_features(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: [B,T,P,D]
        Devuelve: [B,F]
        """
        B, T, P, D = feats.shape
        if self.cfg.pooling == "mean":
            return feats.mean(dim=(1,2))   # [B,D]
        elif self.cfg.pooling == "max":
            return feats.amax(dim=(1,2))   # [B,D]
        elif self.cfg.pooling == "flatten":
            return feats.reshape(B, T*P*D) # [B,T*P*D]
        else:
            raise ValueError