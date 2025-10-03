from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# Causal PNS (CAT-style)
# --------------------------
@dataclass
class PNSCfg:
    max_dims_per_step: int | None = 16   # cuántas dims j intervenir por paso (None = todas)
    different_class_only: bool = True    # elegir k con clase distinta a y_i
    eps: float = 1e-8
    reduce: str = "mean"                 # "mean" o "sum"

class CausalPNSLoss(nn.Module):
    """
    lc = - sum_j sum_i log PNS_hat(i,j)
    con PNS_hat(i,j) ≈ clamp( p(y_i|z_i) - p(y_i|z_i^{j<-k}), 0, 1 )
    - z_cf: z con su dimensión j reemplazada por la de otro ejemplo k (intervención).
    - 'head' se usa para obtener logits (probabilidades) a partir de z y z_cf.
    """
    def __init__(self, cfg: PNSCfg):
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def _pick_cf_indices(self, y: torch.Tensor) -> torch.Tensor:
        B = y.size(0)
        k = torch.randperm(B, device=y.device)
        if not self.cfg.different_class_only:
            return k
        clash = (y == y[k])
        if clash.any():
            k2 = k.roll(1); k[clash] = k2[clash]
        return k

    def forward(self, z: torch.Tensor, y: torch.Tensor, head: nn.Module,
                logits: torch.Tensor | None = None) -> torch.Tensor:
        """
        z:      [B, Dz]  latente antes de la head
        y:      [B]     índices de clase (long)
        head:   z -> logits [B, C]
        logits: [B, C] opcional (para no recalcular con z original)
        """
        B, Dz = z.shape
        if logits is None:
            logits = head(z)  # [B, C]
        p = torch.softmax(logits, dim=-1).gather(1, y.view(-1,1)).squeeze(1)  # [B]

        if (self.cfg.max_dims_per_step is None) or (self.cfg.max_dims_per_step >= Dz):
            dims = torch.arange(Dz, device=z.device)
        else:
            dims = torch.randperm(Dz, device=z.device)[:self.cfg.max_dims_per_step]

        k = self._pick_cf_indices(y)

        per_dim = []
        for j in dims:
            z_cf = z.clone()
            z_cf[:, j] = z[k, j]                          # intervención do(Z_j <- Z_j^k)
            logits_cf = head(z_cf)                        # [B, C]
            p_cf = torch.softmax(logits_cf, dim=-1).gather(1, y.view(-1,1)).squeeze(1)
            pns_hat = (p - p_cf).clamp_min(0.0)           # ≈ PNS observable en [0,1]
            loss_ij = -torch.log(pns_hat + self.cfg.eps)  # -log PNŜ
            per_dim.append(loss_ij.mean() if self.cfg.reduce=="mean" else loss_ij.sum())

        if not per_dim:
            return torch.tensor(0.0, device=z.device)

        return torch.stack(per_dim).mean() if self.cfg.reduce=="mean" else torch.stack(per_dim).sum()


@dataclass
class LossCfg:
    w_cls: float = 1.0
    w_recon: float = 1.0
    w_causal: float = 1.0
    label_smoothing: float = 0.0
    # Hiper de PNS
    pns: PNSCfg = field(default_factory=PNSCfg)

class CATLosses(nn.Module):
    """
    L_total = w_cls * CE + w_recon * MSE + w_causal * PNS
    Partes (claras y separadas):
      - CE:         CrossEntropy (logits vs targets)
      - Recon:      MSE (Φ(Z) vs X que decidas reconstruir)
      - CausalPNS:  intervención sobre z con evaluación vía 'head'
    """
    def __init__(self, cfg: LossCfg, head: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        self.recon_loss = nn.MSELoss()
        self.pns_loss = CausalPNSLoss(cfg.pns)
        self.head = head  # se usa SOLO para PNS (z -> logits)

    # ----- Partes separadas -----
    def CELoss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.cls_loss(logits, targets.long())

    def ReconLoss(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.recon_loss(x_hat, x)

    def CausalLoss(self, z: torch.Tensor, targets: torch.Tensor,
                   logits: torch.Tensor | None = None) -> torch.Tensor:
        return self.pns_loss(z, targets, head=self.head, logits=logits)

    # ----- Forward combinado (argmin l1+l2+l3) -----
    def forward(self,
                logits: torch.Tensor,   # [B, C] (salida de la head)
                targets: torch.Tensor,  # [B]
                x_hat: torch.Tensor,    # Φ(Z)
                x: torch.Tensor,        # X (lo que reconstruyes)
                z: torch.Tensor         # [B, Dz] (pooling/flatten de CAT)
                ):
        l_cls   = self.CELoss(logits, targets)
        l_recon = self.ReconLoss(x_hat, x)
        l_caus  = self.CausalLoss(z, targets, logits=logits)

        total = (self.cfg.w_cls   * l_cls +
                 self.cfg.w_recon * l_recon +
                 self.cfg.w_causal* l_caus)

        return {"loss_total": total, "loss_cls": l_cls, "loss_recon": l_recon, "loss_causal": l_caus}
