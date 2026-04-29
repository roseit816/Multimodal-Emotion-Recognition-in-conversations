"""
SDT: Self-Distillation Transformer for Multimodal ERC
Paper: "A Transformer-Based Model With Self-Distillation for
        Multimodal Emotion Recognition in Conversations"
IEEE TMM, 2024

This module implements the COMPLETE architecture as described in the paper,
equation by equation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ═════════════════════════════════════════════════════════════════════════════
# 1. Temporal Convolutional Feature Projection  (Eq. 1)
# ─────────────────────────────────────────────────────────────────────────────
# Eq. 1:  H_m = ReLU( Conv1D(X_m) )   for each modality m ∈ {t, a, v}
# ═════════════════════════════════════════════════════════════════════════════
class TemporalConvProjector(nn.Module):
    """
    Projects raw modality features into the shared hidden dimension
    using a 1-D temporal convolution (Eq. 1).

    Input:  (B, T, D_in)
    Output: (B, T, D_h)
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2           # same-padding
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D_in)
        x = x.permute(0, 2, 1)                    # → (B, D_in, T)
        x = self.conv(x)                           # → (B, D_h, T)
        x = F.relu(x)
        x = x.permute(0, 2, 1)                    # → (B, T, D_h)
        x = self.norm(x)
        return self.dropout(x)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Positional Embedding  (Eq. 2)
# ─────────────────────────────────────────────────────────────────────────────
# Eq. 2:  Z_m = H_m + PE
#         PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
#         PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
# ═════════════════════════════════════════════════════════════════════════════
class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings (Eq. 2)."""

    def __init__(self, hidden_dim: int, max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, hidden_dim)     # (L, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (L,1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:hidden_dim // 2])
        pe = pe.unsqueeze(0)                       # (1, L, D)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Speaker Embedding  (Eq. 3–4)
# ─────────────────────────────────────────────────────────────────────────────
# Eq. 3:  SE_i = Embedding(s_i)          — lookup per utterance
# Eq. 4:  Z_m  = Z_m + SE                — added to positional-encoded repr
# ═════════════════════════════════════════════════════════════════════════════
class SpeakerEmbedding(nn.Module):
    """
    Learns a speaker-specific embedding and adds it to modality representations.
    (Eq. 3–4)
    """

    def __init__(self, max_speakers: int, hidden_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.embed   = nn.Embedding(max_speakers, hidden_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, speaker_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:           (B, T, D)
            speaker_ids: (B, T) — integer speaker IDs, padded with -1
        """
        # Map padding index -1 → 0 (padding_idx)
        ids = speaker_ids.clamp(min=0)
        se  = self.embed(ids)              # (B, T, D)
        return self.dropout(x + se)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Intra-modal Transformer  (Eq. 5)
# ─────────────────────────────────────────────────────────────────────────────
# Eq. 5:  Z_m = TransformerEncoder(Z_m)
#         — captures temporal context WITHIN a single modality
# ═════════════════════════════════════════════════════════════════════════════
class IntraModalTransformer(nn.Module):
    """Standard Transformer encoder applied to a single modality (Eq. 5)."""

    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int,
                 ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # Pre-LayerNorm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:                (B, T, D)
            key_padding_mask: (B, T) True where PADDING positions are
        Returns:
            (B, T, D)
        """
        return self.encoder(x, src_key_padding_mask=key_padding_mask)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Inter-modal Transformer  (Eq. 6)
# ─────────────────────────────────────────────────────────────────────────────
# Eq. 6:  R_m = CrossAttention(Q=Z_m, K=Z_m', V=Z_m')  for m' ≠ m
#         — captures cross-modal interactions
# ═════════════════════════════════════════════════════════════════════════════
class InterModalTransformerLayer(nn.Module):
    """
    One layer of cross-modal attention where modality m attends to
    the other two modalities.  (Eq. 6)
    """

    def __init__(self, hidden_dim: int, num_heads: int,
                 ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        # Cross-attention: Q from m, K/V from other modalities
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Self-attention on the query modality
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1  = nn.LayerNorm(hidden_dim)
        self.norm2  = nn.LayerNorm(hidden_dim)
        self.norm3  = nn.LayerNorm(hidden_dim)
        self.ffn    = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,              # (B, T, D) — current modality
        context: torch.Tensor,            # (B, T, D) — concatenation of other modalities
        query_mask:   Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm cross-attention
        residual = query
        query_ = self.norm1(query)
        ctx_   = self.norm1(context)
        attn_out, _ = self.cross_attn(
            query=query_,
            key=ctx_,
            value=ctx_,
            key_padding_mask=context_mask,
        )
        query = residual + self.dropout(attn_out)

        # Pre-norm self-attention
        residual = query
        query_ = self.norm2(query)
        self_out, _ = self.self_attn(
            query=query_,
            key=query_,
            value=query_,
            key_padding_mask=query_mask,
        )
        query = residual + self.dropout(self_out)

        # Feed-forward
        residual = query
        query = residual + self.dropout(self.ffn(self.norm3(query)))

        return query


class InterModalTransformer(nn.Module):
    """
    Stacks multiple inter-modal transformer layers.
    Each modality attends to the concatenation of the other two (Eq. 6).
    """

    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int,
                 ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            InterModalTransformerLayer(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        query:   torch.Tensor,
        context: torch.Tensor,
        query_mask:   Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            query = layer(query, context, query_mask, context_mask)
        return query


# ═════════════════════════════════════════════════════════════════════════════
# 6. Modality Encoder  (combines Eq. 1–6)
# ═════════════════════════════════════════════════════════════════════════════
class ModalityEncoder(nn.Module):
    """
    Full per-modality encoder:
      Conv1D projection → Positional Embedding → Speaker Embedding
      → Intra-modal Transformer → Inter-modal Transformer

    Processes text (t), audio (a), visual (v) jointly.
    """

    def __init__(
        self,
        text_dim:    int,
        audio_dim:   int,
        visual_dim:  int,
        hidden_dim:  int,
        num_heads:   int,
        num_intra:   int,
        num_inter:   int,
        ffn_dim:     int,
        max_speakers: int,
        conv_kernel:  int = 3,
        dropout:      float = 0.1,
        max_seq_len:  int = 512,
    ):
        super().__init__()

        # ── Temporal projection (Eq. 1) ─────────────────────────────────────
        self.proj_t = TemporalConvProjector(text_dim,   hidden_dim, conv_kernel, dropout)
        self.proj_a = TemporalConvProjector(audio_dim,  hidden_dim, conv_kernel, dropout)
        self.proj_v = TemporalConvProjector(visual_dim, hidden_dim, conv_kernel, dropout)

        # ── Positional embeddings (Eq. 2) ───────────────────────────────────
        self.pos_t = PositionalEmbedding(hidden_dim, max_seq_len, dropout)
        self.pos_a = PositionalEmbedding(hidden_dim, max_seq_len, dropout)
        self.pos_v = PositionalEmbedding(hidden_dim, max_seq_len, dropout)

        # ── Speaker embeddings (Eq. 3–4) ────────────────────────────────────
        self.spk_t = SpeakerEmbedding(max_speakers, hidden_dim, dropout)
        self.spk_a = SpeakerEmbedding(max_speakers, hidden_dim, dropout)
        self.spk_v = SpeakerEmbedding(max_speakers, hidden_dim, dropout)

        # ── Intra-modal transformers (Eq. 5) ────────────────────────────────
        self.intra_t = IntraModalTransformer(hidden_dim, num_heads, num_intra, ffn_dim, dropout)
        self.intra_a = IntraModalTransformer(hidden_dim, num_heads, num_intra, ffn_dim, dropout)
        self.intra_v = IntraModalTransformer(hidden_dim, num_heads, num_intra, ffn_dim, dropout)

        # ── Inter-modal transformers (Eq. 6) ────────────────────────────────
        # Each modality attends to the other two
        self.inter_t = InterModalTransformer(hidden_dim, num_heads, num_inter, ffn_dim, dropout)
        self.inter_a = InterModalTransformer(hidden_dim, num_heads, num_inter, ffn_dim, dropout)
        self.inter_v = InterModalTransformer(hidden_dim, num_heads, num_inter, ffn_dim, dropout)

    def forward(
        self,
        text:         torch.Tensor,    # (B, T, D_t)
        audio:        torch.Tensor,    # (B, T, D_a)
        visual:       torch.Tensor,    # (B, T, D_v)
        speaker_ids:  torch.Tensor,    # (B, T)
        pad_mask:     Optional[torch.Tensor] = None,  # (B, T) True at PAD pos
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            R_t, R_a, R_v — each (B, T, D_h)
        """
        # ── Eq. 1: Temporal Conv projection ─────────────────────────────────
        H_t = self.proj_t(text)
        H_a = self.proj_a(audio)
        H_v = self.proj_v(visual)

        # ── Eq. 2: Positional embeddings ─────────────────────────────────────
        Z_t = self.pos_t(H_t)
        Z_a = self.pos_a(H_a)
        Z_v = self.pos_v(H_v)

        # ── Eq. 3–4: Speaker embeddings ──────────────────────────────────────
        Z_t = self.spk_t(Z_t, speaker_ids)
        Z_a = self.spk_a(Z_a, speaker_ids)
        Z_v = self.spk_v(Z_v, speaker_ids)

        # ── Eq. 5: Intra-modal transformer ───────────────────────────────────
        Z_t = self.intra_t(Z_t, key_padding_mask=pad_mask)
        Z_a = self.intra_a(Z_a, key_padding_mask=pad_mask)
        Z_v = self.intra_v(Z_v, key_padding_mask=pad_mask)

        # ── Eq. 6: Inter-modal transformer ───────────────────────────────────
        # Text attends to (audio + visual)
        ctx_for_t = (Z_a + Z_v) / 2.0          # element-wise average as context
        R_t = self.inter_t(Z_t, ctx_for_t, query_mask=pad_mask, context_mask=pad_mask)

        # Audio attends to (text + visual)
        ctx_for_a = (Z_t + Z_v) / 2.0
        R_a = self.inter_a(Z_a, ctx_for_a, query_mask=pad_mask, context_mask=pad_mask)

        # Visual attends to (text + audio)
        ctx_for_v = (Z_t + Z_a) / 2.0
        R_v = self.inter_v(Z_v, ctx_for_v, query_mask=pad_mask, context_mask=pad_mask)

        return R_t, R_a, R_v


# ═════════════════════════════════════════════════════════════════════════════
# 7. Hierarchical Gated Fusion  (Eq. 7–11)
# ─────────────────────────────────────────────────────────────────────────────
# Unimodal-level gated fusion (Eq. 7–9):
#   g_m  = σ( W_g · [R_t, R_a, R_v] + b_g )      gate (Eq. 7)
#   F_m  = g_m ⊙ R_m                               modality-specific (Eq. 8)
#   U    = ReLU( W_u · [F_t, F_a, F_v] + b_u )   unimodal fusion (Eq. 9)
#
# Multimodal-level gated fusion (Eq. 10–11):
#   g_mm = σ( W_mm · [R_t, R_a, R_v, U] + b_mm ) (Eq. 10)
#   M    = g_mm ⊙ [R_t, R_a, R_v, U]             (Eq. 11)
# ═════════════════════════════════════════════════════════════════════════════
class HierarchicalGatedFusion(nn.Module):
    """
    Two-stage hierarchical gated fusion (Eq. 7–11).
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        D = hidden_dim

        # ── Unimodal-level gate (Eq. 7) ─────────────────────────────────────
        # Concatenation of all three modalities → gate weights for each modality
        self.gate_t = nn.Linear(3 * D, D)
        self.gate_a = nn.Linear(3 * D, D)
        self.gate_v = nn.Linear(3 * D, D)

        # Unimodal fusion (Eq. 9): [F_t, F_a, F_v] → U
        self.uni_fusion = nn.Sequential(
            nn.Linear(3 * D, D),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(D),
        )

        # ── Multimodal-level gate (Eq. 10) ───────────────────────────────────
        # Concatenation of [R_t, R_a, R_v, U] → gate weights
        self.mm_gate = nn.Linear(4 * D, 4 * D)

        # Final projection to D
        self.mm_proj = nn.Sequential(
            nn.Linear(4 * D, D),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(D),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        R_t: torch.Tensor,   # (B, T, D)
        R_a: torch.Tensor,
        R_v: torch.Tensor,
    ) -> torch.Tensor:
        """Returns fused representation M of shape (B, T, D)."""
        cat = torch.cat([R_t, R_a, R_v], dim=-1)   # (B, T, 3D)

        # ── Eq. 7: Compute gates ─────────────────────────────────────────────
        g_t = torch.sigmoid(self.gate_t(cat))       # (B, T, D)
        g_a = torch.sigmoid(self.gate_a(cat))
        g_v = torch.sigmoid(self.gate_v(cat))

        # ── Eq. 8: Gated modality representations ────────────────────────────
        F_t = g_t * R_t
        F_a = g_a * R_a
        F_v = g_v * R_v

        # ── Eq. 9: Unimodal-level fusion ─────────────────────────────────────
        U = self.uni_fusion(torch.cat([F_t, F_a, F_v], dim=-1))  # (B, T, D)

        # ── Eq. 10: Multimodal gate ───────────────────────────────────────────
        mm_cat = torch.cat([R_t, R_a, R_v, U], dim=-1)           # (B, T, 4D)
        g_mm   = torch.sigmoid(self.mm_gate(mm_cat))              # (B, T, 4D)

        # ── Eq. 11: Gated multimodal fusion ──────────────────────────────────
        M_raw = g_mm * mm_cat                                      # (B, T, 4D)
        M     = self.mm_proj(M_raw)                                # (B, T, D)

        return M


# ═════════════════════════════════════════════════════════════════════════════
# 8. Emotion Classifier  (Eq. 12–13)
# ─────────────────────────────────────────────────────────────────────────────
# Eq. 12:  logits = W_c · M + b_c
# Eq. 13:  ŷ      = softmax(logits)
# ═════════════════════════════════════════════════════════════════════════════
class EmotionClassifier(nn.Module):
    """FC layer with softmax for emotion prediction (Eq. 12–13)."""

    def __init__(self, hidden_dim: int, num_classes: int,
                 dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:  x: (B, T, D) or (B*T, D)
        Returns: logits (B, T, C) or (B*T, C)
        """
        return self.classifier(x)


# ═════════════════════════════════════════════════════════════════════════════
# 9. Self-Distillation Module  (Eq. 14–21)
# ─────────────────────────────────────────────────────────────────────────────
# Student classifiers for each modality (Eq. 15–16):
#   logits_m = W_s · R_m + b_s    — one student per modality
#
# Cross-entropy student loss (Eq. 17):
#   L_s = Σ_m CE(softmax(logits_m), y)
#
# KL divergence loss (Eq. 18):
#   L_KL = Σ_m KL( σ(logits/T) ‖ σ(logits_teacher/T) ) × T²
#
# Final combined loss (Eq. 19–21):
#   L = γ1 · L_task + γ2 · L_s + γ3 · L_KL
# ═════════════════════════════════════════════════════════════════════════════
class SelfDistillationModule(nn.Module):
    """
    Self-Distillation module.
    Adds lightweight student classifiers per modality and computes
    the combined distillation + task loss.
    """

    def __init__(
        self,
        hidden_dim:  int,
        num_classes: int,
        temperature: float = 4.0,
        gamma1: float = 1.0,
        gamma2: float = 1.0,
        gamma3: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.T      = temperature
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3

        # ── Student classifiers (Eq. 15–16) ──────────────────────────────────
        self.student_t = EmotionClassifier(hidden_dim, num_classes, dropout)
        self.student_a = EmotionClassifier(hidden_dim, num_classes, dropout)
        self.student_v = EmotionClassifier(hidden_dim, num_classes, dropout)

    def forward(
        self,
        R_t:            torch.Tensor,   # (B*T, D)
        R_a:            torch.Tensor,
        R_v:            torch.Tensor,
        teacher_logits: torch.Tensor,   # (B*T, C)  — from main classifier
        labels:         torch.Tensor,   # (B*T,)    — ground truth, -1 for padding
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            total_loss (scalar)
            loss_dict  { 'L_task', 'L_student', 'L_kl' }
        """
        mask = labels != -1                              # valid positions

        # ── Eq. 17: Student cross-entropy losses ──────────────────────────────
        logits_st = self.student_t(R_t)                 # (B*T, C)
        logits_sa = self.student_a(R_a)
        logits_sv = self.student_v(R_v)

        def ce_loss(logits, targets):
            return F.cross_entropy(logits[mask], targets[mask])

        L_s = (ce_loss(logits_st, labels)
               + ce_loss(logits_sa, labels)
               + ce_loss(logits_sv, labels))

        # ── Eq. 18: KL divergence loss (teacher → students) ──────────────────
        # Scale logits by temperature before softmax
        teacher_soft = F.log_softmax(teacher_logits[mask] / self.T, dim=-1)

        def kl_loss(logits):
            student_soft = F.softmax(logits[mask] / self.T, dim=-1)
            # KL(student ‖ teacher) × T²
            return F.kl_div(teacher_soft, student_soft, reduction="batchmean") * (self.T ** 2)

        L_kl = (kl_loss(logits_st) + kl_loss(logits_sa) + kl_loss(logits_sv))

        # ── Eq. 19–21: Task loss + combined loss ─────────────────────────────
        L_task = F.cross_entropy(teacher_logits[mask], labels[mask])
        L_total = (self.gamma1 * L_task
                   + self.gamma2 * L_s
                   + self.gamma3 * L_kl)

        return L_total, {
            "L_task":    L_task.detach(),
            "L_student": L_s.detach(),
            "L_kl":      L_kl.detach(),
        }


# ═════════════════════════════════════════════════════════════════════════════
# 10. Full SDT Model
# ═════════════════════════════════════════════════════════════════════════════
class SDTModel(nn.Module):
    """
    Self-Distillation Transformer (SDT) for Multimodal ERC.

    Architecture:
        ModalityEncoder (Eq. 1–6)
          ↓ R_t, R_a, R_v
        HierarchicalGatedFusion (Eq. 7–11)
          ↓ M
        EmotionClassifier (Eq. 12–13)  → teacher logits
        SelfDistillationModule (Eq. 14–21)  → total loss

    Inference: predictions come ONLY from the teacher EmotionClassifier.
    """

    def __init__(
        self,
        text_dim:     int   = 1024,
        audio_dim:    int   = 300,
        visual_dim:   int   = 342,
        hidden_dim:   int   = 256,
        num_classes:  int   = 6,
        num_heads:    int   = 8,
        num_intra:    int   = 2,
        num_inter:    int   = 2,
        ffn_dim:      int   = 512,
        max_speakers: int   = 9,
        conv_kernel:  int   = 3,
        dropout:      float = 0.1,
        temperature:  float = 4.0,
        gamma1:       float = 1.0,
        gamma2:       float = 1.0,
        gamma3:       float = 1.0,
        max_seq_len:  int   = 512,
    ):
        super().__init__()

        # Sub-modules
        self.encoder = ModalityEncoder(
            text_dim=text_dim,
            audio_dim=audio_dim,
            visual_dim=visual_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_intra=num_intra,
            num_inter=num_inter,
            ffn_dim=ffn_dim,
            max_speakers=max_speakers,
            conv_kernel=conv_kernel,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.fusion = HierarchicalGatedFusion(hidden_dim, dropout)
        self.classifier = EmotionClassifier(hidden_dim, num_classes, dropout)
        self.distill = SelfDistillationModule(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            temperature=temperature,
            gamma1=gamma1,
            gamma2=gamma2,
            gamma3=gamma3,
            dropout=dropout,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Forward pass
    # ─────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        text:        torch.Tensor,             # (B, T, D_t)
        audio:       torch.Tensor,             # (B, T, D_a)
        visual:      torch.Tensor,             # (B, T, D_v)
        speaker_ids: torch.Tensor,             # (B, T)
        mask:        Optional[torch.Tensor] = None,  # (B, T) — True at VALID pos
        labels:      Optional[torch.Tensor] = None,  # (B, T) — None at inference
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict:
          - 'logits':     (B, T, C) teacher logits
          - 'loss':       scalar total loss (only when labels provided)
          - 'L_task':     task CE loss component
          - 'L_student':  student CE loss component
          - 'L_kl':       KL distillation component
        """
        B, T, _ = text.shape

        # Padding mask: True at PAD positions (for TransformerEncoder)
        if mask is not None:
            pad_mask = ~mask                                  # (B, T)
        else:
            pad_mask = None

        # ── Encoder (Eq. 1–6) ────────────────────────────────────────────────
        R_t, R_a, R_v = self.encoder(text, audio, visual, speaker_ids, pad_mask)

        # ── Hierarchical Gated Fusion (Eq. 7–11) ─────────────────────────────
        M = self.fusion(R_t, R_a, R_v)                       # (B, T, D)

        # ── Teacher Classifier (Eq. 12–13) ───────────────────────────────────
        logits = self.classifier(M)                           # (B, T, C)

        out = {"logits": logits}

        # ── Self-Distillation Loss (Eq. 14–21) — only during training ────────
        if labels is not None:
            # Flatten batch × time
            B, T_p, D = R_t.shape
            R_t_flat = R_t.reshape(B * T_p, D)
            R_a_flat = R_a.reshape(B * T_p, D)
            R_v_flat = R_v.reshape(B * T_p, D)
            teacher_flat = logits.reshape(B * T_p, -1)
            labels_flat  = labels.reshape(B * T_p)

            total_loss, loss_components = self.distill(
                R_t_flat, R_a_flat, R_v_flat,
                teacher_flat, labels_flat,
            )
            out["loss"] = total_loss
            out.update(loss_components)

        return out

    # ─────────────────────────────────────────────────────────────────────────
    # Inference — predictions ONLY from the teacher classifier
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        text:        torch.Tensor,
        audio:       torch.Tensor,
        visual:      torch.Tensor,
        speaker_ids: torch.Tensor,
        mask:        Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns predicted class indices (B, T) for valid positions.
        Predictions come SOLELY from the trained teacher EmotionClassifier.
        """
        self.eval()
        out    = self.forward(text, audio, visual, speaker_ids, mask)
        preds  = out["logits"].argmax(dim=-1)      # (B, T)
        return preds


# ─────────────────────────────────────────────────────────────────────────────
# Factory helper
# ─────────────────────────────────────────────────────────────────────────────
def build_model(cfg) -> SDTModel:
    """Instantiate SDTModel from a Config object."""
    m = cfg.model
    return SDTModel(
        text_dim=m.text_dim,
        audio_dim=m.audio_dim,
        visual_dim=m.visual_dim,
        hidden_dim=m.hidden_dim,
        num_classes=m.num_classes,
        num_heads=m.num_heads,
        num_intra=m.num_intra_layers,
        num_inter=m.num_inter_layers,
        ffn_dim=m.ffn_dim,
        max_speakers=m.max_speakers,
        conv_kernel=m.conv_kernel_size,
        dropout=m.dropout,
        temperature=m.temperature,
        gamma1=m.gamma1,
        gamma2=m.gamma2,
        gamma3=m.gamma3,
    )
