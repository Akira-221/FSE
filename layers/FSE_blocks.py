"""
Building blocks for Future-distilled Spectral Enhancement (FSE).

Reference: "Future-distilled Spectral Enhancement for Long-term Time Series
Forecasting" (AAAI submission). FSE is a post-hoc, model-agnostic module that
corrects the spectral bias of a frozen baseline forecaster. During training a
future teacher branch distills ground-truth spectral information into a past
student branch through a non-contrastive dual-alignment objective; at
inference only the student branch and the guided spectral correction network
are used.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def spectral_feature(x):
    """
    x: [B, L, M] real signal along the time dimension (dim=1).
    Returns:
        F_complex: [B, K, M] complex spectrum, K = L // 2 + 1
        psi: [B, K, 2M] real feature, concatenation of (Re, Im) along channels
    """
    f_complex = torch.fft.rfft(x, dim=1)
    psi = torch.cat([f_complex.real, f_complex.imag], dim=-1)
    return f_complex, psi


class ResidualMLPLayer(nn.Module):
    """z^n = z^{n-1} + Linear(act(Linear(z^{n-1})))"""

    def __init__(self, dim, hidden_dim=None, activation='gelu'):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()

    def forward(self, z):
        return z + self.fc2(self.act(self.fc1(z)))


class SharedResidualEncoder(nn.Module):
    """N-layer residual encoder shared by the teacher and student branches."""

    def __init__(self, dim, n_layers, hidden_dim=None):
        super().__init__()
        self.layers = nn.ModuleList([ResidualMLPLayer(dim, hidden_dim) for _ in range(n_layers)])

    def forward(self, z0):
        """Returns [z^0, z^1, ..., z^N], length n_layers + 1."""
        zs = [z0]
        z = z0
        for layer in self.layers:
            z = layer(z)
            zs.append(z)
        return zs


class TokenProjection(nn.Module):
    """
    Per-layer projection P_x^n: R^{Kx x D} -> R^{Ky x D}, applied along the
    frequency-token axis so the student representation can be aligned with
    the teacher's token count when the lookback window and horizon differ.
    """

    def __init__(self, k_in, k_out, n_layers):
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(k_in, k_out) for _ in range(n_layers)])

    def forward(self, zs):
        """
        zs: encoder output list of length n_layers + 1 (zs[0] is the
        pre-encoder feature and is skipped; only z^1..z^N are projected).
        Returns a list of length n_layers, each [B, Ky, D].
        """
        out = []
        for n, z in enumerate(zs[1:]):
            zt = z.transpose(1, 2)      # [B, D, Kx]
            zt = self.projs[n](zt)      # [B, D, Ky]
            out.append(zt.transpose(1, 2))  # [B, Ky, D]
        return out


class ReconstructionHead(nn.Module):
    """Regresses the teacher's final-layer representation back to psi(F_y),
    keeping the teacher representation informative rather than collapsing to
    an arbitrary alignment target."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, z_last):
        return self.proj(z_last)


class DualAlignmentLoss(nn.Module):
    """
    Non-contrastive dual-alignment objective derived from the Pearson chi^2
    f-mutual-information instantiation:
      - sample-wise cosine alignment (matched-pair term, teacher stop-grad)
      - EMA-based marginal regularizer (replaces the explicit cross-sample
        negative term with a running second-moment matrix of the teacher
        representation).

    Representations are treated as [B, K, D] token sequences; the marginal
    second-moment matrix is estimated over both the batch and the token axis
    to keep it a tractable D x D matrix.
    """

    def __init__(self, dim, n_layers, ema_beta=0.8, lambda_cos=0.8, lambda_ema=0.2):
        super().__init__()
        self.n_layers = n_layers
        self.ema_beta = ema_beta
        self.lambda_cos = lambda_cos
        self.lambda_ema = lambda_ema
        for n in range(n_layers):
            self.register_buffer(f'ema_cy_{n}', torch.zeros(dim, dim))
        self.register_buffer('ema_ready', torch.zeros(n_layers, dtype=torch.bool))

    def forward(self, z_x_layers, z_y_layers):
        """
        z_x_layers, z_y_layers: lists of length n_layers, each [B, K, D],
        already token-aligned (student side has been projected to Ky tokens).
        """
        total_cos = 0.0
        total_ema = 0.0
        for n in range(self.n_layers):
            zx = F.normalize(z_x_layers[n], dim=-1)
            zy = F.normalize(z_y_layers[n], dim=-1).detach()  # stop-gradient on teacher side

            # --- sample-wise cosine alignment (matched-pair term) ---
            cos_sim = (zx * zy).sum(-1)  # [B, K]
            l_cos = (1.0 - cos_sim).mean()

            # --- EMA marginal regularization (replaces explicit negatives) ---
            b, k, d = zy.shape
            flat_zy = zy.reshape(b * k, d)
            c_y = (flat_zy.t() @ flat_zy) / (b * k)

            ema_buf = getattr(self, f'ema_cy_{n}')
            if self.training:
                with torch.no_grad():
                    if not bool(self.ema_ready[n]):
                        ema_buf.copy_(c_y.detach())
                        self.ema_ready[n] = True
                    else:
                        ema_buf.mul_(self.ema_beta).add_(c_y.detach(), alpha=1.0 - self.ema_beta)

            quad = torch.einsum('bkd,de,bke->bk', zx, ema_buf, zx)
            l_ema = quad.mean()

            total_cos = total_cos + l_cos
            total_ema = total_ema + l_ema

        total_cos = total_cos / self.n_layers
        total_ema = total_ema / self.n_layers
        return self.lambda_cos * total_cos + self.lambda_ema * total_ema


class GuidedSpectralCorrectionNetwork(nn.Module):
    """
    Predicts a bounded, sample-dependent correction to the frozen baseline's
    spectrum, guided by the distilled student representation:
        Theta = min(1, alpha * |F0_y| / (|dF_raw| + eps))
        dF_y  = Theta * dF_raw
        F_hat = F0_y + dF_y
    The risk-control operator Theta prevents the correction from overwriting
    reliable parts of the baseline spectrum.
    """

    def __init__(self, d_context, m_channels, hidden_dim=128, kernel_size=3, alpha=0.5, eps=1e-6):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(d_context, hidden_dim, kernel_size, padding=padding)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_dim, 2 * m_channels, kernel_size, padding=padding)
        self.alpha = alpha
        self.eps = eps
        self.m_channels = m_channels

    def forward(self, f0_y, z_x_last):
        """
        f0_y: complex [B, Ky, M] baseline spectrum
        z_x_last: real [B, Ky, D] student final-layer representation
        Returns the corrected complex spectrum F_hat: [B, Ky, M]
        """
        psi_f0 = torch.cat([f0_y.real, f0_y.imag], dim=-1)  # [B, Ky, 2M]
        g = torch.cat([psi_f0, z_x_last], dim=-1)           # [B, Ky, 2M + D]
        g = g.transpose(1, 2)                               # [B, C_in, Ky]

        h = self.act(self.conv1(g))
        raw = self.conv2(h)                                 # [B, 2M, Ky]
        raw = raw.transpose(1, 2)                            # [B, Ky, 2M]

        real, imag = raw[..., :self.m_channels], raw[..., self.m_channels:]
        delta_raw = torch.complex(real, imag)                # [B, Ky, M]

        mod_delta = delta_raw.abs()
        mod_f0 = f0_y.abs()
        theta = torch.clamp(self.alpha * mod_f0 / (mod_delta + self.eps), max=1.0)
        delta_f = theta.to(delta_raw.dtype) * delta_raw

        return f0_y + delta_f
