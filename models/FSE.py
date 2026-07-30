"""
FSE: Future-distilled Spectral Enhancement (post-hoc, model-agnostic).

Wraps a frozen, pretrained baseline forecaster and refines its prediction in
the frequency domain. During training, a future teacher branch encodes the
ground-truth future spectrum as privileged information and distills it into
a past-conditioned student representation via a non-contrastive
dual-alignment objective. At inference, only the student branch and the
guided spectral correction network are used (the teacher branch and the
reconstruction head are skipped entirely).

Usage: this model must be run with task_name='fse_forecast' (see
exp/exp_fse_forecasting.py), because its forward() needs access to the
ground-truth future during training, which the generic long_term_forecast
training loop does not provide.

Required configs:
    fse_baseline       : name of the baseline model file under models/
                         (e.g. 'TimesNet', 'DLinear', 'PatchTST', ...)
    fse_baseline_ckpt  : path to the pretrained (stage 1) baseline checkpoint
Optional configs (defaults follow the paper's reported hyperparameters):
    fse_layers, fse_alpha, fse_eps, fse_gamma, fse_eta,
    fse_lambda_cos, fse_lambda_ema, fse_ema_beta,
    fse_corr_hidden, fse_corr_kernel_size
"""
import copy
import importlib

import torch
import torch.nn as nn

from layers.FSE_blocks import (
    spectral_feature,
    SharedResidualEncoder,
    TokenProjection,
    ReconstructionHead,
    DualAlignmentLoss,
    GuidedSpectralCorrectionNetwork,
)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_feat = 2 * self.enc_in  # D = 2M, real+imag concatenation
        self.k_y = self.pred_len // 2 + 1
        self.k_x = self.seq_len // 2 + 1

        n_layers = getattr(configs, 'fse_layers', 3)
        self.n_layers = n_layers

        # ---- frozen baseline forecaster (stage 1, pretrained) ----
        if not getattr(configs, 'fse_baseline', None):
            raise ValueError("FSE requires '--fse_baseline <model_name>' pointing to the frozen baseline.")
        baseline_module = importlib.import_module(f"models.{configs.fse_baseline}")
        baseline_configs = copy.copy(configs)
        baseline_configs.task_name = 'long_term_forecast'
        self.baseline = baseline_module.Model(baseline_configs).float()

        ckpt_path = getattr(configs, 'fse_baseline_ckpt', None)
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location='cpu')
            self.baseline.load_state_dict(state_dict)
        else:
            print("[FSE] Warning: no --fse_baseline_ckpt given; baseline is randomly initialized.")

        for p in self.baseline.parameters():
            p.requires_grad_(False)
        self.baseline.eval()

        # ---- shared teacher/student residual encoder ----
        corr_hidden = getattr(configs, 'fse_corr_hidden', 128)
        self.shared_encoder = SharedResidualEncoder(self.d_feat, n_layers, hidden_dim=self.d_feat)
        self.token_proj = TokenProjection(self.k_x, self.k_y, n_layers)
        self.recon_head = ReconstructionHead(self.d_feat)
        self.align_loss = DualAlignmentLoss(
            self.d_feat, n_layers,
            ema_beta=getattr(configs, 'fse_ema_beta', 0.8),
            lambda_cos=getattr(configs, 'fse_lambda_cos', 0.8),
            lambda_ema=getattr(configs, 'fse_lambda_ema', 0.2),
        )
        self.correction_net = GuidedSpectralCorrectionNetwork(
            d_context=self.d_feat + self.d_feat,  # psi(F0_y) [2M] + student latent [D=2M]
            m_channels=self.enc_in,
            hidden_dim=corr_hidden,
            kernel_size=getattr(configs, 'fse_corr_kernel_size', 3),
            alpha=getattr(configs, 'fse_alpha', 0.5),
            eps=getattr(configs, 'fse_eps', 1e-6),
        )

    def train(self, mode=True):
        # Baseline must always stay in eval mode: it is frozen post-hoc and
        # must never pick up BatchNorm/Dropout training-mode side effects.
        super().train(mode)
        self.baseline.eval()
        return self

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, y_true=None):
        with torch.no_grad():
            y0 = self.baseline(x_enc, x_mark_enc, x_dec, x_mark_dec)
            y0 = y0[:, -self.pred_len:, :].detach()
        f0_y, _ = spectral_feature(y0)  # complex [B, Ky, M]

        _, psi_x0 = spectral_feature(x_enc)   # [B, Kx, D]
        zs_x = self.shared_encoder(psi_x0)    # list length N+1
        z_x_layers = self.token_proj(zs_x)    # list length N, each [B, Ky, D]
        z_x_last = z_x_layers[-1]

        l_recon, l_align = None, None
        if self.training and y_true is not None:
            y_future = y_true[:, -self.pred_len:, :]
            _, psi_y0 = spectral_feature(y_future)  # z^0_y = psi(F_y), [B, Ky, D]
            zs_y = self.shared_encoder(psi_y0)
            z_y_layers = zs_y[1:]  # list length N, each [B, Ky, D]

            recon_pred = self.recon_head(zs_y[-1])
            l_recon = (recon_pred - psi_y0).abs().mean()
            l_align = self.align_loss(z_x_layers, z_y_layers)

        f_hat = self.correction_net(f0_y, z_x_last)
        y_hat = torch.fft.irfft(f_hat, n=self.pred_len, dim=1)

        if self.training:
            return y_hat, l_recon, l_align
        return y_hat
