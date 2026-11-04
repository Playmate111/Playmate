import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
from . import get_network


class VarianceSchedule(Module):
    def __init__(self, num_steps, beta_1, beta_T, mode="linear"):
        super().__init__()
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            s = 0.008
            steps = num_steps + 1
            x = torch.linspace(0, num_steps, steps)
            alpha_bars = torch.cos(((x / num_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alpha_bars = alpha_bars / alpha_bars[0]
            betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
            betas = torch.clip(betas, 0.0001, 0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)  # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sigmas_flex", sigmas_flex)
        self.register_buffer("sigmas_inflex", sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps + 1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (
            1 - flexibility
        )
        return sigmas


class NoisePredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        decoder_class = get_network(cfg.DECODER_TYPE)
        self.decoder = decoder_class(**cfg.DECODER)

    def forward(
        self,
        noise_motion_feat,
        audio_feat,
        style_feat,
        prev_motion_feat,
        prev_audio_feat,
        src_canonical_volume,
        timestep,
        finetune_style=False
    ):
        """
        Args:
            noise_motion_feat: (B, T, 1, C_motion)
            audio_feat: (B, T, 1, C_model)
            style_feat: (B, 256, 4096)
            prev_motion_feat:  (B, PT, 1, C_motion)
            prev_audio_feat: (B, PT, 1, C_model)
            src_canonical_volume:  (B, 1, 1, C_motion)
            timestep: (B)
        Returns:
            pred_epsilon: (B, N, C_motion)
        """
        PN = prev_motion_feat.shape[1]

        audio_enc_feat = torch.cat([prev_audio_feat, audio_feat], dim=1)  # (B, PN + N, 1, C_model)
        motion_enc_feat = torch.cat([prev_motion_feat, noise_motion_feat], dim=1)  # (B, PN + N, 1, C_motion)

        pred_epsilon = self.decoder(
            motion_enc_feat, src_canonical_volume, audio_enc_feat, style_feat, timestep,
            finetune_style=finetune_style
        )[:, PN:]  # (B, N, 1, C_motion)

        return pred_epsilon
