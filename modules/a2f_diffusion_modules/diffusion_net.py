import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from .diffusion_util import NoisePredictor, VarianceSchedule
from .generator import ContentW2VEncoder


class DiffusionNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.denoising_net = NoisePredictor(cfg)
        self.var_sched = VarianceSchedule(
                num_steps=self.cfg.DIFFUSION.SCHEDULE.NUM_STEPS,
                beta_1=self.cfg.DIFFUSION.SCHEDULE.BETA_1,
                beta_T=self.cfg.DIFFUSION.SCHEDULE.BETA_T,
                mode=self.cfg.DIFFUSION.SCHEDULE.MODE,
            )
        self.output_dim = self.cfg.DATASET.FACE3D_DIM

        self.audio_encoder = ContentW2VEncoder(**cfg.CONTENT_ENCODER)
        self.null_audio_feat = nn.Parameter(torch.randn(1, 1, 1, cfg.CONTENT_ENCODER.output_dim))
        self.null_style_feat = torch.zeros(1, cfg.DECODER.d_style)

    def forward(
        self,
        motion_feat,
        audio_feat,
        style_feat,
        prev_motion_feat,
        prev_audio_feat,
        src_canonical_volume,
        indicator,
        finetune_style
    ):
        """
        Args:
            motion_feat: (B, T, 1, C_motion)
            audio_feat: (B, T, 1, C_audio)
            style_feat: (B, 256, 4096)
            prev_motion_feat:  (B, PT, 1, C_motion)
            prev_audio_feat: (B, PT, 1, C_audio)
            src_canonical_volume: (B, 1, 1, c_motion)
            indicator: (B, T)
        Returns:
            pred_epsilon: (B, T, 1, C_motion)
        """

        B, PN = prev_audio_feat.shape[:2]

        audio_feat_in = torch.cat([prev_audio_feat, audio_feat], dim=1)  # (B, PN + N, 1, C_audio)
        if finetune_style:
            with torch.no_grad():
                audio_feat_out = self.audio_encoder(audio_feat_in)
            mask_audio = indicator == 0
            audio_feat_out = torch.where(mask_audio.view(-1, 1, 1, 1), self.null_audio_feat, audio_feat_out)
        else:
            audio_feat_out = self.audio_encoder(audio_feat_in)
        prev_audio_feat, audio_feat = audio_feat_out[:, :PN], audio_feat_out[:, PN:]

        # Classifier-free guidance
        mask_audio = torch.rand(B, device=audio_feat.device) > 0.8
        audio_feat = torch.where(mask_audio.view(-1, 1, 1, 1), self.null_audio_feat, audio_feat)

        mask_style = torch.rand(B, device=style_feat.device) > 0.8
        style_feat = torch.where(
            mask_style.view(-1, 1),
            self.null_style_feat.to(style_feat.device),
            style_feat
        )

        # Sample time step
        timestep = self.var_sched.uniform_sample_t(B)  # (B,)

        # The forward diffusion process
        alpha_bar = self.var_sched.alpha_bars[timestep]  # (B,)
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1, 1)  # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1, 1)  # (B, 1, 1)

        eps = torch.randn_like(motion_feat)  # (B, T, 1, C_motion)
        noise_motion_feat = c0 * motion_feat + c1 * eps  # (B, T, 1, C_motion)

        pred_epsilon = self.denoising_net(
            noise_motion_feat,
            audio_feat,
            style_feat,
            prev_motion_feat,
            prev_audio_feat,
            src_canonical_volume,
            timestep,
            finetune_style=finetune_style
        )

        return eps, pred_epsilon

    def ddim_sample(
        self,
        audio_feat,
        style_feat,
        prev_motion_feat,
        prev_audio_feat,
        src_canonical_volume,
        use_cf_guidance=True,
        cfg_scale=1.0,
        ddim_num_step=10,
    ):
        """
        Args:
            audio_feat: (B, T, 1, C_audio)
            style_feat: (B, 256, 4096)
            timestep: (B)
            prev_motion_feat:  (B, PT, 1, C_motion)
            prev_audio_feat: (B, PT, 1, C_audio)
            src_canonical_volume: (B, 1, 1, c_motion)
        Returns:
            denoise_motion_feat: (B, T, 1, C_motion)
        """

        B, PN = prev_audio_feat.shape[:2]
        audio_feat_in = torch.cat([prev_audio_feat, audio_feat], dim=1)  # (B, PN + N, 1, C_audio)
        audio_feat_out = self.audio_encoder(audio_feat_in)
        prev_audio_feat, audio_feat = audio_feat_out[:, :PN], audio_feat_out[:, PN:]

        c = self.var_sched.num_steps // ddim_num_step
        time_steps = torch.tensor(
            np.asarray(list(range(0, self.var_sched.num_steps, c))) + 1
        )
        assert len(time_steps) == ddim_num_step
        prev_time_steps = torch.cat((torch.tensor([0]), time_steps[:-1]))

        B, N = audio_feat.shape[:2]

        if use_cf_guidance:
            audio_feat = torch.cat([audio_feat, self.null_audio_feat.repeat(B, N, 1, 1)], dim=0)
            # audio_feat = torch.cat([audio_feat] * 2, dim=0)
            # style_feat = torch.cat([style_feat, self.null_style_feat.repeat(B, 1, 1)], dim=0)
            prev_motion_feat = torch.cat([prev_motion_feat] * 2, dim=0)
            prev_audio_feat = torch.cat([prev_audio_feat] * 2, dim=0)
            src_canonical_volume = torch.cat([src_canonical_volume] * 2, dim=0)

        x_t = torch.randn([B, N, 1, self.output_dim]).to(audio_feat.device)
        for idx in tqdm(list(range(ddim_num_step))[::-1]):
            t = time_steps[idx]
            t_prev = prev_time_steps[idx]
            ddim_alpha = self.var_sched.alpha_bars[t]
            ddim_alpha_prev = self.var_sched.alpha_bars[t_prev]
            # sigma = self.var_sched.get_sigmas(t, 0)
            t = torch.tensor([t] * B).to(audio_feat.device).float()

            if use_cf_guidance:
                x_t_double = torch.cat([x_t] * 2, dim=0)
                t_double = torch.cat([t] * 2, dim=0)
                cond_output, uncond_output = self.denoising_net(
                    x_t_double,
                    audio_feat,
                    style_feat,
                    prev_motion_feat,
                    prev_audio_feat,
                    src_canonical_volume,
                    t_double.tolist()
                ).chunk(2)
                diff_output = uncond_output + cfg_scale * (cond_output - uncond_output)
            else:
                diff_output = self.denoising_net(
                    x_t,
                    audio_feat,
                    style_feat,
                    prev_motion_feat,
                    prev_audio_feat,
                    src_canonical_volume,
                    t.tolist()
                )

            eps = (x_t - torch.sqrt(ddim_alpha) * diff_output) / torch.sqrt(1 - ddim_alpha)
            c1 = torch.sqrt(ddim_alpha_prev)
            c2 = torch.sqrt(1 - ddim_alpha_prev)
            x_t = c1 * diff_output + c2 * eps

        denoise_motion_feat = x_t

        return denoise_motion_feat

    def ddpm_sample(
        self,
        audio_feat,
        style_feat,
        prev_motion_feat,
        prev_audio_feat,
        src_canonical_volume,
        use_cf_guidance=True,
        cfg_scale=[1.0, 1.0],
        target_mode='x0',
        add_style=False,
        cfg_mode='incremental'
    ):
        """
        Args:
            audio_feat: (B, T, 1, C_audio)
            style_feat: (B, 256, 4096)
            timestep: (B)
            prev_motion_feat:  (B, PT, 1, C_motion)
            prev_audio_feat: (B, PT, 1, C_audio)
            src_canonical_volume: (B, 1, 1, c_motion)
        Returns:
            denoise_motion_feat: (B, T, 1, C_motion)
        """

        B, PN = prev_audio_feat.shape[:2]
        audio_feat_in = torch.cat([prev_audio_feat, audio_feat], dim=1)  # (B, PN + N, 1, C_audio)
        audio_feat_out = self.audio_encoder(audio_feat_in)
        prev_audio_feat, audio_feat = audio_feat_out[:, :PN], audio_feat_out[:, PN:]

        B, N = audio_feat.shape[:2]

        device = audio_feat.device

        if use_cf_guidance:
            if add_style:
                conds = ['audio', 'style']
            else:
                conds = ['audio']
            null_audio_feat = self.null_audio_feat.repeat(B, N, 1, 1)
            null_style_feat = self.null_style_feat.repeat(B, 1).to(device)
            audio_feat_in = [null_audio_feat]
            style_feat_in = [null_style_feat]
            for cond in conds:
                if cond == 'audio':
                    audio_feat_in.append(audio_feat)
                    style_feat_in.append(null_style_feat)
                elif cond == 'style':
                    if cfg_mode == 'independent':
                        audio_feat_in.append(null_audio_feat)
                    elif cfg_mode == 'incremental':
                        audio_feat_in.append(audio_feat)
                    style_feat_in.append(style_feat)

            n_entries = len(audio_feat_in)
            audio_feat = torch.cat(audio_feat_in, dim=0)
            style_feat = torch.cat(style_feat_in, dim=0)
            prev_motion_feat = torch.cat([prev_motion_feat] * n_entries, dim=0)
            prev_audio_feat = torch.cat([prev_audio_feat] * n_entries, dim=0)
            src_canonical_volume = torch.cat([src_canonical_volume] * n_entries, dim=0)

        x_T = torch.randn([B, N, 1, self.output_dim]).to(audio_feat.device)
        traj = {self.var_sched.num_steps: x_T}
        for t in tqdm(range(self.var_sched.num_steps, 0, -1)):
            if t > 1:
                z = torch.randn_like(x_T)
            else:
                z = torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            alpha_bar_prev = self.var_sched.alpha_bars[t - 1]
            sigma = self.var_sched.get_sigmas(t, 0)

            x_t = traj[t]
            t_step = torch.tensor([t] * B).to(audio_feat.device).float()
            if use_cf_guidance:
                x_t_double = torch.cat([x_t] * n_entries, dim=0)
                t_step_double = torch.cat([t_step] * n_entries, dim=0)
                results = self.denoising_net(
                    x_t_double,
                    audio_feat,
                    style_feat,
                    prev_motion_feat,
                    prev_audio_feat,
                    src_canonical_volume,
                    t_step_double.tolist(),
                    finetune_style=add_style
                ).chunk(n_entries)
                results = results
                diff_output = results[0]
                for i in range(0, n_entries - 1):
                    if cfg_mode == 'independent':
                        diff_output += cfg_scale[i] * (results[i + 1] - results[0])
                    elif cfg_mode == 'incremental':
                        diff_output += cfg_scale[i] * (results[i + 1] - results[i])

            else:
                diff_output = self.denoising_net(
                    x_t,
                    audio_feat,
                    style_feat,
                    prev_motion_feat,
                    prev_audio_feat,
                    src_canonical_volume,
                    t_step.tolist(),
                    finetune_style=add_style
                )
            if target_mode == 'noise':
                c0 = 1 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                motion_next = c0 * (x_t - c1 * diff_output) + sigma * z
            elif target_mode == 'x0':
                c0 = (1 - alpha_bar_prev) * torch.sqrt(alpha) / (1 - alpha_bar)
                c1 = (1 - alpha) * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar)
                motion_next = c0 * x_t + c1 * diff_output + sigma * z

            traj[t - 1] = motion_next

        denoise_motion_feat = traj[0]

        return denoise_motion_feat
