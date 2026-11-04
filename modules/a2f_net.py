import torch.nn as nn

from .a2f_diffusion_modules.diffusion_net import DiffusionNet


class A2FDiffNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.diff_net = DiffusionNet(cfg=cfg)

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
        noise, pred_epsilon = self.diff_net(
            motion_feat, audio_feat, style_feat, prev_motion_feat, prev_audio_feat, src_canonical_volume, indicator,
            finetune_style
        )

        return noise, pred_epsilon

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
        denoise_motion_feat = self.diff_net.ddim_sample(
            audio_feat,
            style_feat,
            prev_motion_feat,
            prev_audio_feat,
            src_canonical_volume,
            use_cf_guidance=use_cf_guidance,
            cfg_scale=cfg_scale,
            ddim_num_step=ddim_num_step,
        )

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
        add_style=False
    ):
        denoise_motion_feat = self.diff_net.ddpm_sample(
            audio_feat,
            style_feat,
            prev_motion_feat,
            prev_audio_feat,
            src_canonical_volume,
            use_cf_guidance=use_cf_guidance,
            cfg_scale=cfg_scale,
            target_mode=target_mode,
            add_style=add_style
        )

        return denoise_motion_feat

