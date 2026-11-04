import math

import torch
from torch import nn
from einops import rearrange
from espnet.nets.pytorch_backend.conformer.encoder import Encoder as ConformerEncoder
from .dit_models import DiTBlock


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=600):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # vanilla sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, x.shape[1], :]
        return self.dropout(x)


def enc_dec_mask(T, S, frame_width=2, expansion=0):
    mask = torch.ones(T, S)
    for i in range(T):
        mask[i, max(0, (i - expansion) * frame_width):(i + expansion + 1) * frame_width] = 0

    return (mask == 1)


class ContentW2VEncoder(nn.Module):
    def __init__(
        self,
        seq_len=5,
        blocks=12,
        channels=768,
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = (
            seq_len * blocks * channels
        )  # update input_dim to be the product of blocks and channels.
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, audio_embeds):
        video_length = audio_embeds.shape[1]
        audio_embeds = rearrange(audio_embeds, "bz f b c -> (bz f) b c")
        batch_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, blocks * channels)

        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds = torch.relu(self.proj2(audio_embeds))
        context_tokens = self.proj3(audio_embeds)
        context_tokens = self.norm(context_tokens)

        context_tokens = rearrange(
            context_tokens, "(bz f) c -> bz f c", f=video_length
        )

        context_tokens = rearrange(
            context_tokens, "bz f (m c) -> bz f m c", c=self.output_dim
        )

        return context_tokens


class StyleDit(nn.Module):
    def __init__(
            self,
            d_style=11,
            d_model=512,
            nhead=1,
            num_encoder_layers=2,
    ):
        super().__init__()
        self.d_model = d_model
        self.y_embedder = nn.Linear(d_style, d_model)
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, nhead, mlp_ratio=4.0) for _ in range(num_encoder_layers)
        ])

    def forward(self, motion, style, timestep_emb):
        y = self.y_embedder(style)
        c = timestep_emb + y  # (N, D)
        for block in self.blocks:
            motion = block(motion, c)  # (N, T, D)

        return motion


class NoiseDecoder(nn.Module):
    def __init__(
        self,
        d_model=512,
        d_style=11,
        output_dim=79,
        n_diff_steps=1000,
    ) -> None:
        super().__init__()

        self.feature_dim = d_model
        self.motion_feat_dim = output_dim

        self.TE = PositionalEncoding(self.feature_dim, max_len=n_diff_steps + 1)
        self.diff_step_map = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        self.motion_proj = nn.Linear(self.motion_feat_dim, self.feature_dim)
        self.person_proj = nn.Linear(self.motion_feat_dim, self.feature_dim)

        self.conformer = ConformerEncoder(
            idim=0, attention_dim=self.feature_dim * 4, attention_heads=2, linear_units=self.feature_dim * 4,
            num_blocks=5, input_layer=None, dropout_rate=0.2, positional_dropout_rate=0.2,
            attention_dropout_rate=0.2, normalize_before=False, concat_after=False,
            positionwise_layer_type="linear", positionwise_conv_kernel_size=3, macaron_style=True,
            pos_enc_layer_type="rel_pos", selfattention_layer_type="rel_selfattn", use_cnn_module=True,
            cnn_module_kernel=13)

        self.motion_decoder = nn.Sequential(
            nn.Linear(self.feature_dim * 4, self.feature_dim),
        )

        self.exp_motion_decoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.SiLU(),
            nn.Linear(self.feature_dim // 2, 45)
        )

        self.pose_motion_decoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.SiLU(),
            nn.Linear(self.feature_dim // 2, 7)
        )

        self.style_encoder = StyleDit(d_style=d_style)

    def audio_forward(self, noise_motion_feat, person_feat, audio_feat, style_feat, timestep):
        B, PNN, W_a, C_a = audio_feat.shape

        diff_step_embedding = self.diff_step_map(self.TE.pe[0, timestep])[:, None, None, :].repeat(1, PNN, 1,
                                                                                                   1)  # (B, 1, 1, C_dmodel)
        person_feat = self.person_proj(person_feat).repeat(1, PNN, 1, 1)  # (B, 1, 1, C_dmodel)
        feat_in = self.motion_proj(noise_motion_feat)  # (B, PNN, 1, C_dmodel)
        feat_in = torch.cat([person_feat, diff_step_embedding, feat_in, audio_feat],
                            dim=-1)  # (B, PNN, 1, C_dmodel * 5)

        feat_in = feat_in.reshape(B, PNN, -1)
        feat_out, _ = self.conformer(feat_in, masks=None)  # (B, PNN, C_motion * 5)
        feat_out = feat_out.reshape(B, PNN, 1, -1)  # (B, PNN, 1, C_motion * 5)
        pred_epsilon = self.motion_decoder(feat_out)  # (B, PNN, 1, C_motion)
        pred_epsilon = pred_epsilon.reshape(B, PNN, 1, -1)

        return pred_epsilon, diff_step_embedding

    def forward(self, noise_motion_feat, person_feat, audio_feat, style_feat, timestep, finetune_style=False):
        """
        Args:
            noise_motion_feat: (B, PN + N, 1, C_motion)
            person_feat: (B, 1, 1, C_motion)
            audio_feat: (B, PN + N, 1, C_dmodel)
            style_feat: (B, 512)
            timestep: (B)
        Returns:
            pred_epsilon: (B, PN + N, 1, C_motion)
        """
        B, PNN, W_a, C_a = audio_feat.shape

        if finetune_style:
            with torch.no_grad():
                pred_epsilon, diff_step_embedding = self.audio_forward(
                    noise_motion_feat, person_feat, audio_feat, style_feat, timestep
                )
            exp_motion_feat = self.style_encoder(
                pred_epsilon.reshape(B, PNN, -1), style_feat, diff_step_embedding[:, 0].reshape(B, -1)
            )
            exp_pred_epsilon = exp_motion_feat.reshape(B, PNN, 1, -1)

        else:
            pred_epsilon, _ = self.audio_forward(
                noise_motion_feat, person_feat, audio_feat, style_feat, timestep
            )
            exp_pred_epsilon = pred_epsilon

        exp_pred_epsilon = self.exp_motion_decoder(exp_pred_epsilon)
        pose_pred_epsilon = self.pose_motion_decoder(pred_epsilon)
        pred_epsilon = torch.cat([exp_pred_epsilon, pose_pred_epsilon], dim=-1)

        return pred_epsilon
