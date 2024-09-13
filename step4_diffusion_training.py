
import torch
import torch.nn as nn

from diffusers import UNet2DModel




class DiffusionModel(nn.Module):
    def __init__(self):
        
        self.unet = UNet2DModel(
            sample_size=256,
            in_channels=1,
            out_channels=1,
            center_input_sample=False,
            time_embedding_type='positional',
            freq_shift=0,
            flip_sin_to_cos=True,
            down_block_types=('DownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D'),
            up_block_types=('AttnUpBlock2D', 'AttnUpBlock2D', 'AttnUpBlock2D', 'UpBlock2D'),
            block_out_channels=(224, 448, 672, 896),
            layers_per_block=2,
            mid_block_scale_factor=1,
            downsample_padding=1,
            downsample_type='conv',
            upsample_type='conv',
            dropout=0.0,
            act_fn='prelu',
            attention_head_dim=None,
            norm_num_groups=32,
            attn_norm_num_groups=None,
            norm_eps=1e-05,
            resnet_time_scale_shift='default',
            add_attention=True,
            class_embed_type=None,
            num_class_embeds=None,
            num_train_timesteps=None
        )

    def sample_x_t_given_x_0(self, x_0, t):
        return x_0 + t * t * torch.randn_like(x_0)
    