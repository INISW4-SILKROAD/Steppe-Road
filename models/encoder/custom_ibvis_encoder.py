import os
from functools import partial

import torch
import torch.nn as nn

from imagebind.models.transformer import MultiheadAttention, SimpleTransformer
from imagebind.models.helpers import (EinOpsRearrange, Normalize,
                            SelectElement)

from imagebind.models.multimodal_preprocessors import (PadIm2Video,
                                             PatchEmbedGeneric,
                                             RGBDTPreprocessor,
                                             SpatioTemporalPosEmbeddingHelper)


class CustomIbvisEncoder(nn.Module):
    def __init__(
        self,
        kernel_size=(2, 14, 14),
        video_frames = 2,
        out_embed_dim=512,
        vision_embed_dim=1280,
        vision_num_blocks=32,
        vision_num_heads=16,
    ):
        super().__init__()

        self.modality_preprocessors = self._create_modality_preprocessors(
            video_frames,
            vision_embed_dim,
            kernel_size,
        )

        self.modality_trunks = self._create_modality_trunks(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,            
        )

        self.modality_heads =  nn.Sequential(
            nn.LayerNorm(normalized_shape=vision_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(vision_embed_dim, out_embed_dim, bias=False),
        )

        self.modality_postprocessors = Normalize(dim=-1)

    def _create_modality_preprocessors(
        self,
        video_frames=2,

        vision_embed_dim=1024,
        kernel_size=(2, 14, 14),
    ):
        rgbt_stem = PatchEmbedGeneric(
            proj_stem=[
                PadIm2Video(pad_type="repeat", ntimes=2),
                nn.Conv3d(
                    in_channels=3,
                    kernel_size=kernel_size,
                    out_channels=vision_embed_dim,
                    stride=kernel_size,
                    bias=False,
                ),
            ]
        )
        rgbt_preprocessor = RGBDTPreprocessor(
            img_size=[3, video_frames, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=rgbt_stem,
            depth_stem=None,
        )

        return rgbt_preprocessor

    def _create_modality_trunks(
        self,
        vision_embed_dim=1024,
        vision_num_blocks=24,
        vision_num_heads=16,

    ):
        modality_trunks = SimpleTransformer(
                embed_dim=vision_embed_dim,
                num_blocks=vision_num_blocks,
                ffn_dropout_rate=0.0,
                drop_path_rate=0.0,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=vision_embed_dim,
                    num_heads=vision_num_heads,
                    bias=True,
                    add_bias_kv=False,
                ),
                pre_transformer_layer=nn.Sequential(
                    nn.LayerNorm(vision_embed_dim, eps=1e-6),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
            )

        return modality_trunks

    def forward(self, inputs):
        outputs = {}
        reduce_list = (
            inputs.ndim >= 5
        )
        if reduce_list:
            B, S = inputs.shape[:2]
            inputs = inputs.reshape(
                B * S, *inputs.shape[2:]
            )
        if inputs is not None:
            inputs = self.modality_preprocessors(
                **{'vision': inputs}
            )

            trunk_inputs = inputs["trunk"]
            head_inputs = inputs["head"]
            inputs = self.modality_trunks(**trunk_inputs)
            inputs = self.modality_heads(
                inputs, **head_inputs
            )
            
            inputs = self.modality_postprocessors(
                inputs
            )
            
            if reduce_list:
                inputs = inputs.reshape(B, S, -1)
                inputs = inputs.mean(dim=1)

            outputs = inputs

        return outputs


def cibv_pretrained(out_embed_dim = 512):
    model = CustomIbvisEncoder(
        vision_embed_dim=1280,
        vision_num_blocks=32,
        vision_num_heads=16,
        out_embed_dim=out_embed_dim
    )

    weight_path = f".checkpoints/pretrained_cibv_{out_embed_dim}.pth"
    if not os.path.exists(weight_path):
        print('WARNING: no checkpoint exist - cant load weight')
        return None

    model.load_state_dict(torch.load(weight_path))

    return model


if __name__ == '__main__':
    from imagebind import data
    from custom_ib_model import ModalityType
    import pickle

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device

    # Instantiate model
    model = cibv_pretrained(out_embed_dim=512)
    model.eval()
    model.to(device)
    # torch.save(model.state_dict(), '.checkpoints/mibm_pretrained.pth')

    with open('./data/data.pkl', 'rb') as f:
        input_data = pickle.load(f)

    frac_list=list(input_data['fraction'].values())
    touch_list=list(input_data['touch'].values())
    image_paths=input_data['image']

    # Load data
    inputs = {
        ModalityType.TEXT : data.load_and_transform_text(frac_list, device),
        ModalityType.TOUCH: data.load_and_transform_text(touch_list, device), 
    }

    with torch.no_grad():
        embeddings = model(inputs)

    print(
    torch.softmax(embeddings[ModalityType.TEXT] @ embeddings[ModalityType.TOUCH].T, dim=-1), 
    )