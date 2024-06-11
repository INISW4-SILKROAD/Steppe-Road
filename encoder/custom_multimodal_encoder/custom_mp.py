import torch
from imagebind.models.multimodal_preprocessors import (PadIm2Video,
                                             PatchEmbedGeneric,
                                             RGBDTPreprocessor,
                                             SpatioTemporalPosEmbeddingHelper,
                                             TextPreprocessor)


class TouchPreprocessor(TextPreprocessor):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int,
        causal_masking: bool,
        supply_seq_len_to_head: bool = True,
        num_cls_tokens: int = 0,
        init_param_style: str = "openclip",
    ) -> None:
        super().__init__(        
            vocab_size,
            context_length,
            embed_dim,
            causal_masking,
            supply_seq_len_to_head,
            num_cls_tokens,
            init_param_style
        )

    def forward(self, touch):
        # touch tokens are of shape B x L x D
        touch_tokens = self.token_embedding(touch)
        # concat CLS tokens if any
        if self.num_cls_tokens > 0:
            B = touch_tokens.shape[0]
            class_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole class_tokens impl from Phil Wang, thanks
            touch_tokens = torch.cat((class_tokens, touch_tokens), dim=1)
        touch_tokens = touch_tokens + self.pos_embed
        return_dict = {
            "trunk": {
                "tokens": touch_tokens,
            },
            "head": {},
        }
        # Compute sequence length after adding CLS tokens
        if self.supply_seq_len_to_head:
            touch_lengths = touch.argmax(dim=-1)
            return_dict["head"] = {
                "seq_len": touch_lengths,
            }
        if self.causal_masking:
            return_dict["trunk"].update({"attn_mask": self.mask})
        return return_dict
