#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import importlib
import sys
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from ...lmodels.oophi15.configuration_mixformer_sequential import \
    MixFormerSequentialConfig as PhiConfig
from ...lmodels.oophi15.modeling_mixformer_sequential import \
    MixFormerSequentialForCausalLM
# sys.path.append("../../../")
from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


class CausalLMLoss(nn.Module):
    """Causal Language Modeling loss.

    Reference:
        Improving Language Understanding by Generative Pre-Training.
        https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf.

    """

    def __init__(self, shift_labels: bool = True) -> None:
        super().__init__()

        self.shift_labels = shift_labels
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(
        self, logits: torch.FloatTensor, labels: torch.LongTensor
    ) -> torch.FloatTensor:
        if self.shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return loss


class MyPhi(MixFormerSequentialForCausalLM):
    def get_input_embeddings(self):
        return self.embd.wte

    def embed_tokens(self, inp):
        return self.emb(inp)


class LlavaConfig(PhiConfig):
    model_type = "llava"


class LlavaPhiModel(LlavaMetaModel, MixFormerSequentialForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config: PhiConfig):
        super(LlavaPhiModel, self).__init__(config)

    def embed_tokens(self, inp):
        return self.layers[0](inp).squeeze(0)


class LlavaLlamaForCausalLM(MixFormerSequentialForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(MixFormerSequentialForCausalLM, self).__init__(config)
        # print('Phi INIT')
        self.model = LlavaPhiModel(config)
        test_model = MixFormerSequentialForCausalLM.from_pretrained(
            config._name_or_path
        )
        del self.model.layers
        torch.cuda.empty_cache()
        self.model.layers = test_model.layers
        self.pretraining_tp = 0  # config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing = False

        self.loss = CausalLMLoss()
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def get_input_embeddings(self):
        # print(self)
        return self.model.layers[0].wte

    def get_output_embeddings(self):
        return self.model.layers[25].linear

    def set_input_embeddings(self, new_embeddings):
        self.model.layers[0].wte = new_embeddings

    def set_output_embeddings(self, new_embeddings):
        self.model.layers[25].linear = new_embeddings
        # print('in set_input_embeddings:', new_embeddings)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        alpha=1,
        return_norms=False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # torch.cuda.empty_cache()
        # past_key_values = None

        # n = 0
        # for p in self.parameters():
        #     n += p.numel()
        # print('number of params, mln', int(n / 1e6))
        if inputs_embeds is None and images is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                alpha,
            )

        if inputs_embeds is None:
            # print('inputs_embeds is None')
            inputs_embeds = self.model.layers[0](input_ids)

        norms = []
        id1, id2 = 142, 142 + 14 * 14
        norm = torch.linalg.norm(inputs_embeds[0], 2, -1)
        norms.append(
            [
                (norm[:id1].sum() + norm[id2:].sum()).item() / (len(norm) + id1 - id2),
                norm[id1:id2].mean().item(),
            ]
        )
        # norms.append(norm.mean().item())
        # print('inputs_embeds', inputs_embeds.shape)
        # print('inputs_embeds', inputs_embeds[:, -3:, :4])
        # print('past_kv', inputs_embeds.shape, past_key_values)
        # raise Exception()
        if False and not past_key_values:
            lm_logits = self.model.layers[1:](inputs_embeds)
        else:
            hidden_layer = inputs_embeds
            # print('hidden_layer ', hidden_layer.shape)
            for module in self.model.layers[1:-1]:
                # print('past_key_values', past_key_values)
                if self.gradient_checkpointing and self.training:
                    hidden_layer = self._gradient_checkpointing_func(
                        decoder_layer.__call__, hidden_layer, past_key_values
                    )
                else:
                    # print('past_kv', inputs_embeds.shape, [t.shape for t in past_key_values.key_value_memory_dict.values()])
                    # raise Exception()
                    hidden_layer = module(hidden_layer, past_cache=past_key_values)
                norm = torch.linalg.norm(hidden_layer[0], 2, -1)
                norms.append(
                    [
                        (norm[:id1].sum() + norm[id2:].sum()).item()
                        / (len(norm) + id1 - id2),
                        norm[id1:id2].mean().item(),
                    ]
                )
                # norms.append(norm.mean().item())
            lm_logits = self.model.layers[-1](hidden_layer)

        for n, c in self.model.named_children():
            if "loss" not in n.lower():
                # print(n)
                trus = 0
                falses = 0
                for p in c.parameters():
                    if p.requires_grad:
                        trus += 1
                    else:
                        falses += 1
                # print(trus, falses)
        loss = None
        if labels is not None:
            # print('labels', labels)
            loss = self.loss(lm_logits, labels)

        if return_norms:
            return (
                CausalLMOutputWithPast(
                    loss=loss, logits=lm_logits, past_key_values=past_key_values
                ),
                norms,
            )

        return CausalLMOutputWithPast(
            loss=loss, logits=lm_logits, past_key_values=past_key_values
        )

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        # print(_inputs)
        if images is not None:
            _inputs["images"] = images
        return _inputs


AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
