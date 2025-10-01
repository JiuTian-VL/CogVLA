import re
import torch
import time
import transformers
from dataclasses import dataclass
from transformers.models.llama.modeling_llama import *
from transformers import LlamaConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


def llama_sdpa_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    num_vision: int = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # ======================== NEW ========================
    if attention_mask is None:
        # We need to modify attention mask later, so it can not be None
        attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        attention_mask = _prepare_4d_causal_attention_mask(attention_mask, hidden_states.shape[:2], hidden_states, past_key_values_length=0)
    # =====================================================

    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # In case static cache is used, it is an instance attribute.
    past_key_value = getattr(self, "past_key_value", past_key_value)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # In case we are not compiling, we may set `causal_mask` to None, which is required to dispatch to SDPA's Flash Attention 2 backend, rather
    # relying on the `is_causal` argument.

    # ================= NEW =================
    # `CATTEN` IMPLEMENTATION
    if causal_mask is not None:
        min_dtype = torch.finfo(hidden_states.dtype).min
        num_pad = (causal_mask[:, 0, -1, :] == min_dtype).sum(dim=1)
        num_act = NUM_ACTIONS_CHUNK * ACTION_DIM

        new_mask = causal_mask.clone()
        for idx, n_pad in enumerate(num_pad):
            num_act += 1  # stop token
            if n_pad == 0:
                new_mask[idx, :, -num_act:, -num_act:] = 0
            else:
                new_mask[idx, :, -(num_act + n_pad):-n_pad, -(num_act + n_pad):-n_pad] = 0

            # start = -(num_act + n_pad)
            # end = -n_pad if n_pad else None
            # causal_mask[idx, :, start:end, start:end] = 0

        causal_mask = new_mask
    # =======================================

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        # is_causal=causal_mask is None and q_len > 1,
        is_causal=False,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def check_inf_or_nan(x, x_name):
    if torch.isnan(x).any():
        raise ValueError(
            f"NaN detected in input tokens, this is not intended to happen, please check your model. Before retraining, you could try the model with flash-attn-2 enabled.\n{x_name}:{x}")
    elif torch.isinf(x).any():
        raise ValueError(
            f"Inf detected in input tokens, this is not intended to happen, please check your model. Before retraining, you could try the model with flash-attn-2 enabled.\n{x_name}:{x}")


def shifted_cos_with_ratio(average_ratio, layer_idx, total_layers=32):
    return math.cos(layer_idx * math.pi / (total_layers - 1)) / 2 + average_ratio


def shifted_linear_with_ratio(average_ratio, layer_idx, total_layers=32):
    return (-layer_idx / (total_layers - 1) + 0.5) + average_ratio


decay_func_dict = {
    "shiftedcos": shifted_cos_with_ratio,
    "shiftedlinear": shifted_linear_with_ratio,
}


class TokenRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = nn.Linear(config.hidden_size, 2)

        # nn.init.zeros_(self.router.weight)
        # nn.init.zeros_(self.router.bias)

    def forward(self, x):
        # [bs, seq_len, dim] -> [bs, seq_len, 2]
        return self.router(x)


class FiLMedTokenRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = nn.Linear(config.hidden_size, 2)

        self.scale = get_mlp(config.hidden_size, config.hidden_size // 2, config.hidden_size)
        self.shift = get_mlp(config.hidden_size, config.hidden_size // 2, config.hidden_size)

        # self.scale = nn.Linear(config.hidden_size, config.hidden_size)
        # self.shift = nn.Linear(config.hidden_size, config.hidden_size)
        # nn.init.zeros_(self.router.weight)
        # nn.init.zeros_(self.router.bias)

    def forward(self, x, attention_mask, num_vision):
        if attention_mask is not None:
            min_dtype = torch.finfo(x.dtype).min
            num_pad = (attention_mask[:, 0, -1, :] == min_dtype).sum(dim=1)
        else:
            num_pad = [0] * x.shape[0]

        # Extract text hidden states
        num_act = NUM_ACTIONS_CHUNK * ACTION_DIM + 1  # 1 for stop token
        text_hidden = []
        for cur_x, n_pad in zip(x, num_pad):
            cur_text = cur_x[1 + num_vision: -(num_act + n_pad), :]
            cur_text = torch.mean(cur_text, dim=0)
            text_hidden.append(cur_text)
        text_hidden = torch.stack(text_hidden)  # [bs, dim]

        # FiLM projector
        gamma = self.scale(text_hidden)  # [bs, dim]
        beta = self.shift(text_hidden)  # [bs, dim]

        # Apply FilM to vision (non-replace)
        filmed_vision = x[:, 1: 1 + num_vision, :] * (1 + gamma.view(gamma.shape[0], 1, gamma.shape[1])) + beta.view(beta.shape[0], 1, beta.shape[1])
        _x = torch.cat([x[:, 0:1, :], filmed_vision, x[:, 1 + num_vision:, :]], dim=1)

        # [bs, seq_len, dim] -> [bs, seq_len, 2]
        return self.router(_x)


def get_mlp(in_dim, hidden_dim, out_dim, zero_init=False):
    mlp = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim)
    )
    if zero_init:
        for layer in mlp:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
    return mlp


class LlamaDecoderLFPLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int, ratio: float = 0.5):
        super().__init__(config, layer_idx)

        self.config = config
        self.layer_idx = layer_idx
        self.router_factor = ratio

        if config.lfp_enable_film:
            self.router = FiLMedTokenRouter(config)
        else:
            self.router = TokenRouter(config)

        self.get_num_patches = None
        self.get_num_images_in_input = None

    def forward_w_router_weights(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        router_weights: torch.Tensor = None,  # NEW, [bs, seq_len]
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            num_vision=int(self.get_num_patches() * self.get_num_images_in_input() * self.router_factor),  # NEW
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states) * router_weights.unsqueeze(-1).to(hidden_states.dtype)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        # only apply LFP during prefill
        if hidden_states.shape[1] == 1:
            return super().forward(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                **kwargs,
            )

        """ We assume that each sample in the batch has the same number of visual tokens.
            The hidden_state consists of [st * 1, visual * m, text * n], where m is constant.
        """

        # get constants
        bs, seq_len, dim = hidden_states.shape
        num_visual_tokens = self.get_num_patches() * self.get_num_images_in_input()
        visual_kept_length = int(num_visual_tokens * self.router_factor)
        kept_length = seq_len - (num_visual_tokens - visual_kept_length)

        # router
        if self.config.lfp_enable_film:
            router_weights = self.router(self.input_layernorm(hidden_states), attention_mask, num_visual_tokens)  # [bs, seq_length]
        else:
            router_weights = self.router(self.input_layernorm(hidden_states))  # [bs, seq_length]

        router_logits = F.softmax(router_weights, dim=-1)[:, :, 1]

        # ===================== select kept tokens =====================
        force_select_mask = torch.zeros(router_logits.shape, device=router_logits.device)
        force_select_mask[:, 0] = force_select_mask[:, 1 + num_visual_tokens:] = torch.inf

        _, router_indices = torch.topk(router_logits + force_select_mask, kept_length, dim=1, sorted=True)  # [bs, kept_len]
        router_indices, _ = torch.sort(router_indices, dim=1)

        kept_router_weights = torch.gather(router_logits, dim=1, index=router_indices)
        kept_tokens = torch.gather(hidden_states, dim=1, index=router_indices.unsqueeze(-1).expand(-1, -1, dim))
        kept_position_ids = torch.arange(0, kept_length).unsqueeze(0).to(kept_tokens.device)

        assert self.config._attn_implementation != "flash_attention_2"

        if attention_mask is not None:
            kept_attention_mask_rows = torch.gather(
                attention_mask,
                dim=2,
                index=router_indices.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, seq_len)
            )
            kept_attention_mask = torch.gather(
                kept_attention_mask_rows,
                dim=3,
                index=router_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, kept_length, -1)
            )
        else:
            kept_attention_mask = torch.ones(kept_tokens.shape[:2], dtype=torch.bool, device=kept_tokens.device)
            # prepare_func = _prepare_4d_causal_attention_mask if self.config._attn_implementation == 'eager' else _prepare_4d_causal_attention_mask_for_sdpa
            # kept_attention_mask = prepare_func(kept_attention_mask, (bs, kept_length), kept_tokens, past_key_values_length=0)
            kept_attention_mask = _prepare_4d_causal_attention_mask(kept_attention_mask, (bs, kept_length), kept_tokens, past_key_values_length=0)

        # kept_position_ids = torch.gather(position_ids.repeat(bs, 1), dim=1, index=router_indices)
        # ==============================================================

        # call the forward of transformer decoder layer
        outputs = self.forward_w_router_weights(
            kept_tokens,
            attention_mask=kept_attention_mask,
            position_ids=kept_position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            router_weights=kept_router_weights,
            **kwargs
        )

        hidden_states = hidden_states.scatter(dim=1, index=router_indices.unsqueeze(-1).expand(-1, -1, dim), src=outputs[0])

        check_inf_or_nan(hidden_states, "hidden_states")

        outputs = (hidden_states,) + outputs[1:]

        # # auxiliary loss: force the probability to approach one
        # router_targets = torch.zeros_like(router_logits)
        # for i in range(bs):
        #     router_targets[i, router_indices[i]] = 1
        # aux_loss = F.cross_entropy(router_weights.view(-1, 2), router_targets.view(-1).long())
        # outputs += (aux_loss,)

        return outputs


def llama_lfp_model__init__(self, config: LlamaConfig):
    super(transformers.models.llama.modeling_llama.LlamaModel, self).__init__(config)
    self.config = config
    self.padding_idx = config.pad_token_id
    self.vocab_size = config.vocab_size

    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

    # ============= NEW =============
    lfp_type = config.lfp_type
    ratios = [config.lfp_average_factor for _ in range(config.num_hidden_layers)]

    DECAY_PATTERN = r"(\w+)_decay_(\d+\.\d+)_(\d+\.\d+)"  # "{shiftedcos}_decay_{0.75}_{0.25}"
    if match := re.match(DECAY_PATTERN, lfp_type):
        decay_type = match.group(1)
        max_ratio = float(match.group(2))
        min_ratio = float(match.group(3))

        decay_func = decay_func_dict[decay_type]
        ratios = [decay_func(config.lfp_average_factor, i) for i in range(config.num_hidden_layers)]
        ratios = [max(r, min_ratio) for r in ratios]

        lfp_target_layers = [
            i for i in range(config.num_hidden_layers)
            if ratios[i] <= max_ratio
        ]
    elif lfp_type == "deep_all":
        # include the last layer
        lfp_target_layers = list(range(2, config.num_hidden_layers))
    elif lfp_type == "deep_all_wo_last":
        # exclude the last layer
        lfp_target_layers = list(range(2, config.num_hidden_layers - 1))
    elif lfp_type == "interleave":
        lfp_target_layers = list(range(1, config.num_hidden_layers, 2))
    else:
        raise NotImplementedError(f"Unsupported lfp_type: {lfp_type}")

    config.lfp_target_layers = lfp_target_layers
    print(f"-------------------------------")
    print(f"lfp_target_layers:{[f'layer: {l}, ratio: {ratios[l]}' for l in lfp_target_layers]} ")
    print(f"-------------------------------")

    self.layers = nn.ModuleList([
        LlamaDecoderLFPLayer(config, layer_idx, ratio=ratios[layer_idx])
        if layer_idx in lfp_target_layers
        else LlamaDecoderLayer(config, layer_idx)
        for layer_idx in range(config.num_hidden_layers)
    ])
    # ===============================

    self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.gradient_checkpointing = False

    # Initialize weights and apply final processing
    self.post_init()


def replace_llama_forward():
    transformers.models.llama.modeling_llama.LlamaModel.__init__ = llama_lfp_model__init__


def replace_llama_spda_forward():
    transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attention_forward