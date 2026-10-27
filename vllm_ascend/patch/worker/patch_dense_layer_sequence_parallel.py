"""Sequence-parallel dense decoder layers for DeepseekV2-style models.

When ``use_sequence_parallel_moe`` is enabled, MoE decoder layers run their
post-attention path (fused RMSNorm + MoE) on sequence-sharded activations:
attention stays on full tokens, ``tensor_model_parallel_reduce_scatter``
shards the attention output, and ``sequence_parallel_chunk`` shards the
residual stream. Dense front layers (``layer_idx < first_k_dense_replace``)
are excluded from that path, so their residual stream, both fused RMSNorms
and the dense MLP all operate on the FULL token chunk. During chunked
prefill this stacks several ``(num_tokens, hidden)`` bf16 tensors at once
(residual + norm outputs + MLP output), which profiled as the absolute
activation peak of the whole network (GLM-5.2 DSA, max_num_batched_tokens
= 16000: five (16000, 6144) bf16 buffers alive, ~1277 MB over baseline).

``VLLM_ASCEND_DENSE_LAYERS_SEQUENCE_PARALLEL=1`` makes dense layers join
the sequence-parallel flow the MoE layers already use:

1. the attention ``o_proj`` drops its all_reduce; the layer's
   ``reduce_scatter`` performs the reduction instead;
2. ``use_sequence_parallel_moe`` is flipped on for the layer so the
   forward applies pad + reduce_scatter + residual chunking;
3. the dense MLP keeps its TP-sharded weights; its forward is wrapped
   with ``all_gather`` (SP shard -> full tokens) around the sharded
   GEMMs and ``reduce_scatter`` (full partial output -> SP shard)
   after ``down_proj``.

Replicated-weights SP MLPs (``DeepseekV2MLP(is_sequence_parallel=True)``)
are deliberately avoided: at GLM-5.2's intermediate_size=12288 the weight
replication costs +0.60 GiB per rank, while the activation peak it trims
(MLP-time buffers) is not the network peak in the first place. With
TP-sharded weights the dense-MLP footprint is two full-token buffers
(all_gather output + down_proj partial), which stays below the
attention-time peak.

``VLLM_ASCEND_SEQUENCE_PARALLEL_EMBEDDING=1`` (requires the flag above)
additionally splits the embedding output into sequence-parallel shards so
layer 0 also enters the sequence-parallel flow. Without it, layer 0 keeps
the full embedding alive as its residual (the caller holds a reference
through the whole attention) while also holding the normed full
activations - one extra (num_tokens, hidden) bf16 buffer at the peak
(~187.5 MB at 16000 tokens), which then IS the absolute peak of the
network. The patched decoder-layer forward below gathers the SP
first-layer input for attention.

Mathematical equivalence: ``VocabParallelEmbedding.forward`` all-reduces
internally, so every rank already holds the *complete* embedding. The SP
transition must therefore be a pure split (``sequence_parallel_chunk``:
pad, then slice the rank's own rows - no communication). A
``reduce_scatter`` here would sum tp_size identical copies and scale the
first layer's input by tp_size (the residual stream carries the error;
RMSNorm hides it in hidden_states only), producing garbage output.

The two flags are independent controls:

* ``VLLM_ASCEND_DENSE_LAYERS_SEQUENCE_PARALLEL=0`` restores the upstream
  dense-layer path completely (both patches become no-ops).
* ``VLLM_ASCEND_SEQUENCE_PARALLEL_EMBEDDING=1`` without the flag above is
  a documented no-op: layer 0 must already run the sequence-parallel flow
  for a sharded embedding to be consumed correctly.
"""

import torch
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekAttention,
    DeepseekV2DecoderLayer,
    DeepseekV2MLP,
    DeepseekV2Model,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk

from vllm_ascend import envs
from vllm_ascend.utils import is_mtp_layer

# Marks dense MLPs whose forward must translate between the sequence-sharded
# domain and the full-token domain expected by their TP-sharded weights.
# MoE shared experts and MTP dense MLPs must NOT carry this marker.
_SP_DENSE_MLP = "_vllm_ascend_sp_dense_mlp"

_original_decoder_layer_init = DeepseekV2DecoderLayer.__init__
_original_decoder_layer_forward = DeepseekV2DecoderLayer.forward
_original_embed_input_ids = DeepseekV2Model.embed_input_ids


def _patched_decoder_layer_init(self, vllm_config, prefix, *args, **kwargs):
    _original_decoder_layer_init(self, vllm_config, prefix, *args, **kwargs)

    parallel_config = vllm_config.parallel_config
    config = vllm_config.model_config.hf_config
    if not envs.VLLM_ASCEND_DENSE_LAYERS_SEQUENCE_PARALLEL:
        return
    if not parallel_config.use_sequence_parallel_moe:
        return
    if parallel_config.pipeline_parallel_size != 1:
        return
    if is_mtp_layer(config, prefix):
        return
    if not isinstance(self.mlp, DeepseekV2MLP):
        return

    # The dense MLP keeps its TP-sharded weights; the patched forward below
    # gathers the full tokens before gate_up_proj and scatters the partial
    # output of down_proj back to the local token shard.
    self.mlp.down_proj.reduce_results = False
    setattr(self.mlp, _SP_DENSE_MLP, True)
    o_proj = getattr(self.self_attn, "o_proj", None)
    if o_proj is not None:
        o_proj.reduce_results = False
    self.use_sequence_parallel_moe = True


def _patched_mlp_forward(self, x, already_sequence_parallel: bool = False):
    if not already_sequence_parallel or not getattr(self, _SP_DENSE_MLP, False):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x
    # SP dense MLP: translate to full tokens for the TP-sharded GEMMs and
    # back to the local token shard afterwards. The input size is the padded
    # per-rank token count produced by the attention-side reduce_scatter, so
    # the all_gather output is always divisible by tp_size.
    x = tensor_model_parallel_all_gather(x, 0)
    gate_up, _ = self.gate_up_proj(x)
    x = self.act_fn(gate_up)
    x, _ = self.down_proj(x)
    return tensor_model_parallel_reduce_scatter(x, 0)


def _patched_decoder_layer_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    llama_4_scaling: torch.Tensor | None = None,
) -> torch.Tensor:
    if not envs.VLLM_ASCEND_DENSE_LAYERS_SEQUENCE_PARALLEL:
        return _original_decoder_layer_forward(self, positions, hidden_states, residual, llama_4_scaling)

    full_num_tokens = positions.shape[0]
    # VLLM_ASCEND_SEQUENCE_PARALLEL_EMBEDDING relaxes upstream's
    # residual-is-not-None requirement so the first layer can gather an SP
    # input; with the flag off the condition reduces to the upstream one.
    input_is_sequence_parallel = (
        self.use_sequence_parallel_moe
        and hidden_states.shape[0] != full_num_tokens
        and (residual is not None or envs.VLLM_ASCEND_SEQUENCE_PARALLEL_EMBEDDING)
    )

    # Self Attention
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    if input_is_sequence_parallel:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        hidden_states = hidden_states[:full_num_tokens]

    if self.use_mha:
        hidden_states = self.self_attn(positions, hidden_states)
    else:
        hidden_states = self.self_attn(positions, hidden_states, llama_4_scaling)

    if not isinstance(self.self_attn, DeepseekAttention) and hidden_states.dtype == torch.float16:
        # Fix FP16 overflow
        # We scale both hidden_states and residual before
        # rmsnorm, and rmsnorm result would not affect by scale.
        hidden_states *= 1.0 / self.routed_scaling_factor
        if self.layer_idx == 0:
            # The residual is shared by all layers, we only scale it on
            # first layer.
            residual *= 1.0 / self.routed_scaling_factor

    if self.use_sequence_parallel_moe:
        tp_world_size = get_tensor_model_parallel_world_size()
        # small trick using minus, eg. -17 % 8 = 7
        sp_pad = (-hidden_states.shape[0]) % tp_world_size
        # pad if not divisible by world size
        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, sp_pad))
        hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
        if not input_is_sequence_parallel:
            residual = sequence_parallel_chunk(residual)

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    if self.use_sequence_parallel_moe:
        hidden_states = self.mlp(
            hidden_states,
            already_sequence_parallel=True,
        )
    else:
        hidden_states = self.mlp(hidden_states)

    if isinstance(self.mlp, DeepseekV2MLP) and hidden_states.dtype == torch.float16:
        # Fix FP16 overflow
        # Scaling the DeepseekV2MLP output, it is the input of
        # input_layernorm of next decoder layer.
        # The scaling of DeepseekV2MOE output would be done in the forward
        # of DeepseekV2MOE
        hidden_states *= 1.0 / self.routed_scaling_factor

    return hidden_states, residual


def _patched_embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
    hidden_states = _original_embed_input_ids(self, input_ids)
    if not (envs.VLLM_ASCEND_DENSE_LAYERS_SEQUENCE_PARALLEL and envs.VLLM_ASCEND_SEQUENCE_PARALLEL_EMBEDDING):
        return hidden_states
    # Only hand SP activations to the stack when the first layer actually
    # runs the sequence-parallel flow (set by the decoder-layer patch).
    layers = getattr(self, "layers", None)
    first_layer = layers[0] if layers else None
    if not isinstance(first_layer, DeepseekV2DecoderLayer):
        return hidden_states
    if not getattr(first_layer, "use_sequence_parallel_moe", False):
        return hidden_states
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return hidden_states
    num_tokens = hidden_states.shape[0]
    if num_tokens < tp_size:
        # For tiny batches the decoder layer's shape-based
        # input_is_sequence_parallel detection cannot distinguish a shard
        # from a full tensor (ceil(N/tp) == N when N == 1), so keep the
        # upstream full-token path. The memory win is irrelevant there.
        return hidden_states
    # hidden_states is the COMPLETE embedding on every rank
    # (VocabParallelEmbedding all-reduces internally), so the SP
    # transition is a pure local split - never a reduce_scatter, which
    # would scale the input by tp_size.
    return sequence_parallel_chunk(hidden_states)


DeepseekV2DecoderLayer.__init__ = _patched_decoder_layer_init
DeepseekV2DecoderLayer.forward = _patched_decoder_layer_forward
DeepseekV2MLP.forward = _patched_mlp_forward
DeepseekV2Model.embed_input_ids = _patched_embed_input_ids
