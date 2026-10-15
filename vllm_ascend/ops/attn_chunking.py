# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention token chunking helpers.

Mirrors the pattern of ``vllm_ascend.ops.fused_moe.moe_ffn_chunking``:
sequential per-chunk execution with explicit intermediate freeing so
that the NPU memory allocator reuses the buffer across iterations,
keeping the peak proportional to *one chunk* instead of the full batch.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch


def fixed_chunk_ranges(num_tokens: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    """Yield ordered contiguous ranges with at most *chunk_size* tokens each."""
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    for start in range(0, num_tokens, chunk_size):
        yield start, min(start + chunk_size, num_tokens)


def build_chunk_cu_seqlens(
    cu_seqlens: torch.Tensor, start_token: int, end_token: int
) -> torch.Tensor:
    """Rebuild cumulative sequence lengths for tokens [*start_token*, *end_token*).

    *cu_seqlens* is the full-batch cumulative query lengths, e.g.
    ``[0, len_0, len_0+len_1, ...]``.  A chunk may partially overlap a
    request; the overlap length is counted so that the attention kernel
    sees a valid contiguous sub-sequence.
    """
    token_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    chunk_lens: list[int] = []
    for i in range(token_lens.shape[0]):
        req_start = int(cu_seqlens[i])
        req_end = int(cu_seqlens[i + 1])
        overlap_start = max(req_start, start_token)
        overlap_end = min(req_end, end_token)
        overlap = max(overlap_end - overlap_start, 0)
        if overlap > 0:
            chunk_lens.append(overlap)
    if not chunk_lens:
        return torch.zeros(1, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    return torch.cumsum(
        torch.tensor(chunk_lens, dtype=cu_seqlens.dtype, device=cu_seqlens.device),
        dim=0,
    ).reshape(-1)


def get_chunk_request_indices(
    cu_seqlens: torch.Tensor, start_token: int, end_token: int
) -> list[int]:
    """Return indices of requests that have at least one token in the chunk."""
    indices: list[int] = []
    for i in range(cu_seqlens.shape[0] - 1):
        req_start = int(cu_seqlens[i])
        req_end = int(cu_seqlens[i + 1])
        if req_end > start_token and req_start < end_token:
            indices.append(i)
    return indices


def slice_per_request_tensor(
    tensor: torch.Tensor | None, indices: list[int]
) -> torch.Tensor | None:
    """Slice a per-request tensor (e.g. seqused_kv) by request indices."""
    if tensor is None:
        return None
    if len(indices) == 0:
        return tensor[:0]
    return tensor[indices]


def build_chunk_sas_metadata(
    *,
    metadata_op,
    chunk_cu_seqlens_q: torch.Tensor,
    chunk_seqused_kv: torch.Tensor,
    chunk_block_table: torch.Tensor,
    num_heads_q: int,
    head_dim: int,
    sliding_window: int,
    compress_ratio: int,
    index_topk: int | None,
    cu_seqlens_ori_kv: torch.Tensor | None,
    cu_seqlens_cmp_kv: torch.Tensor | None,
    device: str,
) -> torch.Tensor:
    """Rebuild the opaque sas_metadata tensor for a token chunk.

    Calls the same C++ metadata operator used by the builder so that
    the attention kernel receives consistent per-chunk metadata.
    """
    num_reqs = chunk_cu_seqlens_q.shape[0] - 1
    if num_reqs == 0:
        num_reqs = 1
    max_seqlen_q = int(chunk_cu_seqlens_q.max().item()) if num_reqs > 0 else 1
    max_seqlen_kv = int(chunk_seqused_kv.max().item()) if chunk_seqused_kv.numel() > 0 else 1

    kw = dict(
        num_heads_q=num_heads_q,
        num_heads_kv=1,
        head_dim=head_dim,
        cu_seqlens_q=chunk_cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv if cu_seqlens_ori_kv is not None else chunk_cu_seqlens_q,
        seqused_q=chunk_cu_seqlens_q,
        seqused_kv=chunk_seqused_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        batch_size=num_reqs,
        ori_mask_mode=4,
        ori_win_left=sliding_window - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=True,
        device=device,
    )

    cmp_ratio = max(compress_ratio, 1)
    if compress_ratio > 1:
        kw["has_cmp_kv"] = True
        kw["cmp_mask_mode"] = 3
        kw["cmp_ratio"] = cmp_ratio
        if cu_seqlens_cmp_kv is not None:
            kw["cu_seqlens_cmp_kv"] = cu_seqlens_cmp_kv
        if compress_ratio == 4 and index_topk is not None:
            kw["cmp_topk"] = index_topk
    else:
        kw["cmp_ratio"] = cmp_ratio
        kw["has_cmp_kv"] = False

    return metadata_op(**kw)
