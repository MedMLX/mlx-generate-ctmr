"""Request-scoped budgeting of dense attention intermediates, without importing MLX."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

_budget: ContextVar[int | None] = ContextVar("mlx_generate_ctmr_attention_bytes", default=None)


@contextmanager
def attention_memory_budget(max_bytes: int) -> Generator[None]:
    """Limit estimated dense attention scratch per evaluated query chunk.

    The estimate includes three score-sized arrays (QK product, scaled scores,
    probabilities), using at least FP32 storage. It excludes weights, Q/K/V,
    outputs, backend workspace and allocator cache, so it is not a process memory
    limit. Nested calls restore the previous budget, including on exceptions.
    Each query still attends to all keys; query partitioning may change the runtime
    kernel shape, so bitwise equivalence is not guaranteed by this option.
    Use around a complete CT/MR generation call. Without this explicit option the
    qualified, token-threshold chunking policy is retained.
    """
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("attention memory budget must be a positive integer byte count")
    token = _budget.set(max_bytes)
    try:
        yield
    finally:
        _budget.reset(token)


def query_chunk_size(batch: int, heads: int, queries: int, keys: int, itemsize: int) -> int | None:
    """Choose a query count that fits the active scratch estimate, or None if unset.

    Whole query rows retain every key and the original softmax reduction. Reject
    a budget smaller than one row rather than pretending it bounds this workload.
    Dimensions are validated only when a budget is active; sufficiently large
    partial chunks are rounded down to a multiple of 256 queries.
    """
    budget = _budget.get()
    if budget is None:
        return None
    if min(batch, heads, queries, keys, itemsize) < 1:
        raise ValueError("attention dimensions and itemsize must be positive")
    row_bytes = 3 * batch * heads * keys * max(4, itemsize)
    count = min(queries, budget // row_bytes)
    if count < 1:
        raise ValueError(f"attention budget needs at least {row_bytes} bytes for one query row")
    # Multiples of 256 avoid many tiny dispatches; tails still cover every query.
    return count if count == queries or count < 256 else count // 256 * 256
