import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    d_k = query.shape[-1]
    if scale is None:
        scale = d_k ** -0.5
    attention_weights = mx.matmul(query, key.swapaxes(-2, -1)) * scale
    if mask is not None:
        attention_weights = attention_weights + mask
    attention_weights = softmax(attention_weights, axis=-1)
    return mx.matmul(attention_weights, value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        self.scale = mx.rsqrt(self.head_dim)
        assert wq.shape == (hidden_size, num_heads * self.head_dim)
        assert wk.shape == (hidden_size, num_heads * self.head_dim)
        assert wv.shape == (hidden_size, num_heads * self.head_dim)
        assert wo.shape == (num_heads * self.head_dim, hidden_size)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        N, L, _ = query.shape
        assert query.shape == key.shape == value.shape
        projection_q = linear(query, self.wq).reshape(N, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        projection_k = linear(key, self.wk).reshape(N, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        projection_v = linear(value, self.wv).reshape(N, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        x = scaled_dot_product_attention_simple(projection_q, projection_k, projection_v, scale=self.scale, mask=mask)
        x = x.transpose(0, 2, 1, 3).reshape(N, L, self.hidden_size)
        return linear(x, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
