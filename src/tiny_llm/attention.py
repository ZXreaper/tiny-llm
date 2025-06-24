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
    mask = mx.tril(mx.ones((L, S)), k=(S - L))
    mask = mx.where(mask, mx.array(0), mx.array(-mx.inf)).astype(dtype)
    return mask


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    factor = mx.rsqrt(query.shape[-1]) if scale is None else mx.array(scale)
    factor = factor.astype(query.dtype)
    expected_shape = query.shape

    head_num_q, l_q, d_q = query.shape[-3:]
    head_num_k, l_k, d_k = key.shape[-3:]
    assert head_num_q % head_num_k == 0
    n_repeats = head_num_q // head_num_k

    query = query.reshape(-1, head_num_k, n_repeats, l_q, d_q)
    key = key.reshape(-1, head_num_k, 1, l_k, d_k)
    value = value.reshape(-1, head_num_k, 1, l_k, d_k)

    scores = mx.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        if mask == "causal":
            mask = causal_mask(l_q, l_k, scores.dtype)
            scores = scores + mask
        else:
            mask = mask.reshape(-1, head_num_k, n_repeats, mask.shape[-2], mask.shape[-1])
            scores = scores + mask
    # (-1, head_num_k, n_repeats, l_q, l_k) * (-1, head_num_k, 1, l_k, d_k)
    # (-1, head_num_k, n_repeats, l_q, d_k)  dk == dq
    result = mx.matmul(softmax(scores, axis=-1), value)
    return result.reshape(expected_shape)


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
