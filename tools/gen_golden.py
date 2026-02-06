"""Generate golden data for transformer forward pass verification.

Replicates the MoonBit transformer implementation exactly:
- Pre-norm GPT (LayerNorm -> Attention/FFN -> Residual Add)
- batched_linear: x @ W (W is [in_dim, out_dim])
- Q/K/V/O projections: no bias
- FFN: W1[d_model, d_ff] + b1[d_ff] -> GELU(tanh approx) -> W2[d_ff, d_model] + b2[d_model]
- LayerNorm: eps=1e-5, population variance (divide by N)
- Causal mask: -1e9 for masked positions
"""

import struct
import os
import numpy as np

# Model config
VOCAB_SIZE = 32
D_MODEL = 16
NUM_HEADS = 2
NUM_LAYERS = 2
D_FF = 64
MAX_SEQ_LEN = 32
BATCH_SIZE = 2
SEQ_LEN = 8
EPS = 1e-5
D_K = D_MODEL // NUM_HEADS  # 8

# Fixed tokens
TOKENS = np.array(
    [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]], dtype=np.int32
)


def init_weights(shape, rng):
    """Initialize weights with randn * 0.02, matching MoonBit's init."""
    return (rng.randn(*shape) * 0.02).astype(np.float32)


def layer_norm(x, gamma, beta, eps=EPS):
    """LayerNorm with population variance (divide by N, not N-1)."""
    mean = x.mean(axis=-1, keepdims=True)
    # Population variance (N, not N-1)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    normalized = (x - mean) / std
    return gamma * normalized + beta


def gelu_tanh(x):
    """GELU with tanh approximation matching MoonBit implementation."""
    sqrt_2_pi = np.sqrt(2.0 / np.pi)
    coef = 0.044715
    inner = sqrt_2_pi * (x + coef * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def linear_no_bias(x, w):
    """x @ W where W is [in_dim, out_dim]."""
    return x @ w


def linear_with_bias(x, w, b):
    """x @ W + b."""
    return x @ w + b


def reshape_for_heads(x, batch, seq, num_heads, d_k):
    """[batch, seq, d_model] -> [batch, num_heads, seq, d_k]

    Matches MoonBit reshape_for_heads which treats d_model as [num_heads, d_k]
    interleaved, then transposes axes 1 and 2.
    """
    # [batch, seq, num_heads, d_k]
    x = x.reshape(batch, seq, num_heads, d_k)
    # [batch, num_heads, seq, d_k]
    x = x.transpose(0, 2, 1, 3)
    return x


def reshape_from_heads(x, batch, seq, num_heads, d_k):
    """[batch, num_heads, seq, d_k] -> [batch, seq, d_model]"""
    # [batch, seq, num_heads, d_k]
    x = x.transpose(0, 2, 1, 3)
    # [batch, seq, d_model]
    x = x.reshape(batch, seq, num_heads * d_k)
    return x


def multi_head_attention(x, w_q, w_k, w_v, w_o, num_heads, mask):
    """Multi-head attention matching MoonBit implementation."""
    batch = x.shape[0]
    seq = x.shape[1]
    d_model = x.shape[2]
    d_k = d_model // num_heads

    # Project Q, K, V
    q = linear_no_bias(x, w_q)  # [batch, seq, d_model]
    k = linear_no_bias(x, w_k)
    v = linear_no_bias(x, w_v)

    # Reshape for heads
    q = reshape_for_heads(q, batch, seq, num_heads, d_k)
    k = reshape_for_heads(k, batch, seq, num_heads, d_k)
    v = reshape_for_heads(v, batch, seq, num_heads, d_k)

    # Attention scores: Q @ K^T / sqrt(d_k)
    scale = 1.0 / np.sqrt(d_k)
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale  # [batch, heads, seq, seq]

    # Apply causal mask
    if mask is not None:
        scores = scores + mask  # broadcast [seq, seq] to [batch, heads, seq, seq]

    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # Weighted sum
    attn_out = np.matmul(attn_weights, v)  # [batch, heads, seq, d_k]

    # Reshape back
    concat = reshape_from_heads(attn_out, batch, seq, num_heads, d_k)

    # Output projection
    return linear_no_bias(concat, w_o)


def feed_forward(x, w1, b1, w2, b2):
    """FFN: Linear -> GELU -> Linear, matching MoonBit."""
    h = linear_with_bias(x, w1, b1)
    h = gelu_tanh(h)
    return linear_with_bias(h, w2, b2)


def transformer_block(x, params, num_heads, mask):
    """Single transformer block (pre-norm GPT-style)."""
    ln1_gamma, ln1_beta, w_q, w_k, w_v, w_o, ln2_gamma, ln2_beta, ff_w1, ff_b1, ff_w2, ff_b2 = params

    # Pre-norm attention
    ln1 = layer_norm(x, ln1_gamma, ln1_beta)
    attn = multi_head_attention(ln1, w_q, w_k, w_v, w_o, num_heads, mask)
    x2 = x + attn

    # Pre-norm FFN
    ln2 = layer_norm(x2, ln2_gamma, ln2_beta)
    ff = feed_forward(ln2, ff_w1, ff_b1, ff_w2, ff_b2)
    return x2 + ff


def transformer_forward(tokens, all_params, config, mask):
    """Full transformer forward pass."""
    token_emb_w, pos_emb_w, block_params_list, ln_final_gamma, ln_final_beta, lm_head = all_params

    batch = tokens.shape[0]
    seq = tokens.shape[1]

    # Token embedding
    tok_emb = token_emb_w[tokens]  # [batch, seq, d_model]

    # Positional embedding
    pos_emb = pos_emb_w[:seq]  # [seq, d_model]
    x = tok_emb + pos_emb  # broadcast

    # Transformer blocks
    for block_params in block_params_list:
        x = transformer_block(x, block_params, config["num_heads"], mask)

    # Final LayerNorm
    x = layer_norm(x, ln_final_gamma, ln_final_beta)

    # LM head
    logits = linear_no_bias(x, lm_head)  # [batch, seq, vocab_size]
    return logits


def main():
    rng = np.random.RandomState(42)

    # Initialize all weights
    token_embedding = init_weights((VOCAB_SIZE, D_MODEL), rng)
    pos_embedding = init_weights((MAX_SEQ_LEN, D_MODEL), rng)

    block_params_list = []
    for _ in range(NUM_LAYERS):
        ln1_gamma = np.ones(D_MODEL, dtype=np.float32)
        ln1_beta = np.zeros(D_MODEL, dtype=np.float32)
        w_q = init_weights((D_MODEL, D_MODEL), rng)
        w_k = init_weights((D_MODEL, D_MODEL), rng)
        w_v = init_weights((D_MODEL, D_MODEL), rng)
        w_o = init_weights((D_MODEL, D_MODEL), rng)
        ln2_gamma = np.ones(D_MODEL, dtype=np.float32)
        ln2_beta = np.zeros(D_MODEL, dtype=np.float32)
        ff_w1 = init_weights((D_MODEL, D_FF), rng)
        ff_b1 = np.zeros(D_FF, dtype=np.float32)
        ff_w2 = init_weights((D_FF, D_MODEL), rng)
        ff_b2 = np.zeros(D_MODEL, dtype=np.float32)
        block_params_list.append(
            (ln1_gamma, ln1_beta, w_q, w_k, w_v, w_o, ln2_gamma, ln2_beta, ff_w1, ff_b1, ff_w2, ff_b2)
        )

    ln_final_gamma = np.ones(D_MODEL, dtype=np.float32)
    ln_final_beta = np.zeros(D_MODEL, dtype=np.float32)
    lm_head = init_weights((D_MODEL, VOCAB_SIZE), rng)

    # Causal mask: -1e9 for future positions
    mask = np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float32)
    for i in range(SEQ_LEN):
        for j in range(SEQ_LEN):
            if j > i:
                mask[i, j] = -1e9

    # Forward pass
    config = {"num_heads": NUM_HEADS}
    all_params = (
        token_embedding,
        pos_embedding,
        block_params_list,
        ln_final_gamma,
        ln_final_beta,
        lm_head,
    )
    logits = transformer_forward(TOKENS, all_params, config, mask)

    print(f"Logits shape: {logits.shape}")
    print(f"Logits range: [{logits.min():.6f}, {logits.max():.6f}]")
    print(f"Logits[0,0,:5]: {logits[0, 0, :5]}")

    # Save weights.bin
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "testdata", "golden")
    os.makedirs(out_dir, exist_ok=True)

    weights_path = os.path.join(out_dir, "weights.bin")
    with open(weights_path, "wb") as f:
        # token_embedding [vocab_size * d_model]
        f.write(token_embedding.tobytes())
        # pos_embedding [max_seq_len * d_model]
        f.write(pos_embedding.tobytes())
        # Per layer
        for block in block_params_list:
            ln1_gamma, ln1_beta, w_q, w_k, w_v, w_o, ln2_gamma, ln2_beta, ff_w1, ff_b1, ff_w2, ff_b2 = block
            f.write(ln1_gamma.tobytes())
            f.write(ln1_beta.tobytes())
            f.write(w_q.tobytes())
            f.write(w_k.tobytes())
            f.write(w_v.tobytes())
            f.write(w_o.tobytes())
            f.write(ln2_gamma.tobytes())
            f.write(ln2_beta.tobytes())
            f.write(ff_w1.tobytes())
            f.write(ff_b1.tobytes())
            f.write(ff_w2.tobytes())
            f.write(ff_b2.tobytes())
        # ln_final
        f.write(ln_final_gamma.tobytes())
        f.write(ln_final_beta.tobytes())
        # lm_head
        f.write(lm_head.tobytes())

    print(f"Wrote {os.path.getsize(weights_path)} bytes to {weights_path}")

    # Save tokens.bin (little-endian i32)
    tokens_path = os.path.join(out_dir, "tokens.bin")
    with open(tokens_path, "wb") as f:
        f.write(TOKENS.tobytes())
    print(f"Wrote {os.path.getsize(tokens_path)} bytes to {tokens_path}")

    # Save logits.bin (little-endian f32)
    logits_flat = logits.astype(np.float32)
    logits_path = os.path.join(out_dir, "logits.bin")
    with open(logits_path, "wb") as f:
        f.write(logits_flat.tobytes())
    print(f"Wrote {os.path.getsize(logits_path)} bytes to {logits_path}")

    # Verify expected sizes
    expected_weight_floats = (
        VOCAB_SIZE * D_MODEL  # token_embedding
        + MAX_SEQ_LEN * D_MODEL  # pos_embedding
        + NUM_LAYERS * (
            D_MODEL  # ln1_gamma
            + D_MODEL  # ln1_beta
            + D_MODEL * D_MODEL  # w_q
            + D_MODEL * D_MODEL  # w_k
            + D_MODEL * D_MODEL  # w_v
            + D_MODEL * D_MODEL  # w_o
            + D_MODEL  # ln2_gamma
            + D_MODEL  # ln2_beta
            + D_MODEL * D_FF  # ff_w1
            + D_FF  # ff_b1
            + D_FF * D_MODEL  # ff_w2
            + D_MODEL  # ff_b2
        )
        + D_MODEL  # ln_final_gamma
        + D_MODEL  # ln_final_beta
        + D_MODEL * VOCAB_SIZE  # lm_head
    )
    print(f"Expected weight floats: {expected_weight_floats} ({expected_weight_floats * 4} bytes)")
    print(f"Expected logits floats: {BATCH_SIZE * SEQ_LEN * VOCAB_SIZE} ({BATCH_SIZE * SEQ_LEN * VOCAB_SIZE * 4} bytes)")
    print(f"Expected tokens ints: {BATCH_SIZE * SEQ_LEN} ({BATCH_SIZE * SEQ_LEN * 4} bytes)")


if __name__ == "__main__":
    main()
