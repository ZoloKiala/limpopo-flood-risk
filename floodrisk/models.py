"""Shared ViT components (identical architecture to the research notebooks)."""
import tensorflow as tf
from tensorflow.keras import layers

from . import config


class TransformerBlock(layers.Layer):
    """Pre-norm encoder block: x = x + MHA(Norm(x)); x = x + MLP(Norm(x))."""

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout),
        ])

    def call(self, x, training=False):
        h = self.norm1(x)
        x = x + self.attn(h, h, training=training)
        x = x + self.mlp(self.norm2(x), training=training)
        return x


class PatchViT(tf.keras.Model):
    """Generic dense-prediction ViT: (B, H, W, C) -> (B, H, W, 1) logits."""

    def __init__(self, grid_h, grid_w, patch, channels,
                 embed_dim=config.EMBED_DIM, depth=config.DEPTH,
                 num_heads=config.NUM_HEADS, mlp_dim=config.MLP_DIM, **kwargs):
        super().__init__(**kwargs)
        self.p = patch
        self.gh, self.gw = grid_h // patch, grid_w // patch
        self.projection = layers.Dense(embed_dim)
        self.pos_embedding = self.add_weight(
            name="pos", shape=(1, self.gh * self.gw, embed_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True)
        self.blocks = [TransformerBlock(embed_dim, num_heads, mlp_dim)
                       for _ in range(depth)]
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.pixel_head = layers.Dense(patch * patch)

    def call(self, x, training=False):
        b, p = tf.shape(x)[0], self.p
        t = tf.image.extract_patches(
            x, sizes=[1, p, p, 1], strides=[1, p, p, 1],
            rates=[1, 1, 1, 1], padding="VALID")
        t = tf.reshape(t, (b, -1, p * p * x.shape[-1]))
        t = self.projection(t) + self.pos_embedding
        for blk in self.blocks:
            t = blk(t, training=training)
        t = self.pixel_head(self.norm(t))
        t = tf.reshape(t, (b, self.gh, self.gw, p * p))
        return tf.nn.depth_to_space(t, p)


def masked_bce(y_true, y_pred):
    """Binary cross-entropy ignoring pixels labeled -1 (nodata / masked)."""
    y_true = tf.cast(y_true, tf.float32)
    valid = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
    y = tf.clip_by_value(y_true, 0.0, 1.0)
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_pred)
    return tf.reduce_sum(bce * valid) / (tf.reduce_sum(valid) + 1e-6)
