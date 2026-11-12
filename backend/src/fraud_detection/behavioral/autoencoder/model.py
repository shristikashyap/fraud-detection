"""Keras sequence autoencoder architecture and masked reconstruction errors."""

from __future__ import annotations

import numpy as np


def build_autoencoder(sequence_length: int, feature_count: int, *, latent_units: int = 8):
    """Build a dense encoder-decoder over fixed-length encoded sequences."""
    if sequence_length < 1 or feature_count < 1 or latent_units < 1:
        raise ValueError("Autoencoder dimensions must be positive.")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for Phase 5 model training. "
            "Install the backend 'tensorflow' extra."
        ) from exc

    input_size = sequence_length * feature_count
    encoder_units = max(latent_units * 2, input_size // 2)
    inputs = tf.keras.layers.Input(
        shape=(sequence_length, feature_count), name="activity_sequence"
    )
    flattened = tf.keras.layers.Flatten(name="flatten_sequence")(inputs)
    encoded = tf.keras.layers.Dense(encoder_units, activation="relu")(flattened)
    latent = tf.keras.layers.Dense(latent_units, activation="relu", name="latent")(encoded)
    decoded = tf.keras.layers.Dense(encoder_units, activation="relu")(latent)
    reconstructed = tf.keras.layers.Dense(input_size, activation="linear")(decoded)
    outputs = tf.keras.layers.Reshape(
        (sequence_length, feature_count), name="reconstructed_sequence"
    )(reconstructed)

    model = tf.keras.Model(inputs, outputs, name="behavioral_sequence_autoencoder")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )
    return model


def masked_reconstruction_errors(
    original: np.ndarray, reconstructed: np.ndarray
) -> np.ndarray:
    """Calculate per-window MSE over valid timesteps, ignoring left padding."""
    original = np.asarray(original, dtype=np.float32)
    reconstructed = np.asarray(reconstructed, dtype=np.float32)
    if original.shape != reconstructed.shape or original.ndim != 3:
        raise ValueError("Original and reconstructed batches must have equal 3D shapes.")
    if original.shape[-1] < 1:
        raise ValueError("Sequence feature dimension must not be empty.")
    mask = original[..., -1]
    squared_errors = np.mean(np.square(original - reconstructed), axis=-1)
    valid_counts = mask.sum(axis=1)
    if np.any(valid_counts <= 0):
        raise ValueError("Every sequence must contain at least one valid timestep.")
    errors = (squared_errors * mask).sum(axis=1) / valid_counts
    if not np.isfinite(errors).all():
        raise ValueError("Reconstruction errors must be finite.")
    return errors.astype(np.float64)
