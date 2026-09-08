"""Differentiable 3-D complex FFT for Jittor 1.3.11.

Jittor exposes a differentiable two-dimensional cuFFT operator.  A 3-D FFT is
separable, so one plane transform followed by one transform along the remaining
axis provides the required operation without leaving the Jittor graph.
Complex tensors use a final dimension of size two: real, imaginary.
"""

from __future__ import annotations

import jittor as jt
from jittor import nn


def fft3d(value: jt.Var, inverse: bool = False) -> jt.Var:
    if value.ndim != 5 or value.shape[-1] != 2:
        raise ValueError("fft3d expects [batch, depth, height, width, 2]")
    if value.dtype not in (jt.float32, jt.float64):
        raise TypeError("fft3d supports float32 and float64 complex pairs")

    batch, depth, height, width, _ = value.shape
    plane_input = value.reshape((batch * depth, height, width, 2)).contiguous()
    plane_freq = nn._fft2(plane_input, inverse=inverse)

    axis_input = (
        plane_freq.reshape((batch, depth, height, width, 2))
        .permute([0, 2, 3, 1, 4])
        .reshape((batch * height * width, depth, 1, 2))
        .contiguous()
    )
    axis_freq = nn._fft2(axis_input, inverse=inverse)
    return (
        axis_freq.reshape((batch, height, width, depth, 2))
        .permute([0, 3, 1, 2, 4])
        .contiguous()
    )


def complex_multiply(left: jt.Var, right: jt.Var) -> jt.Var:
    if left.shape[-1] != 2 or right.shape[-1] != 2:
        raise ValueError("complex_multiply expects real/imaginary pairs")
    real = left[..., 0] * right[..., 0] - left[..., 1] * right[..., 1]
    imag = left[..., 0] * right[..., 1] + left[..., 1] * right[..., 0]
    return jt.stack([real, imag], dim=-1)


def hermitian_expand_last_axis(half_spectrum: jt.Var, full_width: int) -> jt.Var:
    """Expand an rFFT half spectrum for use with the full complex inverse FFT."""
    if half_spectrum.ndim != 5 or half_spectrum.shape[-1] != 2:
        raise ValueError("half spectrum must be [batch, depth, height, half_width, 2]")
    expected_half = full_width // 2 + 1
    if half_spectrum.shape[3] != expected_half:
        raise ValueError("half spectrum width does not match full_width")

    depth = half_spectrum.shape[1]
    height = half_spectrum.shape[2]
    reverse_depth = jt.array([(-index) % depth for index in range(depth)], dtype=jt.int32)
    reverse_height = jt.array([(-index) % height for index in range(height)], dtype=jt.int32)
    tail_source = list(range((full_width - 1) // 2, 0, -1))
    if not tail_source:
        return half_spectrum
    tail_index = jt.array(tail_source, dtype=jt.int32)

    mirrored = half_spectrum[:, reverse_depth]
    mirrored = mirrored[:, :, reverse_height]
    mirrored = mirrored[:, :, :, tail_index]
    mirrored = jt.stack([mirrored[..., 0], -mirrored[..., 1]], dim=-1)
    return jt.concat([half_spectrum, mirrored], dim=3)
