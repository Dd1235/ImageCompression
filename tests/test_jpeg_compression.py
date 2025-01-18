import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.jpeg_compression import apply_dct, apply_idct

# pip install pytest
# pytest tests/


def test_dct_idct_cycle():
    # Create a test patch (8x8 example)
    test_patch = np.array(
        [
            [52, 55, 61, 66, 70, 61, 64, 73],
            [63, 59, 66, 90, 109, 85, 69, 72],
            [62, 59, 68, 113, 144, 104, 66, 73],
            [63, 58, 71, 122, 154, 106, 70, 69],
            [67, 61, 68, 104, 126, 88, 68, 70],
            [79, 65, 60, 70, 77, 68, 58, 75],
            [85, 71, 64, 59, 55, 61, 65, 83],
            [87, 79, 69, 68, 65, 76, 78, 94],
        ],
        dtype=np.float64,
    )

    # Apply DCT
    dct_patch = apply_dct(np.array([test_patch]))[0]

    # Apply IDCT
    reconstructed_patch = apply_idct(np.array([dct_patch]))[0]

    # Assert the difference is minimal
    assert np.allclose(
        test_patch, reconstructed_patch, atol=1e-6
    ), "IDCT did not perfectly reconstruct the original patch!"


def test_dct_output():
    # Create a simple test patch
    test_patch = np.ones((8, 8), dtype=np.float64)

    # Apply DCT
    dct_patch = apply_dct(np.array([test_patch]))[0]

    # Verify the DC component (top-left) is non-zero, and others are zero for uniform input
    assert np.isclose(
        dct_patch[0, 0], np.sqrt(8 * 8)
    ), "DC component is incorrect for uniform input!"
    assert np.allclose(
        dct_patch[1:, :], 0
    ), "Non-DC components should be zero for uniform input!"


def test_idct_output():
    # Create a simple DCT patch with only a DC component
    dct_patch = np.zeros((8, 8), dtype=np.float64)
    dct_patch[0, 0] = np.sqrt(8 * 8)

    # Apply IDCT
    reconstructed_patch = apply_idct(np.array([dct_patch]))[0]

    # Verify the reconstructed patch is uniform
    expected_patch = np.ones((8, 8), dtype=np.float64)
    assert np.allclose(
        reconstructed_patch, expected_patch
    ), "IDCT output is incorrect for a pure DC component!"
