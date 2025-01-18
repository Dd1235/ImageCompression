import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.jpeg_compression import apply_dequantization, apply_quantization


def test_quantization_and_dequantization():
    """
    Test quantization and de-quantization process on a sample DCT patch.
    """

    dct_patch = np.array(
        [
            [160, 120, 80, 40, 20, 10, 5, 2],
            [90, 70, 50, 30, 15, 7, 3, 1],
            [45, 35, 25, 15, 7, 3, 1, 0],
            [22, 17, 12, 7, 3, 1, 0, 0],
            [11, 8, 6, 3, 1, 0, 0, 0],
            [5, 4, 3, 2, 1, 0, 0, 0],
            [3, 2, 2, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )

    print("Original DCT Patch:")
    print(dct_patch)

    quality = 50
    quantized_patch = apply_quantization(np.array([dct_patch]), quality=quality)[0]
    print("\nQuantized Patch:")
    print(quantized_patch)

    dequantized_patch = apply_dequantization(
        np.array([quantized_patch]), quality=quality
    )[0]
    print("\nDe-Quantized Patch:")
    print(dequantized_patch)

    diff = dct_patch - dequantized_patch
    print("\nDifference Between Original and De-Quantized Patch:")
    print(diff)

    assert np.allclose(
        dct_patch, dequantized_patch, atol=20
    ), "De-quantization differs significantly from the original DCT patch!"


if __name__ == "__main__":
    test_quantization_and_dequantization()
