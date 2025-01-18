import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.jpeg_compression import (
    generate_zigzag_order,
    run_length_decode,
    run_length_encode,
)


def test_rle():
    quantized_block = np.array(
        [
            [10, 3, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    print("Original Quantized Block:")
    print(quantized_block)

    # Step 2: Generate zigzag order
    zigzag_order = generate_zigzag_order()
    print("\nZigzag Order:")
    print(zigzag_order)

    # Step 3: Run-Length Encoding
    encoded = run_length_encode(quantized_block, zigzag_order)
    print("\nRun-Length Encoded Data:")
    print(encoded)

    # Step 4: Run-Length Decoding
    decoded_block = run_length_decode(encoded, block_size=8, zigzag_order=zigzag_order)
    print("\nDecoded Block:")
    print(decoded_block)

    # Step 5: Verify Reconstruction
    diff = quantized_block - decoded_block
    print("\nDifference Between Original and Decoded Block:")
    print(diff)


if __name__ == "__main__":
    test_rle()
