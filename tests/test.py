import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from skimage import color, data

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.jpeg_compression import *


def test():

    # --- Grayscale JPEG Compression Example ---
    # Load a grayscale image from skimage
    original_gray = color.rgb2gray(data.astronaut())
    original_gray = (original_gray * 255).astype(np.uint8)

    # Pad image
    patch_size = 8
    padded_gray = pad_image(original_gray, patch_size)
    patches = split_image_into_patches(padded_gray, patch_size)

    # DCT, Quantization, and RLE Encoding
    dct_patches = apply_dct(patches, patch_size)
    quantized_patches = apply_quantization(dct_patches, quality=50)

    # (Optional) RLE on each patch
    zz_order = generate_zigzag_order(patch_size)
    rle_encoded = [run_length_encode(patch, zz_order) for patch in quantized_patches]

    # Decompression steps
    # Decode RLE back to quantized patches
    decoded_patches = np.array(
        [run_length_decode(code, patch_size, zz_order) for code in rle_encoded]
    )
    dequantized = apply_dequantization(decoded_patches, quality=50)
    idct_patches = apply_idct(dequantized, patch_size)
    reconstructed_gray = combine_patches_into_image(
        idct_patches, padded_gray.shape, patch_size
    )

    # Display the results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original_gray, cmap="gray")
    plt.title("Original Grayscale")
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_gray, cmap="gray")
    plt.title("Reconstructed Grayscale")
    plt.show()

    # --- RGB JPEG Compression Example ---
    # Load an RGB image
    original_rgb = data.astronaut()
    compressed_rgb = compress_rgb_image(original_rgb, quality=50, patch_size=8)
    reconstructed_rgb = decompress_rgb_image(compressed_rgb, patch_size=8, quality=50)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title("Original RGB")
    plt.subplot(1, 2, 2)
    plt.imshow(np.clip(reconstructed_rgb, 0, 255).astype(np.uint8))
    plt.title("Reconstructed RGB")
    plt.show()

    # --- PCA-Based Compression Example for a Single Grayscale Image ---
    n_components = 50
    pca_data = pca_compress_image(original_gray, n_components)
    reconstructed_pca_gray = pca_reconstruct_image(pca_data)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original_gray, cmap="gray")
    plt.title("Original Grayscale")
    plt.subplot(1, 2, 2)
    plt.imshow(np.clip(reconstructed_pca_gray, 0, 255).astype(np.uint8), cmap="gray")
    plt.title("PCA-Reconstructed Grayscale")
    plt.show()

    # --- PCA-Based Compression Example for a Dataset of Grayscale Images ---
    # For demonstration, we duplicate the same image several times.
    images = [original_gray for _ in range(10)]
    pca_dataset = pca_compress_dataset(images, n_components=50)
    reconstructed_images = pca_reconstruct_dataset(pca_dataset)

    # Display one of the reconstructed images
    plt.figure()
    plt.imshow(np.clip(reconstructed_images[0], 0, 255).astype(np.uint8), cmap="gray")
    plt.title("PCA-Reconstructed from Dataset")
    plt.show()


if __name__ == "__main__":
    test()
