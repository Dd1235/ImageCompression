import math

import numpy as np
from skimage.util import view_as_windows


def pad_image(image, patch_size=8):
    """
    Pad the given image so that its dimensions are multiple of patch size.

    Parameters:
    - image: 2D numpy array (grayscale image only implemented for now)
    - patch_size: int, size of the patch, default is 8x8

    Returns:
    - padded_image: 2D np array with dimensions multiple of patch size
    """

    height, width = image.shape

    # calculate the padding necessary
    pad_height = (patch_size - height % patch_size) % patch_size
    # modulo to handle case when height is multiple of patch size
    pad_width = (patch_size - width % patch_size) % patch_size

    padded_image = np.pad(
        array=image,
        pad_width=((0, pad_height), (0, pad_width)),
        mode="constant",
        constant_values=0,
    )
    return padded_image


def split_image_into_patches(image, patch_size=8):
    """
    Splits the image into non-overlapping patches of the specified size.

    Parameters:
    - image: 2D numpy array (grayscale image).
    - patch_size: Integer, size of the patch (default is 8x8).

    Returns:
    - patches: 3D numpy array where each patch is patch_size x patch_size.
    """
    # extract non-overlapping patches
    patches = view_as_windows(image, (patch_size, patch_size), step=patch_size)
    # (numrows, numcols, patch_size, patch_size)

    # Reshape patches into a 3D array for easier processing
    patches = patches.reshape(
        -1, patch_size, patch_size
    )  # -1, infer based on the other dimensions, here it is numrows*numcols
    print("Reshaped Patches shape:", patches.shape)

    return patches


def combine_patches_into_image(patches, image_shape, patch_size=8):
    """
    Combine non-overlapping patches back into an image.

    Parameters:
    - patches: 3D numpy array where each slice is a patch of size (patch_size x patch_size).
    - image_shape: Tuple, shape of the original image.

    Returns:
    - padded_image: 2D numpy array with dimensions same as the original image.
    """
    padded_image = np.zeros(image_shape, dtype=patches.dtype)
    n_rows = image_shape[0] // patch_size
    n_cols = image_shape[1] // patch_size
    idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            padded_image[
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ] = patches[idx]
            idx += 1
    return padded_image


def generate_dct_matrix(size):
    """
    Generate the DCT matrix of the given size

    Parameters:
    - size: integer, the size of the DCT matrix(eg. 8 for 8x8 DCT matrix)

    Returns:
    - dct_matrix: 2D numpy array of shape (size, size)
    """

    dct_matrix = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            a = math.sqrt(1 / size) if i == 0 else math.sqrt(2 / size)
            dct_matrix[i, j] = a * math.cos(((2 * j + 1) * i * math.pi) / (2 * size))
    return dct_matrix


def apply_dct(patches, patch_size=8):
    """
    Apply 2D dct to each patch

    Parameters:
    - patches: 3D np array, with each slice representing a patch of (patch_size x patch_size)
    - patch_size: int, size of the patch

    Returns:
    - dct_patches: 3D np array of DCT-transformed patches
    """
    dct_matrix = generate_dct_matrix(patch_size)
    dct_transpose = dct_matrix.T
    dct_patches = np.empty_like(patches, dtype=np.float64)

    for idx, patch in enumerate(patches):
        dct_patches[idx] = np.dot(dct_matrix, np.dot(patch, dct_transpose))

    return dct_patches


def apply_idct(dct_patches, patch_size=8):
    """
    Apply 2D Inverse DCT to each patch

    Parameters:
    - dct_patches: 3D np array, with each slice representing a DCT-transformed patch of (patch_size x patch_size)
    - patch_size: int, size of the patch

    Returns:
    - reconstructed_patches: 3D np array of IDCT-transformed patches
    """

    dct_matrix = generate_dct_matrix(patch_size)
    dct_transpose = dct_matrix.T
    reconstructed_patches = np.empty_like(dct_patches, dtype=np.float64)

    for idx, dct_patch in enumerate(dct_patches):
        reconstructed_patches[idx] = np.dot(
            dct_transpose, np.dot(dct_patch, dct_matrix)
        )

    return reconstructed_patches


def create_quantization_matrix(quality=50):
    """
    Create a standard JPEG quantization matrix scaled by the quality factor

    Parameters:
    - quality: integer, quality factor(1 to 100)Higher quality means lower compression

    Returns:
    - quantization matrix: 2D np array (8x8)
    """

    # Base quantization matrix (JPEG standard)
    base_matrix = np.array(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ],
        dtype=np.float64,
    )

    scale = (5000 / quality) if quality < 50 else (200 - 2 * quality)
    quantization_matrix = np.floor((scale * base_matrix + 50) / 100).astype(np.int32)

    # ensure no zero values
    quantization_matrix[quantization_matrix == 0] = 1

    return quantization_matrix


def apply_quantization(dct_patches, quality=50):
    """
    apply quantization to dct transformed patches

    parameters:
    - dct_paches: 3d np array, each slide is a dct-transformed patch
    - quality: integer, quality factor (1, 100)

    returns:
    - quantized_patches: 3d np array of quantized patches
    """

    quantization_matrix = create_quantization_matrix(quality)
    quantized_patches = np.empty_like(dct_patches, dtype=np.int32)

    for idx, dct_patch in enumerate(dct_patches):
        quantized_patches[idx] = np.round(dct_patch / quantization_matrix)

    return quantized_patches


def apply_dequantization(quantized_patches, quality=50):
    """
    Applies dequantization to quantized DCT patches

    Parameters:
    - quantized_patches: 3D np array, with each slice representing a quantized DCT-transformed patch of (patch_size x patch_size)
    - quality: Integer, quality factor (1 to 100)

    Returns:
    - dct_patches: 3D np array of dequantized patches
    """

    quantization_matrix = create_quantization_matrix(quality)

    dct_patches = np.empty_like(quantized_patches, dtype=np.float64)

    for idx, quantized_patch in enumerate(quantized_patches):
        dct_patches[idx] = quantized_patch * quantization_matrix

    return dct_patches


def generate_zigzag_order(block_size=8):
    """
    Generates the zigzag traversal order

    Parameters:
    - bloc_size: int, size of the block (default is 8x8)

    Returns:
    - zigzag_order: List of (row,col) index pairs in zigzag order
    """

    indices = np.indices((block_size, block_size)).reshape(2, -1).T
    diag_indices = sorted(indices, key=lambda x: (x[0] + x[1], x[0]))
    return diag_indices


def run_length_encode(block, zigzag_order):
    """
    Encodes an 8x8 block using run-length encoding in zigzag order

    Parameters:
    - block: 2D np array of shape (8,8) quantized coefficients
    - zigzag_order: List of (row,col) index pairs in zigzag order

    Returns:
    - encoded: List of (skip, value) pairs
    """

    # for now only takes care of repeating zeros

    encoded = []
    skip = 0

    for row, col in zigzag_order:
        value = block[row, col]
        if value == 0:
            skip += 1
        else:
            encoded.append((skip, value))
            skip = 0

    encoded.append((0, 0))  # End of block marker
    return encoded


def run_length_decode(encoded, block_size=8, zigzag_order=None):
    """
    Decodes a list of Run-Length Encoded (RLE) pairs back into an 8x8 block.

    Parameters:
    - encoded: List of (skip, value) pairs.
    - block_size: Integer, size of the block (default is 8).
    - zigzag_order: List of (row, col) index pairs in zigzag order.

    Returns:
    - block: 2D numpy array (decoded 8x8 block).
    """
    if zigzag_order is None:
        zigzag_order = generate_zigzag_order(block_size)

    block = np.zeros((block_size, block_size), dtype=np.int32)
    position = 0

    for skip, value in encoded:
        if (skip, value) == (0, 0):
            break
        position += skip
        row, col = zigzag_order[position]
        block[row, col] = value
        position += 1

    return block


# ==============================
# Huffman Coding for RLE Data
# ==============================


# A simple node class for the Huffman tree
class HuffmanNode:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def build_frequency_table(rle_data):
    """
    Build a frequency table for symbols in the RLE data.
    rle_data: list of RLE lists (each is a list of (skip, value) pairs)
    """
    freq = {}
    for patch in rle_data:
        for symbol in patch:
            freq[symbol] = freq.get(symbol, 0) + 1
    return freq


def build_huffman_tree(freq_table):
    """
    Build the Huffman tree given a frequency table.
    """
    import heapq

    heap = []
    for symbol, freq in freq_table.items():
        heapq.heappush(heap, HuffmanNode(symbol, freq))
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq, left, right)
        heapq.heappush(heap, merged)
    return heap[0]


def generate_huffman_codes(node, prefix="", code_table=None):
    """
    Recursively traverse the Huffman tree to generate a code table.
    """
    if code_table is None:
        code_table = {}
    if node.symbol is not None:
        code_table[node.symbol] = prefix
    else:
        generate_huffman_codes(node.left, prefix + "0", code_table)
        generate_huffman_codes(node.right, prefix + "1", code_table)
    return code_table


def build_decoding_tree(code_table):
    """
    Build a decoding tree (as nested dictionaries) from the Huffman code table.
    """
    root = {}
    for symbol, code in code_table.items():
        node = root
        for bit in code:
            if bit not in node:
                node[bit] = {}
            node = node[bit]
        node["symbol"] = symbol
    return root


def huffman_encode_rle(rle_data):
    """
    Apply Huffman encoding to a list of RLE lists.
    Returns a tuple (encoded_patches, code_table) where encoded_patches is a list
    of bit strings for each patch.
    """
    freq_table = build_frequency_table(rle_data)
    tree = build_huffman_tree(freq_table)
    code_table = generate_huffman_codes(tree)
    encoded_patches = []
    for patch in rle_data:
        encoded = "".join(code_table[symbol] for symbol in patch)
        encoded_patches.append(encoded)
    return encoded_patches, code_table


def huffman_decode_string(encoded, decoding_tree):
    """
    Decode a single Huffman-encoded bitstring using the decoding tree.
    """
    decoded = []
    node = decoding_tree
    for bit in encoded:
        node = node[bit]
        if "symbol" in node:
            decoded.append(node["symbol"])
            node = decoding_tree
    return decoded


def huffman_decode_rle(encoded_patches, code_table):
    """
    Decode a list of Huffman-encoded bit strings back into RLE lists.
    """
    decoding_tree = build_decoding_tree(code_table)
    decoded_patches = []
    for encoded in encoded_patches:
        patch = huffman_decode_string(encoded, decoding_tree)
        decoded_patches.append(patch)
    return decoded_patches


# ==============================
# Image Compression / Decompression Pipeline
# ==============================


# def compress_rgb_image(image, quality=50, patch_size=8, use_huffman=False):
#     """
#     Compress an RGB image by processing each channel independently.
#     If use_huffman is True, applies Huffman encoding on the RLE output.
#     Returns a list (one per channel) with compressed data.
#     """
#     compressed_channels = []
#     h, w, _ = image.shape
#     padded_channels = [pad_image(image[:, :, i], patch_size) for i in range(3)]

#     for channel in padded_channels:
#         patches = split_image_into_patches(channel, patch_size)
#         dct_patches = apply_dct(patches, patch_size)
#         quantized_patches = apply_quantization(dct_patches, quality)
#         zz_order = generate_zigzag_order(patch_size)
#         rle_encoded = [
#             run_length_encode(patch, zz_order) for patch in quantized_patches
#         ]

#         channel_data = {"shape": channel.shape}
#         if use_huffman:
#             encoded_huff, code_table = huffman_encode_rle(rle_encoded)
#             channel_data["huffman"] = {
#                 "encoded": encoded_huff,
#                 "code_table": code_table,
#             }
#         else:
#             channel_data["rle"] = rle_encoded

#         compressed_channels.append(channel_data)
#     return compressed_channels


# def decompress_rgb_image(
#     compressed_channels, patch_size=8, quality=50, use_huffman=False
# ):
#     """
#     Decompress an RGB image from its compressed channels.
#     """
#     decompressed_channels = []
#     for channel_data in compressed_channels:
#         shape = channel_data["shape"]
#         zz_order = generate_zigzag_order(patch_size)
#         if use_huffman:
#             encoded = channel_data["huffman"]["encoded"]
#             code_table = channel_data["huffman"]["code_table"]
#             rle_encoded = huffman_decode_rle(encoded, code_table)
#         else:
#             rle_encoded = channel_data["rle"]

#         quantized_patches = [
#             run_length_decode(encoded, patch_size, zz_order) for encoded in rle_encoded
#         ]
#         quantized_patches = np.array(quantized_patches)
#         dct_patches = apply_dequantization(quantized_patches, quality)
#         patches = apply_idct(dct_patches, patch_size)
#         decompressed_channel = combine_patches_into_image(patches, shape, patch_size)
#         decompressed_channels.append(decompressed_channel)
#     decompressed_image = np.stack(decompressed_channels, axis=-1)
#     return decompressed_image


def compress_rgb_image(image, quality=50, patch_size=8):
    """
    Compress an RGB image by processing each channel independently.
    Returns a dictionary with compressed data for each channel.
    """
    compressed_channels = []
    h, w, _ = image.shape
    # Ensure that height and width are multiples of patch_size for each channel
    padded_channels = []
    for i in range(3):
        padded = pad_image(image[:, :, i], patch_size)
        padded_channels.append(padded)

    # Process each channel
    for channel in padded_channels:
        patches = split_image_into_patches(channel, patch_size)
        dct_patches = apply_dct(patches, patch_size)
        quantized_patches = apply_quantization(dct_patches, quality)
        # apply RLE for each patch using zigzag order
        zz_order = generate_zigzag_order(patch_size)
        rle_encoded = [
            run_length_encode(patch, zz_order) for patch in quantized_patches
        ]
        compressed_channels.append(
            {
                "shape": channel.shape,
                "rle": rle_encoded,
            }
        )
    return compressed_channels


def decompress_rgb_image(compressed_channels, patch_size=8, quality=50):
    """
    Decompress an RGB image from its compressed channels.
    """
    decompressed_channels = []
    for channel_data in compressed_channels:
        shape = channel_data["shape"]
        rle_encoded = channel_data["rle"]
        zz_order = generate_zigzag_order(patch_size)
        # Decode RLE for each patch
        quantized_patches = [
            run_length_decode(encoded, patch_size, zz_order) for encoded in rle_encoded
        ]
        quantized_patches = np.array(quantized_patches)
        # Dequantize and then apply inverse DCT on each patch
        dct_patches = apply_dequantization(quantized_patches, quality)
        patches = apply_idct(dct_patches, patch_size)
        # Combine patches back to full channel image
        decompressed_channel = combine_patches_into_image(patches, shape, patch_size)
        decompressed_channels.append(decompressed_channel)
    # Stack channels to form an RGB image
    decompressed_image = np.stack(decompressed_channels, axis=-1)
    return decompressed_image


# PCA implementation
def pca_compress_image(image, n_components):
    """
    Compress a single image (grayscale or RGB) using PCA (via SVD).
    For grayscale images, `image` is 2D; for RGB images, it is 3D.

    Returns a dictionary with compressed components and the mean.
    """
    # For grayscale images
    if image.ndim == 2:
        # Subtract the mean
        mean = np.mean(image, axis=0)
        centered = image - mean
        # Compute SVD
        U, S, VT = np.linalg.svd(centered, full_matrices=False)
        # Keep n_components
        U_reduced = U[:, :n_components]
        S_reduced = S[:n_components]
        VT_reduced = VT[:n_components, :]
        compressed = {
            "mean": mean,
            "U": U_reduced,
            "S": S_reduced,
            "VT": VT_reduced,
            "shape": image.shape,
        }
        return compressed
    elif image.ndim == 3:
        # For RGB images, process each channel independently
        channels = []
        for i in range(3):
            comp = pca_compress_image(image[:, :, i], n_components)
            channels.append(comp)
        return channels
    else:
        raise ValueError("Unsupported image dimension.")


def pca_reconstruct_image(compressed):
    """
    Reconstruct a PCA-compressed image.
    Works for grayscale images (2D) and for RGB images (list of 3 channel dicts).
    """
    if isinstance(compressed, dict):
        # Grayscale reconstruction
        U = compressed["U"]
        S = compressed["S"]
        VT = compressed["VT"]
        mean = compressed["mean"]
        # Reconstruct the centered image
        centered_reconstructed = np.dot(U, np.dot(np.diag(S), VT))
        reconstructed = centered_reconstructed + mean
        return reconstructed
    elif isinstance(compressed, list):
        # RGB reconstruction: process each channel and stack them
        channels = [pca_reconstruct_image(comp) for comp in compressed]
        return np.stack(channels, axis=-1)
    else:
        raise ValueError("Invalid compressed data format.")


def pca_compress_dataset(images, n_components):
    """
    Compress a dataset of grayscale images using PCA.
    Each image is flattened to form a vector.

    Parameters:
    - images: list or array of grayscale images (each as 2D array)
    - n_components: number of principal components to retain.

    Returns a dictionary with:
      - mean: the mean image (flattened)
      - components: principal components (eigenvectors)
      - coefficients: projection of each image on the components
      - original_shape: shape of each image (assumed to be same for all images)
    """
    # Flatten each image
    flat_images = [img.flatten() for img in images]
    data = np.vstack(flat_images)
    mean = np.mean(data, axis=0)
    centered_data = data - mean

    # Compute SVD on the dataset
    U, S, VT = np.linalg.svd(centered_data, full_matrices=False)
    components = VT[:n_components, :]
    coefficients = np.dot(centered_data, components.T)
    return {
        "mean": mean,
        "components": components,
        "coefficients": coefficients,
        "original_shape": images[0].shape,
    }


def pca_reconstruct_dataset(pca_data):
    """
    Reconstruct images from PCA-compressed dataset.
    """
    mean = pca_data["mean"]
    components = pca_data["components"]
    coefficients = pca_data["coefficients"]
    reconstructed_flat = np.dot(coefficients, components) + mean
    # Reshape each image
    reconstructed_images = [
        img_flat.reshape(pca_data["original_shape"]) for img_flat in reconstructed_flat
    ]
    return reconstructed_images
