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
            a = (
                math.sqrt(1 / size)
                if i == 0
                else math.sqrt(1 / size) if i == 0 else math.sqrt(2 / size)
            )
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
