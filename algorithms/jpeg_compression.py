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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create a dummy grayscale image
    dummy_image = np.array(
        [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120], [130, 140, 150, 160]],
        dtype=np.uint8,
    )

    print("Original Image:")
    print(dummy_image)

    # Step 1: Pad the image
    patch_size = 3  # Example patch size
    padded_image = pad_image(dummy_image, patch_size)
    print("\nPadded Image:")
    print(padded_image)

    # Visualize padding
    plt.imshow(padded_image, cmap="gray")
    plt.title("Padded Image")
    plt.show()

    # Step 2: Split the image into patches
    patches = split_image_into_patches(padded_image, patch_size)
    print("\nExtracted Patches (Each 3x3):")
    for i, patch in enumerate(patches):
        print(f"Patch {i + 1}:\n{patch}")
