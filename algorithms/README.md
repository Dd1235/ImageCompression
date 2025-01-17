# Image compression and reconstruction pipeline using JPEG


Plan:

1. Preprocessing
- pad the images to ensure their dimensions align with patch sizes
- Divide the image into non-overlapping patches
2. Compression


Some functions from external libraries that are prerequisites for understanding:

- `np.pad(...)`
- `skimage.util.view_as_windows(arr_in, window_shape, step=1)`