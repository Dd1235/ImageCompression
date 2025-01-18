# Image compression and reconstruction pipeline using JPEG


Plan:

1. Preprocessing

- pad the images to ensure their dimensions align with patch sizes
- Divide the image into non-overlapping patches
2. Compression

- [lecture slides](https://www.commsp.ee.ic.ac.uk/~tania/teaching/DIP%202014/DCT%20other.pdf)
- [mathworks article](https://in.mathworks.com/help/images/discrete-cosine-transform.html?utm_source=chatgpt.com)

- Apply DCT to each patch


The mathematical definition of the DCT matrix entry for an \(N \times N\) matrix is:
\[
M[i][j] = \alpha(i) \cdot \cos\left(\frac{(2j+1) \cdot i \cdot \pi}{2N}\right)
\]
Where:
- \( \alpha(i) \) is the normalization factor:
  \[
  \alpha(i) =
  \begin{cases}
  \sqrt{\frac{1}{N}}, & \text{if } i = 0 \\
  \sqrt{\frac{2}{N}}, & \text{otherwise}
  \end{cases}
  \]
- \( i \) is the row index (frequency index).
- \( j \) is the column index (spatial index).
- \( N \) is the size of the matrix.



| Row\Col | 0          | 1          | 2          | 3          | 4          | 5          | 6          | 7          |
|---------|------------|------------|------------|------------|------------|------------|------------|------------|
| **0**   | \( \frac{1}{\sqrt{8}} \) | \( \frac{1}{\sqrt{8}} \) | \( \frac{1}{\sqrt{8}} \) | \( \frac{1}{\sqrt{8}} \) | \( \frac{1}{\sqrt{8}} \) | \( \frac{1}{\sqrt{8}} \) | \( \frac{1}{\sqrt{8}} \) | \( \frac{1}{\sqrt{8}} \) |
| **1**   | \( \sqrt{\frac{2}{8}} \cos\frac{\pi}{16} \) | \( \sqrt{\frac{2}{8}} \cos\frac{3\pi}{16} \) | \( \dots \) | ... | ... | ... | ... | ... |
| **2**   | \( \dots \) | \( \dots \) | \( \dots \) | \( \dots \) | \( \dots \) | \( \dots \) | \( \dots \) | \( \dots \) |
| ...     | ... | ... | ... | ... | ... | ... | ... | ... |
| **7**   | \( \sqrt{\frac{2}{8}} \cos\frac{7\pi}{16} \) | \( \sqrt{\frac{2}{8}} \cos\frac{21\pi}{16} \) | \( \dots \) | ... | ... | ... | ... | ... |

#### Observations:
1. The first row is constant, representing the average (DC component).
2. Subsequent rows represent increasing frequencies, showing how much of each cosine frequency contributes to the signal.
3. Values decay quickly for higher frequencies due to normalization.


- Quantization:

    - [khan academy article](https://www.khanacademy.org/computing/ap-computer-science-principles/x2d2f703b37b450a3:digital-information/x2d2f703b37b450a3:from-analog-to-digital-data/a/converting-analog-data-to-binary)

    - say intervel of 25, then all the amplitudes between 0-25 will be quantized to 0, 25-50 to 25, and so on, this will reduce the number of bits required to store the image, its like 75 is much easier to represent in binary than 74.341246 or something like that. Quantization is something like 'rounding' y values after sampling. 

    - this is how we remove the high frequency components
    - given an 8x8 matrix, divide each by a value (from quantization table) and round it off. So if the value in quantization table is high, leads to smaller value in the compressed matrix. 
    - in the decoding process, we multiply the compressed matrix element by element, so we purposefully end up losing information in this step. 
    eg if image[0][0] = 338. Standard value for quantization table is 16, then compressed_image[0][0] = 338/16 = 21.125, which is rounded off to 21. Then in the decoding process, we multiply 21 by 16 to get 336, which is not equal to 338, so we lose information here.
    - the quantization tables are provided from visual experiments.
    - in practice, there is a different table for the Luma(Y) and Chrominance(Cb, Cr) channels.
    - now exploit redundancy.