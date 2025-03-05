# Image Processing Algorithms

This repository contains implementations of various image processing algorithms, covering techniques for filtering, segmentation, blending, and more. Each algorithm has its own directory with corresponding input materials and output results.

## Table of Contents
1. [Active Contour](#1-active-contour)
2. [Color Processing and Blurring](#2-color-processing-and-blurring)
3. [Convolution](#3-convolution)
4. [Histogram Specification](#4-histogram-specification)
5. [Homography and Image Warping](#5-homography-and-image-warping)
6. [Hough Transform](#6-hough-transform)
7. [Hybrid Images](#7-hybrid-images)
8. [Image Completion and Hole Filling](#8-image-completion-and-hole-filling)
9. [Image Enhancement](#9-image-enhancement)
10. [Image Morphing](#10-image-morphing)
11. [Bird Segmentation in Images ](#11-bird-segmentation-in-images)
12. [Image Sharpening](#12-image-sharpening)
13. [K-means Clustering](#13-k-means-clustering)
14. [Mean-Shift Clustering](#14-mean-shift-clustering)
15. [Image Blending](#15-image-blending)
16. [Prokudin-Gorskii Images](#16-prokudin-gorskii-images)
17. [SLIC Superpixel Segmentation](#17-slic-superpixel-segmentation)
18. [Template Matching](#18-template-matching)
19. [Texture Synthesis](#19-texture-synthesis)

## Usage
Each algorithm is implemented in a separate directory. To explore an algorithm:
1. Navigate to the corresponding directory.
2. Inside the `input/` folder, you'll find the input materials required for the algorithm.
3. The `output/` folder contains the final results and intermediate outputs.

---

## Algorithms

### 1. Active Contour
- Implements active contour (snakes) for image segmentation.

  ![Active Contour Video](/Active%20Contours/Output/contour.gif)

### 2. Color Processing and Blurring
- Changes the color of a flower in the image using color transformations.
<table>
  <tr>
    <td><img src="Color Processing and Blurring/Input/Flowers.jpg" width="250" height="250"></td>
    <td><img src="Color Processing and Blurring/Output/res06.jpg" width="250" height="250"></td>
  </tr>
</table>


### 3. Convolution
- 

### 4. Histogram Specification
- Adjusts an image’s histogram to match a specified target histogram.
<table>
  <tr>
    <td><img src="Histogram Specification/Input/Pink.jpg" width="200" height="200"></td>
    <td><img src="Histogram Specification/Output/Flower Histogram.jpg" width="200" height="200"></td>
      </tr>
</table>

<table>
  <tr>
    <td><img src="Histogram Specification/Input/Dark.jpg" width="200" height="200"></td>
    <td><img src="Histogram Specification/Output/res11.jpg" width="200" height="200"></td>
    <td><img src="Histogram Specification/Output/res10.jpg" width="200" height="200"></td>
      </tr>
</table>
 

### 5. Homography and Image Warping
- Computes homography transformations and warps images accordingly.

### 6. Hough Transform
- Detects lines and shapes in images using the Hough Transform.

### 7. Hybrid Images
- Blends high and low frequencies of two images to create hybrid images.
<table>
  <tr>
    <td><img src="Hybrid Images/Input/res19-near.jpg" width="200" height="200"></td>
    <td><img src="Hybrid Images/Input/res20-far.jpg" width="200" height="200"></td>
      </tr>
</table>

<table>
  <tr>
    <td><img src="Hybrid Images/Output/res30-hybrid-near.jpg" width="200" height="200"></td>
    <td><img src="Hybrid Images/Output/res31-hybrid-far.jpg" width="50" height="50"></td>
      </tr>
</table>

### 8. Image Completion and Hole Filling
- Fills missing parts of an image using inpainting techniques.
<table>
  <tr>
    <td><img src="Image Completion and Hole Filling/Input/im04.jpg" width="170" height="230"></td>
    <td><img src="Image Completion and Hole Filling/Output/res16.jpg" width="170" height="230"></td>
    <td><img src="Image Completion and Hole Filling/Input/im03.jpg" width="300" height="230"></td>
    <td><img src="Image Completion and Hole Filling/Output/res15.jpg" width="300" height="230"></td>
  </tr>
</table>


### 9. Image Enhancement
- Enhances image quality using contrast adjustment.
<table>
  <tr>
    <td><img src="Image Enhancement/Input/Enhance1.JPG" width="370" height="250"></td>
    <td><img src="Image Enhancement/Output/res01.jpg" width="370" height="250"></td>
  </tr>
</table>


### 10. Image Morphing
- Morphs one image into another through feature interpolation.

  **Morphing Video:**  
  ![Morphing Video](/Image%20Morphing/Output/morph.gif)

### 11. Bird Segmentation in Images  
- Designed a method to segment birds in images using Felzenszwalb segmentation and K-Means clustering.
<table>
  <tr>
    <td><img src="Image Segmentation/Input/birds.jpg" width="370" height="250"></td>
    <td><img src="Image Segmentation/Output/res10.jpg" width="370" height="250"></td>
  </tr>
</table>

### 12. Image Sharpening
- Applies sharpening filters to enhance edge details.

### 13. K-means Clustering
- Implemented the K-Means algorithm.

### 14. Mean-Shift Clustering
- Segments images using the mean-shift clustering algorithm.

<table>
  <tr>
    <td><img src="Mean-Shift/Input/park.jpg" width="300" height="300"></td>
    <td><img src="Mean-Shift/Output/res05.jpg" width="300" height="300"></td>
  </tr>
</table>

### 15. Image Blending
This section includes two powerful image blending techniques:  

- **Multiresolution Blending and Feathering**: Seamlessly blends multiple images at different resolutions, ensuring smooth transitions between them.  
- **Poisson Blending**: Uses Poisson Image Editing to integrate images naturally by preserving gradient consistency.  

<table>
  <tr>
    <td><img src="Poisson Blending/Input/res05.jpg" width="260" height="160"></td>
    <td><img src="Poisson Blending/Input/res06.jpg" width="260" height="160"></td>
    <td><img src="Poisson Blending/Output/res07.jpg" width="260" height="160"></td>
  </tr>
</table>


### 16. Prokudin-Gorskii Images
- Aligns and reconstructs historical color images from glass negatives.

### 17. SLIC Superpixel Segmentation
- Segments images into superpixels using the SLIC algorithm.
<table>
  <tr>
    <td><img src="SLIC/Input/slic.jpg" width="260" height="160"></td>
    <td><img src="SLIC/Output/res06.jpg" width="260" height="160"></td>
    <td><img src="SLIC/Output/res07.jpg" width="260" height="160"></td>
    <td><img src="SLIC/Output/res08.jpg" width="260" height="160"></td>
    <td><img src="SLIC/Output/res09.jpg" width="260" height="160"></td>
  </tr>
</table>


### 18. Template Matching
- Finds matching patterns in images using template matching.
<table>
  <tr>
    <td><img src="Template Matching/Input/Greek-ship.jpg" width="300"></td>
    <td><img src="Template Matching/Input/patch.png" width="50"></td>
    <td><img src="Template Matching/Output/res15.jpg" width="300"></td>
  </tr>
</table>

### 19. Texture Synthesis
- Generates textures from example patches using synthesis algorithms.


<table>
  <tr>
    <td><img src="Texture Synthesis/Input/texture06.jpg" width="100" height="100"> </td>
    <td><img src="Texture Synthesis/Output/res11.jpg" width="300"></td>
    <td><img src="Texture Synthesis/Input/myTexture2.jpg" width="100" height="100"></td>
    <td><img src="Texture Synthesis/Output/res14.jpg" width="300"></td>
  </tr>
</table>
