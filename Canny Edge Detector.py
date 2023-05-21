# Filename: hw2Completed.py
# Author: Sarah Whynott
# Date Created: September 11, 2022
# Homework 2 - Create a Canny Edge Detector

import cv2
import numpy as np
import math
from tkinter.filedialog import askopenfilename

def convolution(src, kernel):
    """
    Convoles an input image with the given kernel.
    """

    # Copy your code from class here. Please do not import your
    # convolution file and call that function from here. It
    # will cause issues when grading.

    outputArr = src.astype('int16')

    rows, cols = src.shape

    kernelRows, kernelCols = kernel.shape

    padWidth = (int)(kernelRows // 2)

    padImg = cv2.copyMakeBorder(src, padWidth, padWidth, padWidth, padWidth, cv2.BORDER_CONSTANT, None, 255)

    sumVal = 0

    halfKernelRows = (int)(kernelRows // 2)
    halfKernelCols = (int)(kernelCols // 2)

    for i in range(rows):
        for j in range(cols):
            currMatrix = padImg[(padWidth + i - halfKernelRows):(padWidth + i + halfKernelRows + 1),(padWidth + j - halfKernelCols):(padWidth + j + halfKernelCols + 1)]
            currMatrix = kernel * currMatrix
            outputArr[i][j] = np.sum(currMatrix)
    
    sumKernel = np.sum(kernel)

    if sumKernel > 1:
        outputArr = outputArr / sumKernel
    
    return outputArr


def localMax(src):
    """
    Given an array, it finds the local maxima in the x direction
    and y direction.
    """
    height, width = src.shape

    for k in range(height):
        for l in range(width - 2):
            if(src[k][l + 1] > src[k][l]):
                if(src[k][l + 1] > src[k][l + 2]):
                    src[k][l] = 0
                    src[k][l + 2] = 0

    
    for m in range(width):
        for n in range(height - 2):
            if(src[n + 1][m] > src[n][m]):
                if(src[n + 1][m] > src[n + 2][m]):
                    src[n][m] = 0
                    src[n + 2][m] = 0

    return src

def finalDetection(strong, weak):
    """
    Promotes weak edges to strong edges if a weak edge
    is connected to a strong edge.
    """
    finalHeight, finalWidth = weak.shape

    for o in range(finalHeight):
        for p in range(finalWidth):
            if(weak[o][p] == 75):
                try:
                    if(strong[o + 1][p - 1] == 150 or strong[o][p - 1] == 150 or strong[o - 1][p - 1] == 150 
                    or strong[o + 1][p] == 150 or strong[o - 1][p] == 150
                    or strong[o + 1][p + 1] == 150 or strong[o][p + 1] == 150 or strong[o - 1][p + 1] == 150):
                        strong[o][p] = 150
                except IndexError:
                    pass

    return strong


if __name__ == "__main__":

    # Select the image to open
    filename = askopenfilename()

    # Create a threshold for weak and strong edges. This
    # value should range between zero and 255. I have 
    # selected some arbitrary values. You are welcome to 
    # change them, as these are dependent on the input image
    strongthreshold = 150
    weakthreshold = 100

    # Open the image in grayscale
    src = cv2.imread(filename,0);
    cv2.imshow("Original Grayscale Image", src)
    cv2.waitKey(0)

    # Begin by smoothing the image using a Gaussian blur.
    # You can use your convolution function to do so or call
    # the OpenCV function.
    gaussianKernel = np.array([[1,2,1],[2,4,2],[1,2,1]])

    smoothImg = convolution(src, gaussianKernel)

    cv2.imshow('Smoothed Image', smoothImg.astype('uint8'))
    cv2.waitKey(0)

    # Next, find the gradients in the x and y directions. You
    # will have two separate output arrays. Use the Sobel
    # kernels to perform two convolutions, one for horizontal
    # and the other for vertical gradients.
    xSobelKernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    ySobelKernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    xGradient = convolution(src, xSobelKernel)
    yGradient = convolution(src, ySobelKernel)
    
    # Display the output of the two convolutions using imshow
    # Keep in mind that your resulting array with have values
    # that are between -4*255 and 4*255. Pixels with no change
    # in horiz or vertical gradient should appear as gray (127)
    cv2.imshow('Horizontal Gradient Image', xGradient.astype('uint8'))
    cv2.waitKey(0)

    cv2.imshow('Vertical Gradient Image', yGradient.astype('uint8'))
    cv2.waitKey(0)

    # Compute the gradient matrix by taking the square root of
    # the sum of the squares of the gradient matrices.
    gradientMatrix = np.hypot(xGradient, yGradient)

    cv2.imshow('Gradient Matrix Image', gradientMatrix.astype('uint8'))
    cv2.waitKey(0)

    # Compute non-maximum suppression for the single gradient
    # array. 
    nonMaxImg = localMax(gradientMatrix)

    cv2.imshow('Non-Maximum Suppression Image', nonMaxImg.astype('uint8'))
    cv2.waitKey(0)

    # Create two arrays for strong and weak edges. In the strong
    # edge image, any values that are above the strong threshold
    # are considered strong. Weak edges are edges that are above
    # the weak threshold but below the strong edge threshold
    height, width = nonMaxImg.shape

    strongImg = np.zeros((height,width), dtype=np.int16)
    weakImg = np.zeros((height,width), dtype=np.int16)
    
    strongX, strongY = np.where(nonMaxImg >= 150)
    strongImg[strongX, strongY] = 150
    cv2.imshow('Strong Edges Image', strongImg.astype('uint8'))
    cv2.waitKey(0)
    
    weakX, weakY = np.where((nonMaxImg <= 150) & (nonMaxImg >= 75))
    weakImg[weakX, weakY] = 75
    cv2.imshow('Weak Edges Image', weakImg.astype('uint8'))
    cv2.waitKey(0)
    
    # Final detection. Any weak edge that touches a strong edge is
    # promoted to strong edge. that are Combine Weak and Strong Edges
    finalImg = finalDetection(strongImg, weakImg)

    cv2.imshow('Final Detection Image', finalImg.astype('uint8'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
