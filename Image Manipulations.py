# Name: Sarah Whynott
# Date: 9/1/2022
# Assignment: Homework 1
# 
# Villanova CSC5930/9010 Computer Vision
#
# Image Manipulation - rotations, transformations, etc.

import cv2
import numpy as np

from tkinter.filedialog import askopenfilename

def split_image(src):
    """
    Given a three channel image, this function splits the image
    into blue, green, and red and returns each channel separately
    """
    return src[:, :, 2], src[:, :, 1],  src[:, :, 0]

def clip(src):
    """
    Sets any value in the array over 255 to 255 and any value less
    than 0 to zero.
    """
    src[src > 255] = 255
    src[src < 0] = 0

def brighten(src, val):
    """
    Brightens the image by uniformly increasing each pixel intensity
    by the given value
    """
    brightenImg = src.astype('int16')

    brightenImg = brightenImg + val

    clip(brightenImg)

    return brightenImg.astype('uint8')

def darken(src, val):
    """
    Darkens the image by uniformly increasing each pixel intensity
    by the given value
    """
    darkenImg = src.astype('int16')

    darkenImg = darkenImg - val

    clip(darkenImg)

    return darkenImg.astype('uint8')

def normalize(src):
    """
    Stretching the range of the image so that the minimal value is
    zero and the maximum value is 255
    """
    normalizeImg = src.astype('int16')

    normalizeImg = (src - src.min())/((src.max() - src.min())/255.0)

    clip(normalizeImg)

    return normalizeImg.astype('uint8')

def pad(src, width, val=0):
    """
    Pads the image with a constant value given the specified width
    """
    origHeight, origWidth, channels = src.shape

    newHeight = origHeight + (width * 2)
    newWidth = origWidth + (width * 2)

    padImg = np.full( (newHeight, newWidth, channels), val, dtype = 'uint8')

    padImg[width:newHeight - width, width:newWidth - width,:] = src

    return padImg

def clockwise(src):
    """
    Rotates the image clockwise 90 degrees
    """
    clockwiseImg = np.rot90(src, 3)

    return clockwiseImg

def cclockwise(src):
    """
    Rotates the image counter-clockwise 90 degrees
    """
    cclockwiseImg = np.rot90(src)

    return cclockwiseImg

def quadrants(src):
    """
    Splits the image into four regions and returns them
    in the following order:
    top-left, top-right, bottom-left, bottom-right
    """
    height, width, channels = src.shape

    halfHeight = height // 2
    halfWidth = width // 2

    left = src[:, :halfWidth]           
    right = src[:, halfWidth:]  

    topLeft = left[:halfHeight, :]
    topRight = right[:halfHeight, :]
    bottomLeft = left[halfHeight:, :]
    bottomRight = right[halfHeight:, :]
 
    return topLeft, topRight, bottomLeft, bottomRight

def downscale(src):
    """
    Returns an array half the size by removing every other element
    from the rows and columns
    """
    downscaleImg = src[::2, ::2]

    return downscaleImg

def upscale(src):
    """
    Returns an array twice the input size by duplicating neighboring
    values in the rows and columns
    """
    upscaleImg = np.repeat(src, 2, axis = 0)
    upscaleImg = np.repeat(upscaleImg, 2, axis = 1)

    return upscaleImg

def grayscale(src, b, g, r):
    """
    Converts a 3 dimensional array (color image) to one-dimensional array
    (grayscale image) using the weights given for b, g, and r 
    b, g, and r are floating point values that should sum to 1.0
    """
    blue, green, red = cv2.split(src)

    blue = (blue / 255.0) * b
    green = (green / 255.0) * g
    red = (red / 255.0) * r

    grayscaleImg = blue + green + red

    return grayscaleImg

def main():
    """
    This program
    """

    # Select the image that you will be working with for this assignment.
    # You should select a 3-channel color image.
    filename = askopenfilename()

    # Read the image as-is (don't resize, convert to grayscale, etc.)
    # Save the image to a variable called src. This will be the source image.
    src = cv2.imread(filename)

    cv2.imshow("Original Image", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Implement the function split_image above to split the image into
    # blue, red, and green channels and display all three images
    blue, green, red = split_image(src)

    cv2.imshow("Blue Channel Image", blue)
    cv2.waitKey(0)

    cv2.imshow("Green Channel Image", green)
    cv2.waitKey(0)

    cv2.imshow("Red Channel Image", red)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Implement the function brighten image and display the resulting
    # image when increasing the images intensity by 20
    cv2.imshow("Brightened Image", brighten(src, 20))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Implement the function darken image and display the resulting
    # image when increasing the images intensity by 20
    cv2.imshow("Darkened Image", darken(src, 20))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Implement the function normalize and display the resulting
    # image. Call this function 3 times and show each image channel from
    # your input image
    cv2.imshow("Normalized Image", normalize(src))
    cv2.waitKey(0)

    blueNormal, greenNormal, redNormal = split_image(src)

    cv2.imshow("Blue Normalized Image", normalize(blueNormal))
    cv2.waitKey(0)

    cv2.imshow("Green Normalized Image", normalize(greenNormal))
    cv2.waitKey(0)

    cv2.imshow("Red Normalized Image", normalize(redNormal))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Implement the function pad to place a uniform constant value
    # around the border of the image. Display the resulting image.
    cv2.imshow("Padded Image", pad(src, 50))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Implement the function clockwise to rotate your image 90 degrees.
    # Display the resulting image.
    cv2.imshow("Clockwise Image", clockwise(src))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Implement the function cclockwise to rotate your image 90 degrees
    # counterclockwise. Display the resulting image.
    cv2.imshow("Counterclockwise Image", cclockwise(src))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Implement the function quadrants which divides the image equally
    # into four components. Display the resulting four images.
    topLeft, topRight, bottomLeft, bottomRight = quadrants(src)

    cv2.imshow("Top-Left Image", topLeft)
    cv2.waitKey(0)

    cv2.imshow("Top-Right Image", topRight)
    cv2.waitKey(0)

    cv2.imshow("Bottom-Left Image", bottomLeft)
    cv2.waitKey(0)

    cv2.imshow("Bottom-Right Image", bottomRight)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Implement the function downscale that shrinks the image by a factor
    # of two in the horizontal and vertical direction and display the
    # resulting image.
    cv2.imshow("Downscale Image", downscale(src))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Implement the function upscale that enlarges the image by a factor
    # of two in the horizontal and vertical direction and display the
    # resulting image.
    cv2.imshow("Upscale Image", upscale(src))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Implement the function grayscale that takes in a 3-channel image
    # and creates a grayscale image based upon the scaled values given.
    cv2.imshow("Grayscale Image", grayscale(src, .114, .587, .299))
    # values from https://github.com/opencv/opencv/blob/8c0b0714e76efef4a8ca2a7c410c60e55c5e9829/modules/imgproc/src/color.simd_helpers.hpp
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # You should display and close the image after each section.


if __name__ == "__main__":
    main()

