# Filename: hw3.py
# Author: Sarah Whynott
# Date Created: 9/17/2022
# Homework 3: Object Segmentation
#
# This assignment counts the number of shapes in the images

import cv2
import numpy as np
from tkinter.filedialog import askopenfilename

def conncomponents(src):
    """
    A two pass approach for finding connected components. Assigns a
    foreground pixel a new label if its left and upper neighbors
    are background pixels. Otherwise, the lower of the two values are
    assigned to it. The second pass recifies the pixels that have
    corresponding values.

    Input should be a binary image where black pixels are background
    and foreground pixels (objects) are white.
    """

    return src

if __name__ == "__main__":

    filename = askopenfilename()

    # Read in the image and convert the image to binary.
    src = cv2.imread(filename, 0)

    cv2.imshow('Grayscale Image', src)
    cv2.waitKey(0)

    threshold = 175

    rows, cols = src.shape

    # Threshold the image. Make sure the background pixels black.
    binaryImg = np.zeros((rows,cols))

    belowThreshold = np.where(src < threshold)

    binaryImg[belowThreshold] = 255

    cv2.imshow('Binary Image', binaryImg)
    cv2.waitKey(0)

    src = binaryImg

    label = 1.0

    Dict = {}

    # Run the connected components algorithm.
    for row in range(rows):
        for col in range(cols):
            if(src[row, col] != 0):
                left = src[row, col - 1]
                top = src[row - 1, col]
                if(left == 0 and top == 0):
                    src[row, col] = label
                    label = label + 1
                else:
                    if(left == 0):
                        src[row, col] = top
                    else:    
                        if(top == 0):
                            src[row, col] = left
                        else:
                            if(top == left):
                                src[row, col] = top
                            else:   
                                src[row, col] = min(top, left)
                                Dict[max(top, left)] = min(top,left)

    for key,value in Dict.items():
        while(value in Dict.keys()):
            value = Dict[value]
            Dict[key] = value

    Dict2 = {}
    newLabel = 0

    for row in range(rows):
        for col in range(cols):
            if(src[row, col] in Dict.keys()):
                src[row, col] = Dict[src[row, col]]
            if(src[row, col] not in Dict2.keys()):
                Dict2[src[row, col]] = newLabel
                newLabel = newLabel + 1


    # Print out the number of unique values from the connected
    # components algorithm. This is the number of shapes, including
    # the background.
    print("Number of shapes, including the background: ", newLabel)

    # Scale the shape labels (1,2,3,4...) between 0 and 255 so
    # that these values can been seen in a grayscale image.
    scaleInc = 255/(newLabel - 1)

    for key2, value2 in Dict2.items():
        Dict2[key2] = value2 * scaleInc

    for row in range(rows):
        for col in range(cols):
            src[row, col] = Dict2[src[row, col]]
            
    cv2.imshow('Scaled Image', src.astype('uint8'))
    cv2.waitKey(0)             
    
    # Bonus: Color the shapes (randomly)
