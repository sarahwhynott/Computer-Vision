# Filename: hw4Completed.py
# Author: Sarah Whynott
# Date Created: 9/29/22
#
# Hough Transformation for finding lines in an image

import cv2
import numpy as np
from tkinter.filedialog import askopenfilename
import time

def convolution(src, kernel):
    """
    Convolves an input image with the given kernel.
    Uses a reflective padding for convolution.
    """

    dst = np.zeros(src.shape,np.int32)

    rows = src.shape[0]
    cols = src.shape[1]
    klen = kernel.shape[0]
    padwid = klen//2

    padimg = np.pad(src, padwid, mode = 'reflect')

    for row in range(rows):
        for col in range(cols):
            
            dst[row,col] = np.sum(np.multiply(padimg[row:row+klen,col:col+klen],kernel))

    ksum = np.sum(kernel)
    return dst / ksum if ksum > 1 else dst 

def calcdirgrad(horiz, vert):
    """
    Calculates the approximate angle from the horizontal
    and vertical gradients.
    From BB: Using the arctan function, calculate the approximate angle in the image. 
    This will be in radians and you'll want to change it to degrees.
    """
    
    #decided not to implement - do not grade
    '''
    approxAngle = np.arc(horiz / vert)

    approxAngle = np.degrees(approxAngle)
    
    return approxAngle
    '''

def incrementAccumulator(src, H, offset):
    """
    Given an input image, this function increments the
    accumulator for each point in the source (edge) image.
    Because the value of d can be both positive and negative,
    it is critical to provide an offset value to shift the
    function so all values are positive. This same value
    must be subtracted from the accumulator when
    drawing the lines on the image.
    """

    # Setting every other pixel to zero to reduce the amount
    # of computation
    tmp = src.copy()
    tmp[::2,::2] = 0

    rows, cols = tmp.shape

    # Create a for-loop that iterates through each non-zero
    # pixel in the edge image.

        # Each point in the edge image is potentially a member of
        # a line in any direction. Thus, we must check every
        # direction. 
        # Begin by checking every direction, 180 degrees. If you
        # are able to complete the assignment checking all 180
        # degrees, modify this function so that it takes in another
        # argument, which is the array of angles calculated from
        # the gradient image. Use those angle to narrow your
        # search window instead of checking all angles.
        # Note: NumPy trigonometric functions usually expect
        # angles in radians. Thus, you will need to convert.
        # ***EXTRA CREDIT HERE ***

    for row in range(rows):
        for col in range(cols):
            if(tmp[row][col] != 0):
                for theta in range(180):
                    xcostheta = np.cos(np.pi * theta/180)
                    ysintheta = np.sin(np.pi * theta/180)
                    d = xcostheta + ysintheta + np.radians(offset)
                    d = int(d)
                    H[d, theta] += 1



def scaleArray(src, nmin=0, nmax=255):
    """
    Given an input array, this function scales they
    array between the minimum and maximum value, and
    returns an np.uint8 array, so that the array can be displayed
    as an image.
    """

    dst = src.copy().astype(np.float32)

    srcmin = np.amin(src)
    srcmax = np.amax(src)
    srcrange = srcmax - srcmin

    dst = 255 * (dst - srcmin) / srcrange

    return dst.astype(np.uint8)

def localMax(src):
    """
    This function finds local maxima within an input array. 
    A point is considered a local maxima if its value is greater
    than all of its neighbors. 
    """
    height, width = src.shape

    for i in range(height - 2):
        for j in range(width - 2):
            if(src[i + 1][j + 1] > src[i][j] and src[i + 1][j + 1] > src[i][j + 1] 
            and src[i + 1][j + 1] > src[i][j + 2] and src[i + 1][j + 1] > src[i + 1][j]
            and src[i + 1][j + 1] > src[i + 2][j] and src[i + 1][j + 1] > src[i + 2][j + 1]
            and src[i + 1][j + 1] > src[i + 1][j + 2]  and src[i + 1][j + 1] > src[i + 2][j + 2]):
                src[i][j] = 0
                src[i][j + 1]= 0
                src[i][j + 2] = 0
                src[i + 1][j] = 0
                src[i + 2][j] = 0
                src[i + 2][j + 1] = 0
                src[i + 1][j + 2] = 0
                src[i + 2][j + 2] = 0


    return src

def drawpolarlines(src, local_max_array, offset):
    """
    Draws the lines onton the output image
    """

    houghlines = src.copy()
    
    # Gets all of the nonzero values of the accumulator
    polarlines = np.transpose(np.nonzero(local_max_array))
    
    print("Drawing lines on source image.")
    for i in range(polarlines.shape[0]):

        theta, rho = polarlines[i,:]
        a = np.cos(np.pi * theta/180)
        b = np.sin(np.pi * theta/180)

        # Before the lines can be drawn, the offset from the 
        # accumulator must be substracted.
        x0 = int(a * (rho-offset))
        y0 = int(b * (rho-offset))

        # The value in offset is used here for a different
        # purpose. Lines drawn on an image need two points,
        # a starting and ending point. In order to make sure
        # the line sufficiently goes from one side of the image
        # the other, we need to guarantee that the points choosen
        # will spand the entire image. The value in offset is
        # large enough to guarantee that will happen.
        x1 = int(x0 - b * offset)
        y1 = int(y0 + a * offset)
        x2 = int(x0 + b * offset)
        y2 = int(y0 - a * offset)

        cv2.line(houghlines,(x1,y1),(x2,y2), (0,255,0),2)

    return houghlines

def main():
    """
    Finds lines in an image using the Hough Transformation.
    """
    
    # Reads in the file and coverts it to grayscale. This
    # program expects the input file to be a
    filename = askopenfilename()
    src = cv2.imread(filename)
    gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    time.sleep(1)
    
    # Create all of the windows that will be needed for this assignment.
    cv2.namedWindow("Source",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Edge Image",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Hough Lines",cv2.WINDOW_NORMAL)

    cv2.waitKey(10)

    # Create a Canny edge image from the input image
    print("Creating Edge Image")
    edgeimg = cv2.Canny(gray,100,150)

    #Next, you'll calculate the horizontal and vertical gradients using a Sobel kernel. 
    #A convolution function has been included in this file. 

    #decided not to implement - do not grade
    '''
    xSobelKernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    ySobelKernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    xGradient = convolution(edgeimg, xSobelKernel) #vertical lines
    yGradient = convolution(edgeimg, ySobelKernel) #horiz lines

    dirgrad = calcdirgrad(xGradient, yGradient)
    '''

    # Calculate the maximum length a line could be in the image
    maxrho = int(np.ceil(max(edgeimg.shape[0],edgeimg.shape[1]) * 1.414))

    # Create the accumulator array. 
    H = np.zeros((180,int(2*maxrho)),np.int32)

    # For each point in the edge image, increment the accumulator.
    # You will implement this function.
    incrementAccumulator(edgeimg, H, maxrho)

    # Find the local maximas in the accumulator. 
    # You will implement this function
    local_max_array = localMax(H)
    
    # There will be some noise in the local max array. This will
    # cause lots of erroneous lines to be drawn on the image. Normalize
    # the local max array from 0 to 255. Set a threshold, such as 30 or 
    # 50 and suppress any pixels with values below this threshold. These
    # are pixels that were local max but had few votes. 
    # You can save that back into the same array local_max_array
    normalizeArray = scaleArray(local_max_array, 0, 255)

    threshold = 30

    belowThreshold = np.where(normalizeArray < threshold)

    normalizeArray[belowThreshold] = 0

    local_max_array = normalizeArray

    # Lastly, pass your local_max_array to this drawing function.
    # This will draw the lines on a new image, and return the array.
    houghlines = drawpolarlines(src, local_max_array, maxrho)

    # Now, show off all of your hard work!
    cv2.imshow("Source", src)
    cv2.imshow("Edge Image", edgeimg)
    cv2.imshow("Hough Lines",houghlines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
