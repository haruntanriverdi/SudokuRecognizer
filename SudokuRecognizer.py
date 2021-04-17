import cv2
import numpy as np

# ADDITIONAL LIBRARIES
from skimage.segmentation import clear_border
from tensorflow.python.keras.models import load_model
import operator
from matplotlib import pyplot as plt
import imutils

# Define your own functions, classes, etc. here:

def mnistPCA():

    return[]

def RecognizeSudoku(warp):

    #  An example of SudokuArray is given below :
    #   0 0 0 7 0 0 0 8 0
    #   0 9 0 0 0 3 1 0 0
    #   0 0 6 8 0 5 0 7 0
    #   0 2 0 6 0 0 0 4 9
    #   0 0 0 2 0 0 0 5 0
    #   0 0 8 0 4 0 0 0 7
    #   0 0 0 9 0 0 0 3 0
    #   3 7 0 0 0 0 0 0 6
    #   1 0 5 0 0 4 0 0 0
    # Where 0 represents empty cells in sudoku puzzle. SudokuArray must be a numpy array.

    # WRITE YOUR SUDOKU RECOGNIZER CODE HERE.
    # code code code
    # code code code
    # code code code
    # maybe more code

    SudokuArray = []

    stepX = warp.shape[1] // 9
    stepY = warp.shape[0] // 9

    model = load_model('model.hdf5')

    # ----- Loop over the sudoku grid locations ------
    for y in range(9):
        row = []

        for x in range(9):
            # ----- Compute the starting and ending (x, y)-coordinates of the current cell ------
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY

            # ----- Add the (x, y)-coordinates to our cell locations list ------
            row.append((startX, startY, endX, endY))

            # ----- Crop the cell from the warped transform image and then -----
            cell = warp[startY:endY, startX:endX]

            # ----- Convert sudoku cell to threshold image
            thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            thresh = clear_border(thresh)

            # ----- Find contours -----
            cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

            # cv2.imshow('cell', thresh)  # Display the image
            # cv2.waitKey(0)

            # ----- Calculate the area of contours ----
            area = 0
            for cnt in cnts:
                a = cv2.contourArea(cnt)
                area = area + a

            # ----- If area lower than 100, cell is empty. Define 0 -----
            if area < 100:
                SudokuArray.append(0)

            # ----- Else, resize MNIST dataset and predict the number in the cell -----
            else:

                thresh = cv2.resize(thresh, (28, 28))
                # display_image(image)
                thresh = thresh.astype('float32')
                thresh = thresh.reshape(1, 28, 28, 1)
                thresh /= 255

                #plt.imshow(thresh.reshape(28, 28), cmap='Greys')
                #plt.show()

                pred = model.predict(thresh.reshape(1, 28, 28, 1), batch_size=1)
                number = pred.argmax()

                SudokuArray.append(number)

    SudokuArray = np.array(SudokuArray)
    SudokuArray = SudokuArray.reshape(9,9)

    #print(SudokuArray)

    return SudokuArray



# Returns Sudoku puzzle bounding box following format [(topLeftx, topLefty), (bottomRightx, bottomRighty)]
# (You can change the body of this function anyway you like.)
def detectSudoku(img):

    # ----- OriginalImage -> Grayscale -> Blur -> Threshold -----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray.copy(), (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh, thresh)

    # ----- Find contours and sort by area, descending -----
    contours, h = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # ----- Get largest contour -----
    polygon = contours[0]

    # ----- operator.itemgetter - get index of point -----
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    # ----- get corner points of sudoku and draw circle -----
    points = [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]
    for point in points:
        img = cv2.circle(img, tuple(int(x) for x in point), 5, (0, 0, 255), -1)

    # ----- Scalar distance between a and b -----
    def distance_between(a, b):
        # sqrt(x^2 + y^2)      where (x -> ====) and (y -> ++++)
        return np.sqrt(((b[0] - a[0]) ** 2) + ((b[1] - a[1]) ** 2))

    # ----- Crop rectangular portion from image and wraps it into a square of similar size -----
    top_left, top_right, bottom_right, bottom_left = points[0], points[1], points[2], points[3]

    # ----- Float for perspective transformation -----
    source_rect = np.array(np.array([top_left, bottom_left, bottom_right, top_right],
                                    dtype='float32'))

    # ----- Get longest side in rectangle -----
    side = max([
        distance_between(bottom_right, top_right),
        distance_between(top_left, bottom_left),
        distance_between(bottom_right, bottom_left),
        distance_between(top_left, top_right)
    ])
    dest_square = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

    # ----- Skew the image by comparing 4 before and after points -----
    m = cv2.getPerspectiveTransform(source_rect, dest_square)

    # ----- Perspective Transformation on original grayscale image -----
    warp = cv2.warpPerspective(gray, m, (int(side), int(side)))

    # cv2.imshow('image2', warp)  # Display the image
    # cv2.waitKey(0)

    return warp






