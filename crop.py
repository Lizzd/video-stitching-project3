import cv2
import imutils
import numpy as np

def remove_border(stitched_img):
    stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))
    
    # create thresholded binary image
    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray, 0, 255 , cv2.THRESH_BINARY)[1]
    cv2.imshow("Threshold Image", thresh_img)
    cv2.waitKey(0)

    # find the contour of thresh_img
    contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # get the area of interst
    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)
    cv2.imshow("first areaOI", thresh_img)
    cv2.waitKey(0)
    # get the bounding rect of areaOI and create a mask for thresh_img
    mask = np.zeros(thresh_img.shape, dtype="uint8")
    x, y, w, h = cv2.boundingRect(areaOI)
    cv2.rectangle(mask, (x,y), (x + w, y + h), 255, -1)

    # loop over to find the max minimum rect
    minRectangle = mask.copy()
    sub = mask.copy()
    cv2.imshow("first minRectangle", minRectangle)
    cv2.waitKey(0)
    while cv2.countNonZero(sub) > 20:
        minRectangle = cv2.erode(minRectangle, None)
        cv2.imshow("eroded loop minRectangle", minRectangle)
        cv2.waitKey(0)
        sub = cv2.subtract(minRectangle, thresh_img)
        print("non zero count: ", cv2.countNonZero(sub))
        cv2.imshow("sub res", sub)
        cv2.waitKey(0)

    # cv2.imshow("minRectangle after erosion", minRectangle)

    # get the final result
    # contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # contours = imutils.grab_contours(contours)
    # areaOI = max(contours, key=cv2.contourArea)
    minRectangle = minRectangle[10:minRectangle.shape[0] - 10, 10: minRectangle.shape[1] - 10]

    cv2.imshow("minRectangle Image", minRectangle)
    cv2.waitKey(0)

    return minRectangle

