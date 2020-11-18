import cv2
import argparse
import numpy as np
from imutils import contours
from imutils.perspective import four_point_transform
import imutils

inputFromWebcam = cv2.VideoCapture(0);


def findTestCircles(threshFrame):
    bubbleContours = cv2.findContours(threshFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbleContours = imutils.grab_contours(bubbleContours)
    bubbleContoursFiltered = []

    for contour in bubbleContours:
        (x,y,width,height) = cv2.boundingRect(contour)
        aspectRatio = width/float(height)

        if aspectRatio >= 0.9 and aspectRatio <= 1.1:
            bubbleContoursFiltered.append(contour)
    print(len(bubbleContoursFiltered))
    cv2.drawContours(threshFrame, bubbleContours, -1, (0,255,0),3)
    cv2.imshow("th", threshFrame)



#frame loop
while(True):
        ret, frame = inputFromWebcam.read()

        #Image treatment until edges
        greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurFrame = cv2.GaussianBlur(greyFrame,(5,5),0)
        edgesFrame = cv2.Canny(blurFrame, 75,200)

        #Contour treatment
        contoursFirstPass = cv2.findContours(edgesFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contoursFirstPass = imutils.grab_contours(contoursFirstPass)

        contourOfTest = None

        if len(contoursFirstPass) > 0:
            contoursSorted = sorted(contoursFirstPass, key=cv2.contourArea, reverse=True)
            for cont in contoursSorted:
                #Now we look for the biggest four corner polygon
                perimeter = cv2.arcLength(cont, True)
                approximatedPoly = cv2.approxPolyDP(cont, 0.02 * perimeter, True)
                if len(approximatedPoly) == 4:
                    contourOfTest = approximatedPoly
                    break

        if contourOfTest is not None:
            cv2.drawContours(frame, contourOfTest, -1, (0,255,0),3)
            #Perspective transform to isolate the test document
            testFrame = four_point_transform(frame, contourOfTest.reshape(4,2))
            transformed = four_point_transform(greyFrame, contourOfTest.reshape(4,2))

            #We apply the threshold to obtain the circle contours
            thresholdFrame = cv2.threshold(transformed, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
             #cv2.imshow("thresh", thresholdFrame)

            findTestCircles(thresholdFrame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break






inputFromWebcam.release()
cv2.destroyAllWindows()
