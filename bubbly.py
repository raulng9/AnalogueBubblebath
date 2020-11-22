import cv2
import argparse
import numpy as np
from imutils import contours
from imutils.perspective import four_point_transform
import imutils
import random

inputFromWebcam = cv2.VideoCapture(0);

#primero sort vertical y luego horizontal, luego se splittea en los grupos
#equivalentes al número de preguntas por fila
def sortAnswers(contoursOfAnswers):
    questionsSortedVertical = contours.sort_contours(contoursOfAnswers, method="top-to-bottom")[0]
    #print(len(questionsSortedVertical))
    questionsSortedHorizontal = None
    for(q, i) in enumerate(np.arange(0, len(questionsSortedVertical), 3)):
        questionsSortedHorizontal = contours.sort_contours(questionsSortedVertical[i:i + 3])[0]
        filledIn = None
    print(len(questionsSortedHorizontal))
    return questionsSortedHorizontal

def gradeAndDraw(questionsContours):
    for(j,contour) in enumerate(questionsContours):
        mask = np.zeros(thresholdFrame.shape, dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)
        cv2.imshow("mascara", mask)

        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        totalNonZero = cv2.countNonZero(mask)

        if filledIn is None or totalNonZero > filledIn[0]:
            filledIn = (totalNonZero,j)

def findTestCircles(threshFrame):
    bubbleContours = cv2.findContours(threshFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbleContours = imutils.grab_contours(bubbleContours)
    bubbleContoursFiltered = []

    for contour in bubbleContours:
        (x,y,width,height) = cv2.boundingRect(contour)
        aspectRatio = width/float(height)

        if width >= 20 and height >= 20 and aspectRatio >= 0.9 and aspectRatio <= 1.3:
            bubbleContoursFiltered.append(contour)

    return bubbleContoursFiltered;




def splitListInGroups(listToSplit, sizeOfGroups):
    finalList = []
    for i in range(0, len(listToSplit), sizeOfGroups):
        yield listToSplit[i:i + sizeOfGroups]




#actual work
while(True):
        ret, frame = inputFromWebcam.read()
        frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
        #Image treatment until edges
        greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurFrame = cv2.GaussianBlur(greyFrame,(5,5),0)
        edgesFrame = cv2.Canny(blurFrame, 75,200)

        #Contour treatment
        contoursFirstPass = cv2.findContours(edgesFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contoursFirstPass = imutils.grab_contours(contoursFirstPass)

        contourOfTest = None
        contoursOfOptions = None
        transformedFrame = None

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
            transformedFrame = four_point_transform(greyFrame, contourOfTest.reshape(4,2))

            #We apply the threshold to obtain the circle contours
            thresholdFrame = cv2.threshold(transformedFrame, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            contoursOfOptions = findTestCircles(thresholdFrame)

        if contoursOfOptions is not None:
            if len(contoursOfOptions) == 9:
                #emptyFrameForRows = np.zeros((len(contourOfTest), len(contourOfTest[0])), np.uint8)
                questionsFullySorted = sortAnswers(contoursOfOptions)
                rows = list(splitListInGroups(contoursOfOptions,3))
                transformedFrameToColor = cv2.cvtColor(transformedFrame, cv2.COLOR_GRAY2RGB)
                for row in rows:
                    colorForRow = (random.randrange(0,255),random.randrange(0,255),random.randrange(0,255))
                    cv2.drawContours(transformedFrameToColor, row, -1, colorForRow, 2)

                #cv2.imshow("rows separated", transformedFrameToColor)
                gradeAndDraw(questionsFullySorted)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



inputFromWebcam.release()
cv2.destroyAllWindows()
