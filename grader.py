import cv2
import argparse
import numpy as np
from imutils import contours
from imutils.perspective import four_point_transform
import imutils
import random
import os
import sys
import math
import pytesseract
from pytesseract import Output
import data_retriever
from data_retriever import Student
from data_retriever import retrieve_students
inputFromWebcam = cv2.VideoCapture(0);

#global variables
answers = {}
questionsPerRow = None
listOfStudents = []
currentStudent = ""
averageForCurrentStudent = 0
studentImageForDisplay = None

answersFrame = None
currentCorrectAnswers = None

keepScanning = True

#primero sort vertical y luego horizontal, luego se splittea en los grupos
#equivalentes al número de preguntas por fila
def sort_and_grade_answers(contoursOfAnswers, referenceFrame, originalFrame):
    global currentCorrectAnswers
    questionsSortedVertical = contours.sort_contours(contoursOfAnswers, method="top-to-bottom")[0]
    correctAnswers = 0
    originalFrameForMask = cv2.cvtColor(referenceFrame, cv2.COLOR_BGR2GRAY)
    originalFrameForMask = cv2.threshold(originalFrameForMask, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    for(q, i) in enumerate(np.arange(0, len(questionsSortedVertical), 3)):
        questionsSortedHorizontal = contours.sort_contours(questionsSortedVertical[i:i + 3])[0]
        filledIn = None
        contoursFilled = []
        contourForIteration = None
        for(j,contour) in enumerate(questionsSortedHorizontal):
            mask = np.zeros(originalFrameForMask.shape, dtype="uint8")
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mask = cv2.bitwise_and(originalFrameForMask, originalFrameForMask, mask=mask)
            totalNonZero = cv2.countNonZero(mask)
            if filledIn is None or totalNonZero > filledIn[0]:
                filledIn = (totalNonZero,j)
                contourForIteration = contour

        contoursFilled.append(contourForIteration)
        color = (0, 0, 255)
        k = answers[q]
	    # check to see if the bubbled answer is correct
        if k == filledIn[1]:
            color = (0, 255, 0)
            correctAnswers += 1
	    # draw the outline of the correct answer on the test
        cv2.drawContours(originalFrame, contoursFilled, -1, color, 3)

    currentCorrectAnswers = correctAnswers
    show_exam_information(correctAnswers, originalFrame)
    print("Test graded")



def find_test_circles(threshFrame):
    bubbleContours = cv2.findContours(threshFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbleContours = imutils.grab_contours(bubbleContours)
    bubbleContoursFiltered = []

    for contour in bubbleContours:
        (x,y,width,height) = cv2.boundingRect(contour)
        aspectRatio = width/float(height)
        if width >= 20 and height >= 20 and aspectRatio >= 0.9 and aspectRatio <= 1.3:
            bubbleContoursFiltered.append(contour)

    return bubbleContoursFiltered;


def get_test_details(filename):
    testFile = open(filename, "r")
    testData = testFile.read()
    stringWithData = testData.split()
    testDataMapped = map(int, stringWithData)
    listOfMappedData = list(testDataMapped)
    questionsPerRow = listOfMappedData[0]
    for i in range(0,len(listOfMappedData)-1):
        answers[i]=listOfMappedData[i+1]
    print("Test answers loaded")

def truncate(number, decimals=0):
    if not isinstance(decimals, int):
        raise TypeError("needs to be an int")
    elif decimals < 0:
        raise ValueError("must be 0 or positive")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

def load_student_image():
    global studentImageForDisplay
    pathForStudent = "StudentPictures/" + currentStudent + ".jpg"
    studentImg = cv2.imread(pathForStudent, cv2.IMREAD_UNCHANGED)
    width = 100
    height = 100
    dim = (width, height)
    resizedImg = cv2.resize(studentImg, dim, interpolation = cv2.INTER_AREA)
    studentImageForDisplay = resizedImg

def display_student_image():
    x_offset=y_offset=10
    answersFrame[y_offset:y_offset+studentImageForDisplay.shape[0], x_offset:x_offset+studentImageForDisplay.shape[1]] = studentImageForDisplay
    cv2.destroyAllWindows()
    cv2.imshow("Results and student", answersFrame)

def show_exam_information(correctAnswers, finalFrame):
    global answersFrame
    if finalFrame is None:
        return
    finalScore = 0
    if correctAnswers != None:
        finalScore = correctAnswers/len(answers)*100
    testSheetHeight, testSheetWidth, testSheetChannel = finalFrame.shape
    cv2.putText(finalFrame, "Final score: " + "{:.2f}%".format(finalScore), (20,int(testSheetHeight)-200),
    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, bottomLeftOrigin=False)
    if currentStudent != "":
        cv2.putText(finalFrame, "Student: " + currentStudent + " Average: " + "{:.2f}".format(averageForCurrentStudent), (int(testSheetWidth/5)+30, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, bottomLeftOrigin=False)
        load_student_image()
        answersFrame = finalFrame
        display_student_image()
    else:
        cv2.destroyAllWindows()
        cv2.imshow("Test results", finalFrame)
    answersFrame = finalFrame


def find_name_contour(transformedFrame):
    if currentStudent != "" and answersFrame is not None:
        return
    #Image treatment until edges
    blurSheetFrame = cv2.GaussianBlur(transformedFrame,(5,5),0)
    edgesSheetFrame = cv2.Canny(blurSheetFrame, 75,200)
    #Contour treatment
    contoursSecondPass = cv2.findContours(edgesSheetFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contoursSecondPass = imutils.grab_contours(contoursSecondPass)

    contourOfName = None

    if len(contoursSecondPass) > 0:
        contoursSorted = sorted(contoursSecondPass, key=cv2.contourArea, reverse=True)
        for cont in contoursSorted:
            #Now we look for the biggest four corner polygon
            perimeter = cv2.arcLength(cont, True)
            approximatedPoly = cv2.approxPolyDP(cont, 0.02 * perimeter, True)
            if len(approximatedPoly) == 4:
                contourOfName = approximatedPoly
                break
        if contourOfName is not None:
            cv2.drawContours(transformedFrame, contourOfName, -1, (0,255,0),3)
            heightOfCropBR = transformedFrame.shape[1]
            widthOfCropBR = transformedFrame.shape[0]
            #we pull out the top left coordinate
            topLeftCoordName = contourOfName[1][0].tolist()
            bottomRightCoordName = contourOfName[3][0].tolist()

            topLeftExtra = [topLeftCoordName[0]+50, topLeftCoordName[1]]
            bottomRightExtra = [bottomRightCoordName[0]-30,bottomRightCoordName[1]]

            topLeftCoordNoName= contourOfName[2][0].tolist()
            bottomRightCoordNoName = [heightOfCropBR,widthOfCropBR]

            #the actual cropping done by slicing the frame from corner to corner (y coords go first (!))
            frameNameBox = transformedFrame[topLeftCoordName[1]:bottomRightCoordName[1], topLeftCoordName[0]:bottomRightCoordName[0]]
            frameNameBoxExtra = transformedFrame[topLeftExtra[1]:bottomRightExtra[1], topLeftExtra[0]:bottomRightExtra[0]]

            frameWithoutNameBox = transformedFrame[topLeftCoordNoName[1]:bottomRightCoordNoName[1], topLeftCoordNoName[0]:bottomRightCoordNoName[0]]

            if frameNameBoxExtra.shape[0] > 0 and frameNameBoxExtra.shape[1] > 0:
                custom_config_tesseract = r'--oem 3 --psm 6'
                nameFoundOCR = pytesseract.image_to_string(frameNameBoxExtra, config=custom_config_tesseract)
                alphaNameFound = ''.join(filter(str.isalpha, nameFoundOCR))
                check_for_student(alphaNameFound)


            if frameWithoutNameBox.shape[0] > 0 and frameWithoutNameBox.shape[1] > 0:
                frameWithoutNameBoxColor = frameWithoutNameBox.copy()
                frameWithoutNameBoxColorCopy = frameWithoutNameBoxColor.copy()
                frameWithoutNameBox = cv2.cvtColor(frameWithoutNameBox, cv2.COLOR_BGR2GRAY)
                #We apply the threshold to obtain the circle contours
                frameWithoutNameBox = cv2.threshold(frameWithoutNameBox, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                contoursOfOptions = find_test_circles(frameWithoutNameBox)
                if contoursOfOptions is not None:
                    if len(contoursOfOptions) == 9:
                        questionsFullySorted = sort_and_grade_answers(contoursOfOptions,frameWithoutNameBoxColor, frameWithoutNameBoxColorCopy)


def check_for_student(studentName):
    global currentStudent
    global averageForCurrentStudent
    for student in listOfStudents:
        if student.name == studentName:
            print("Student found")
            currentStudent = student.name
            averageForCurrentStudent = sum(student.listOfMarks)/len(student.listOfMarks)
            show_exam_information(currentCorrectAnswers, answersFrame)


#actual frame retrieval and management
def frame_scan():
    #infinite loop that retrieves the video frame
    while(keepScanning):
        ret, frame = inputFromWebcam.read()
        frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)

        contoursOfOptions = None
        transformedFrame = None

        find_name_contour(frame)

        #to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    global listOfStudents
    get_test_details(sys.argv[1])
    listOfStudents = retrieve_students()
    frame_scan()


if __name__ == "__main__":
    main()


inputFromWebcam.release()
cv2.destroyAllWindows()
