import HandTrackingModule as htm
from tensorflow.keras.models import load_model
import WindowController as Win
import cv2
import numpy as np
import keyboard
import pygame
import time
import os

wc = Win.WindowControl()

buttonPressed = False
buttonCounter = 0
buttonDelay = 25

def gesture_recognize(lmList, modeValue):
    global buttonPressed

    if modeValue == "Win_control":
        labels = ['', 'Close', '', 'Resize', 'Open Chrome', 'Toggle', 'Maximize', '', '', '']
        model = load_model('Test.h5')
    elif modeValue == "Emoji":
        labels = ['okay','peace','thumbs up','thumbs down','call me','stop','rock','live long','fist', 'smile']
        model = load_model('Test.h5')

    # If landmarks are found, prepare the image for the CNN model
    if len(lmList) > 0 and buttonPressed is False:
        # Reshape the landmarks into a (1, 21, 2) array
        landmarks = np.array(lmList)[:21, 1:]
        landmarks = landmarks.reshape((1, 21, 2))

        # Use the CNN model to predict the class of the hand gesture
        prediction = model.predict(landmarks)
        class_index = np.argmax(prediction)
        class_label = labels[class_index]
        buttonPressed = True
        return class_label

def strt():
    global buttonPressed, buttonCounter, buttonDelay

    ############## Color Attributes ###############
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    GREEN = (0, 255, 0)
    BACKGROUND = (255, 255, 255)
    FORGROUND = (0, 255, 0)
    BORDER = (0, 255, 0)
    lastdrawColor = (0, 0, 1)
    drawColor = (0, 0, 255)
    BOUNDRYINC = 5

    ############## CV2 Attributes ###############
    cap = cv2.VideoCapture(0)
    width, height = 1280, 720
    cap.set(3, width)  # 640, 1280
    cap.set(4, height)  # 480, 720
    imgCanvas = np.zeros((height, width, 3), np.uint8)


    ############## PyGame Attributes ###############
    pygame.init()
    DISPLAYSURF = pygame.display.set_mode((width, height), flags=pygame.HIDDEN)
    pygame.display.set_caption("Digit Board")
    number_xcord = []
    number_ycord = []

    ############## Header Files Attributes ###############
    folderPath = "header"
    myList = os.listdir(folderPath)
    overlayList = []

    for imPath in myList:
        image = cv2.imread(f'{folderPath}/{imPath}')
        overlayList.append(image)
    header = overlayList[0]

    ############## Predication Model Attributes ###############
    label = ""
    PREDICT = "off"
    AlphaMODEL = load_model("Alpha_Model.h5")
    NumMODEL = load_model("bestmodel.h5")

    AlphaLABELS = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j',
                   10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't',
                   20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z', 26: ''}
    NumLABELS = {0: '0', 1: '1', 2: '2',
                 3: '3', 4: '4', 5: '5',
                 6: '6', 7: '7', 8: '8',
                 9: '9'}


    ############## HandDetection Attributes ###############
    detector = htm.handDetector(detectionCon=0.85)
    xp, yp = 0, 0
    brushThickness = 15
    modeValue = "OFF"
    modeColor = RED

    while True:
        SUCCESS, img = cap.read()
        img = cv2.flip(img, 1)

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        cv2.putText(img, "Press A for Alphabate Recognisition Mode ", (0, 145), 3, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "Press N for Digit Recognisition Mode ", (0, 162), 3, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "Press E for Emoji Recognition Mode ", (0, 179), 3, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "Press W for Window Control mode ", (0, 196), 3, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "Press O for Turn Off Recognisition Mode ", (0, 213), 3, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "Press Q for Quitting The Window", (0, 230), 3, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f'{"RECOGNISITION IS "}{modeValue}', (0, 247), 3, 0.5, modeColor, 1, cv2.LINE_AA)

        if keyboard.is_pressed('a'):
            if PREDICT != "alpha":
                PREDICT = "alpha"
                modeValue, modeColor = "ALPHABATES", GREEN

        if keyboard.is_pressed('n'):
            if PREDICT != "num":
                PREDICT = "num"
                modeValue, modeColor = "NUMBER", YELLOW

        if keyboard.is_pressed('o'):
            if PREDICT != "off":
                PREDICT = "off"
                modeValue, modeColor = "OFF", YELLOW

        if keyboard.is_pressed('e'):
            if PREDICT != "emoji":
                PREDICT = "emoji"
                modeValue, modeColor = "Emoji", YELLOW

        if keyboard.is_pressed('w'):
            if PREDICT != "win_control":
                PREDICT = "win_control"
                modeValue, modeColor = "Win_control", YELLOW

        if keyboard.is_pressed('q'):
            cap.release()
            cv2.destroyAllWindows()
            quit()

            xp, yp = 0, 0
            label = ""
            number_xcord = []
            number_ycord = []
            time.sleep(0.5)

        if PREDICT == 'win_control':
            con_label = gesture_recognize(lmList, modeValue)
            buttonPressed = True
            class_label = str(con_label)

            # Draw the predicted label on the image
            cv2.putText(img, class_label, (1000, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if con_label == 'Resize':
                time.sleep(1)
                wc.resize_window()
                time.sleep(1)
            elif con_label == 'Maximize':
                time.sleep(1)
                wc.maximize_window()
                time.sleep(1)
            elif con_label == 'Close':
                time.sleep(1)
                wc.close_window()
                time.sleep(1)
            elif con_label == 'Toggle':
                time.sleep(1)
                wc.toggle_window()
                time.sleep(1)
            elif con_label == 'Open Chrome':
                time.sleep(1)
                wc.open_window()
                time.sleep(1)

        if PREDICT == 'emoji':
                label_str = gesture_recognize(lmList, modeValue)
                buttonPressed = True
                class_label = str(label_str)
                emoji = {'okay': '👌', 'peace' : '✌',  'thumbs up' : '👍', 'thumbs down' : '👎', 'call me' :'🤙' , 'stop' : '✋', 'rock' : '🤘', 'live long' : '🖖',  'fist' : '✊', 'smile' : '😊'}
                if class_label != 'None':
                    fh = open('Output.txt', 'a', encoding='utf-8')
                    fh.write(emoji[class_label])
                    #print(emoji[class_label])
                    fh.close()
                # Draw the predicted label on the image
                cv2.putText(img, class_label, (1000, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        if buttonPressed:
            buttonCounter += 1
            if buttonCounter > buttonDelay:
                buttonCounter = 0
                buttonPressed = False

        if PREDICT == 'alpha' or PREDICT == 'num':
            if len(lmList) > 0:

                x1, y1 = lmList[8][1:]
                x2, y2 = lmList[12][1:]

                fingers = detector.fingersUp()

                #This part of the code checks for 2 finger selection
                if fingers[1] and fingers[2]:

                    # add

                    number_xcord = sorted(number_xcord)
                    number_ycord = sorted(number_ycord)

                    #Based on the coordinates it draws the bounding box
                    if (len(number_xcord) > 0 and len(number_ycord) > 0 and PREDICT != "off"):
                        if drawColor != (0, 0, 0) and lastdrawColor != (0, 0, 0):
                            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDRYINC, 0), min(width, number_xcord[
                                -1] + BOUNDRYINC)
                            rect_min_y, rect_max_y = max(0, number_ycord[0] - BOUNDRYINC), min(
                                number_ycord[-1] + BOUNDRYINC, height)
                            number_xcord = []
                            number_ycord = []

                            pygame.PixelArray(DISPLAYSURF) #creates a new image of bounding box
                            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x,
                                      rect_min_y:rect_max_y].T.astype(np.float32)

                            cv2.rectangle(imgCanvas, (rect_min_x, rect_min_y), (rect_max_x, rect_max_y), BORDER, 3)
                            image = cv2.resize(img_arr, (28, 28))
                            # cv2.imshow("Tmp",image)
                            image = np.pad(image, (10, 10), 'constant', constant_values=0)
                            image = cv2.resize(image, (28, 28)) / 255
                            # cv2.imshow("Tmp",image)

                            if PREDICT == "alpha":
                                label = str(AlphaLABELS[np.argmax(AlphaMODEL.predict(image.reshape(1, 28, 28, 1)))])
                            if PREDICT == "num":
                                label = str(NumLABELS[np.argmax(NumMODEL.predict(image.reshape(1, 28, 28, 1)))])
                            pygame.draw.rect(DISPLAYSURF, BLACK, (0, 0, width, height))

                            fh = open('Output.txt', 'a')
                            fh.write(label)
                            fh.close()

                            cv2.rectangle(imgCanvas, (rect_min_x + 50, rect_min_y - 30), (rect_min_x, rect_min_y),BACKGROUND, -1)
                            cv2.putText(imgCanvas, label, (rect_min_x, rect_min_y - 5), 3, 1, FORGROUND, 1, cv2.LINE_AA)
                        else:
                            number_xcord = []
                            number_ycord = []

                    #to select which color canvas
                    xp, yp = 0, 0
                    if y1 < 125:
                        lastdrawColor = drawColor
                        if 0 < x1 < 200:
                            imgCanvas = np.zeros((height, width, 3), np.uint8)
                        elif 210 < x1 < 320:
                            header = overlayList[0]
                            drawColor = (0, 0, 255)
                        elif 370 < x1 < 470:
                            header = overlayList[1]
                            drawColor = (0, 255, 255)
                        elif 520 < x1 < 630:
                            header = overlayList[2]
                            drawColor = (0, 255, 0)
                        elif 680 < x1 < 780:
                            header = overlayList[3]
                            drawColor = (255, 0, 0)
                        elif 890 < x1 < 1100:
                            header = overlayList[4]
                            drawColor = (0, 0, 0)
                            imgCanvas = np.zeros((height, width, 3), np.uint8)
                        elif 1160 < x1 < 1250:
                            cap.release()
                            cv2.destroyAllWindows()
                            quit()

                    cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

                #if only single fingle drawing is enabled
                elif fingers[1] and fingers[2] == False:

                    # add
                    number_xcord.append(x1)
                    number_ycord.append(y1)
                    # addEnd

                    cv2.circle(img, (x1, y1 - 15), 15, drawColor, cv2.FILLED)
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    if drawColor == (0, 0, 0):
                        pass
                    else:
                        cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                        pygame.draw.line(DISPLAYSURF, WHITE, (xp, yp), (x1, y1), brushThickness)
                    xp, yp = x1, y1
                else:
                    xp, yp = 0, 0

            #Coverting Gray image to BGR
            imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(img, imgInv)
            img = cv2.bitwise_or(img, imgCanvas)

        img[0:132, 0:1280] = header
        pygame.display.update()
        # cv2.imshow("Paint",imgCanvas)
        cv2.imshow("Input Window", img)
        cv2.waitKey(1)

strt()