import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Load your trained model and label file
classifier = Classifier(
r"C:\\Users\\ADMIN\\Downloads\\converted_keras\\keras_model.h5",
 r"C:\\Users\\ADMIN\\Downloads\\converted_keras\\labels.txt"
)

offset = 20
imgSize = 300
counter = 0

# Optional: label list
labels = ["Hello","house","ok","stop", "Thank you", "Yes"]

while True:
    success, img = cap.read()
    if not success:
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Get prediction
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        print(prediction, index)

        # Draw bounding boxes and label
        cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x-offset+400, y-offset+10), (0,255,0), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (0,255,0), 4)

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)

    # Press 'q' to quit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
