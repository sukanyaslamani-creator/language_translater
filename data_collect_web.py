from flask import Flask, render_template, request
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import threading

app = Flask(__name__)

# Load your trained model and labels
classifier = Classifier(
   r"C:\\Users\\ADMIN\\Downloads\\converted_keras\\keras_model.h5",
   r"C:\\Users\\ADMIN\\Downloads\\converted_keras\\labels.txt"

   
)

labels = ["Hello","house","ok","stop", "Thank you", "Yes"]  # Replace/add your actual labels

offset = 20
imgSize = 300

def run_detection(username):
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)

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

            # Prevent crash if crop is empty
            if imgCrop.size == 0:
                continue

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize

            # Get prediction
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            predicted_text = labels[index]

            # Draw on image
            cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x-offset+400, y-offset+10), (0,255,0), cv2.FILLED)
            cv2.putText(imgOutput, predicted_text, (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (0,255,0), 4)

        cv2.imshow(f"Sign Language Detection - {username}", imgOutput)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    username = request.form['username']
    # Run detection in a separate thread
    threading.Thread(target=run_detection, args=(username,)).start()
    return f"Detection started for {username}! Close the OpenCV window to stop."

if __name__ == "__main__":
    app.run(debug=True)
