#importing the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
from cv2 import cv2
import os

#loading the face detector model from drive
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

#loading the mask detector model from drive
maskNet = load_model("mask_detector.model")

#initializing the video stream
print("[INFO] starting video stream...")
cap = cv2.VideoCapture("test.mp4")


def mask_detector(frame, faceNet, maskNet):

    #extracting the height and width of the image
    (h, w) = frame.shape[:2]

    #Creating blob from the image
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))

    #passing the blob through the network and obtaining the face
    faceNet.setInput(blob)
    detections = faceNet.forward()

    #Initialising the variables
    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        #extracting the confidence associated with the detection
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            #computing the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            #Ensuring the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            #Extracting the Region of Interest (Face)
            face = frame[startY:endY, startX:endX]
            #Converting the image from BGR to RGB
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            #Resizing the image to standard size (224x224)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            #Adding faces and bounding boxes to the list
            faces.append(face)
            locs.append((startX, startY, endX, endY))
            
    #Prediction is made only if there is atleast one face in image
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)	
    return (locs, preds)

while True:
    #Obtaining a frame from the video stream
    ret,frame = cap.read()
    #Checking whether a frame is available or not
    if ret is True:
        #Resizing the frame to width of 400px
        frame = imutils.resize(frame, width=600)
        #To determine face location and mask prediction
        (locs, preds) = mask_detector(frame, faceNet, maskNet)
        for (box, pred) in zip(locs, preds):
            #Obtaining the bounding box Co-ordinates
            (startX, startY, endX, endY) = box
            #Predicting the label
            (Mask, NoMask) = pred
            if Mask > NoMask:
                label = "Mask"
                color = (0,255,0)
            else:
                label = "No Mask"
                color = (0,0,255)
            
            label = "{}: {:.2f}%".format(label, max(Mask, NoMask) * 100)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        cv2.imshow("Mask Detector", frame)
        key = cv2.waitKey(1)
        #Press Esc to stop the video
        if key == 27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

