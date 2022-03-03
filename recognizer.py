import cv2
import numpy as np
import os


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
cascadePath = "face_detection_model/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX


camera = cv2.VideoCapture(0)
camera.set(3, 640)  # set video widht
camera.set(4, 480)  # set video height
# min window size to be recognized as a face
minW = 0.1*camera.get(3)
minH = 0.1*camera.get(4)

# set counter for faces
id = 0
names = ['None', 'Nasreldeen']

# While camera is start do the next
while True:
    # step1: construct a frame
    ret, img = camera.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: detect the faces in the frame
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    # step 3: loop over faces
    for(cam_x, cam_y, f_width, f_hiegh) in faces:
        cv2.rectangle(img, (cam_x, cam_y), (cam_x+f_width,
                      cam_y+f_hiegh), (0, 255, 0), 2)
        id, confidence = recognizer.predict(
            gray[cam_y:cam_y+f_hiegh, cam_x:cam_x+f_width])

        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(
            img,
            str(id),
            (cam_x+5, cam_y-5),
            font,
            1,
            (255, 255, 255),
            2
        )
        cv2.putText(
            img,
            str(confidence),
            (cam_x+5, cam_y+f_hiegh-5),
            font,
            1,
            (255, 255, 0),
            1
        )

    cv2.imshow('camera', img)
    # Press 'ESC' for exiting video
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
# Do a bit of cleanup
print("\n Exiting Program...")
camera.release()
cv2.destroyAllWindows()
