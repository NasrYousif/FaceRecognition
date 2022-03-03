import os
import time
import cv2
from imutils.video import VideoStream
import imutils


def make_dataset(cascadpath, output_dir):
    """
    This function make a custom dataset for face recognition using your webcam

    Parameters
    ===========
    cascadpath : srting
             Path into the xml file of your algorithm
    output_dir : string
             Path where to save your camera shots
    """
    detector = cv2.CascadeClassifier(cascadpath)

    # start camera streaming
    print("Your Camera is live...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    total = 0

    # While camera is start do the next
    while True:
        # step1: construct a frame with size 600 x 600
        frame = vs.read()
        orig = frame.copy()
        frame = imutils.resize(frame, width=600)

        # Step 2: detect the faces in the frame
        faces = detector.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30))

      # step 3: loop over faces
        for (img_x, img_y, f_width, f_hiegh) in faces:
           # cv2.rectangle(image, start_point, end_point, color, thickness)
            cv2.rectangle(frame, (img_x, img_y),
                          (img_x + f_width, img_y + f_hiegh), (0, 255, 0), 2)

        # step 4: output the farm windows and start shots with ("K") or exit with ("Q")
        cv2.imshow("Image", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("k"):
            p = os.path.sep.join([output_dir, "img.1.{}.png".format(
                str(total))])
            cv2.imwrite(p, orig)
            total += 1

        elif key == ord("q"):
            break


cascadpath = 'face_detection_model/haarcascade_frontalface_default.xml'
output_dir = 'dataset/'
make_dataset(cascadpath, output_dir)
