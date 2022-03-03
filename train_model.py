import cv2
import numpy as np
from PIL import Image
import os


# Path for face image database
path = 'dataset/'
cascadePath = "face_detection_model/haarcascade_frontalface_default.xml"
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cascadePath)


def findImageLabel(path):
    """
    This function extract the labels from the image name.
    example: 
            image name: img.1.0
            1 = the user id
            0 = face id
    """
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    return faceSamples, ids


print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = findImageLabel(path)
recognizer.train(faces, np.array(ids))
# Save the model intoroot directory trainer.yml
recognizer.write('trainer.yml')

# save the model into output directory
recognizer.save("output/serialized_recognizer.cv2")
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(
    len(np.unique(ids))))
