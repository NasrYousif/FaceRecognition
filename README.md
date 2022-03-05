# Face Recognition using Custom dataset
A simple implementation of cascadeClassifier of OpenCV algorithm to build real-timr face recognition system
![faceRecognition](https://user-images.githubusercontent.com/44967072/156584360-0e6e1ebb-9e87-49ec-8949-894c73613ee0.gif)
## Steps to build your custom system
### 1. Make a custom dataset.
Run <code>make_dataset.py</code> to open your camera and make shots by preesing the key <code> "k" </code> for every shot for every user face that you want to recognize, after finish press <code> "q" </code> to close the window.\n
### 2. Train your model
Open <code>train_model.py</code> and run it to generate your <code>trainer.yml</code> file contain 128-D array to reconize your faces.
### 3. Check your realtime prediction
Run <code>recognizer.py</code> and look at the camera to see if your face recognized well, 

## References
- [OpenCV Library](https://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0)
