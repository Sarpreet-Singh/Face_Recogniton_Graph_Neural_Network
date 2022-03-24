Face Recogniton using Graph Neural Network
=====================================
The project scans 1 face for recogniton, and as many faces as the user wants to compare it against.


Dependencies
--------------
* NumPy
* Dlib + Landmark Detector
* OpenCV
* PyTorch
* DGL

```bash
pip install numpy dlib opencv-python torch dgl
[Landmark Detector](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2)
```

Step 1: Scan Face
----------------------
```bash
Use WebCam or insert path of pre-registred video and specify 2 output directories.
1 directory the face that needs to be recognized and the other one the faces used for comparison.
```

Step 2: Create dataset 
----------------------
```bash
Specify the input directories to create the DGLGraphs
```

Step 3: Training & Evaluation
----------------------
```bash
Run with default config
```
