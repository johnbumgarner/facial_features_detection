# under construction

<p align="center">
  <img src="https://github.com/johnbumgarner/facial_detection_prediction-/blob/master/graphic/facial_recognition.png">
</p>

# Overview Facial Detection and Prediction

<p align="justify">

Most humans can look at two photos and quickly determine if the images are either similarity or dissimilarity in nature. Computers can be programmed to perform a similar task, but the results can vary, because of multiple factors(e.g., lighting conditions, perspectives) that humans can instinctively do automatically.  Humans have little difficulty seeing the subtle differents between a rose and a camellia or a gardenia and a rose. A computer on the other hand will recognize these objects as flowers, but would likely classify all of these images as a single type of flower, roses.   

There are numerous use cases for image similarities technologies. These use cases range from duplicate image detection to domain specific image clustering. Identifying duplicate images in Apple Photo is a common use case for many of us dealing with a large digital image library. Some of us have likey used Google’s Reverse Image Search to look for a specific photo that we want to know more about. Google will scour its massive database for images similar to the one used in your query. 

</p>

## Primary objective of this repository
<p align="justify">
This repository is going to examine several of the methods used to ascertain if two or more images have similarity or dissimilarity. The set of images used in these image simarility tests are publicly available photographs of well-known female actresses. One dataset has 12 images of these actresses wearing earrings. The different photos of Jennifer Aniston are within the first dataset.  The second dataset consists of 50 images of actresses with no duplicates.  The second set is more diversed, because it included mutiple skin tones and hair colors.   

Another objective of this repository is to determine the capabilities and limitations of the Python libraries used to perform these image simarility tests.
</p>

## Facial Detection and Features Identification

### Open Computer Vision Library (OpenCV):

<p align="justify">

This experiment used the CV2 modules <i>OpenCV-Python</i> and <i>OpenCV-Contrib-Python.</i>. These modules provide functions designed for real-time computer vision, image processing and machine learning. 

OpenCV is being used today for a wide range of applications which include:

- Automated inspection and surveillance
- Video/image search and retrieval
- Medical image analysis
- Criminal investigation
- Vehicle tag recognition
- Street view image stitching
- Robot and driver-less car navigation and control
- Signature pattern detection on documents

This experiment will focus on the basics of face detection using the Haar feature-based Cascade classifiers. The Haar Cascade classifiers used in this experiment were:

1. haarcascade_frontalface_default.xml
2. haarcascade_eye.xml
3. haarcascade_mcs_nose.xml
4. haarcascade_smile.xml

Additional Haar Cascade classifiers are available from these locations:

1. https://github.com/opencv/opencv/tree/master/data/haarcascades
2. http://alereimondo.no-ip.org/OpenCV/34.version?id=60

It's worth noting that <i>Python</i> occasionally has issues locating the Haar Cascade classifiers on your system.  To solve this you can use the <i>Python</i> module <i>os</i> to find the absolute paths for the classifiers installed.  
```python
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_frontal_face_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
haar_eye_model = os.path.join(cv2_base_dir, 'data/haarcascade_eye.xml')
haar_mouth_model = os.path.join(cv2_base_dir, 'data/haarcascade_smile.xml')
haar_nose_model = os.path.join(cv2_base_dir, 'data/haarcascade_mcs_nose.xml')
```

#### Haar Cascade - Facial Detection

One of the most basic Haar Cascade classifiers is the one used to detect the facial area of a human face looking directly at the camera. This base-level algorithm comes pretrained, so it is able to identify images that have human face characteristics and their associated parameters and one that have no human face characteristics.  


```python
# This code was extraction from mutiple functions in the script facial_detection_haar_cascade.py

image_name = 'natalie_portman.jpeg'
photograph = cv2.imread(image_name)
grayscale_image = cv2.cvtColor(photograph, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.3, minNeighbors=5)
for (x_coordinate, y_coordinate, width, height) in faces:
   cv2.rectangle(photograph, (x_coordinate, y_coordinate), (x_coordinate + width, y_coordinate + height), (255, 0, 255), 2)
```

The image of <i>Natalie Portman</i> below has a <i>bounding box</i> drawn around the entire facial area identified by the Haar Cascade classifier  <i>haarcascade_frontalface_default.xml.</i>

<p align="left">
  <img src="https://github.com/johnbumgarner/facial_detection_prediction-/blob/master/graphic/facial_front_detection.jpg">
</p>


#### Haar Cascade - Eye Detection

The eye area is another human face characteristic that can be identified using the Haar Cascade classifier <i>haarcascade_eye.xml</i>, which is used in collaboration with the Haar Cascade classifier for identifing frontal faces. The image of Natalie Portman below has a <i>bounding box</i> drawn around the eye area identified by the Haar Cascade classifier <i>haarcascade_eye.xml.</i>

<p align="left">
  <img src="https://github.com/johnbumgarner/facial_detection_prediction-/blob/master/graphic/eye_detection.jpg">
</p>


#### Haar Cascade - Nose Detection

The nose area classifier <i>haarcascade_mcs_nose.xml</i> also works in concert with the Haar Cascade classifier used to identify frontal faces. The image of Natalie Portman below has a <i>bounding box</i> drawn around the nose area identified by the Haar Cascade classifier <i>haarcascade_mcs_nose.xml.</i>

<p align="left">
  <img src="https://github.com/johnbumgarner/facial_detection_prediction-/blob/master/graphic/nose_detection.jpg">
</p>

#### Haar Cascade - Mouth Detection

This classifier uses the Haar Cascade <i>haarcascade_smile.xml</i> in conjunction with the Haar Cascade classifier used to identify frontal faces.

The image of Natalie Portman below has a <i>bounding box</i> drawn around the mouth area identified by the Haar Cascade classifier <i>haarcascade_smile.xml.</i>

<p align="left">
  <img src="https://github.com/johnbumgarner/facial_detection_prediction-/blob/master/graphic/mouth_detection.jpg">
</p>

#### Haar Cascade - All Facial Characteristics Detection

The image of Natalie Portman below has <i>bounding boxes</i> drawn around the all the facial characteristics previously identified by all the Haar Cascade classifiers above.

<p align="left">
  <img src="https://github.com/johnbumgarner/facial_detection_prediction-/blob/master/graphic/all_features_detection.jpg">
</p>






#### ROUGH FROM OLD REPO


<p align="justify">
The LBPH face recognition algorithm was used in this experiment. Local Binary Pattern (LBP) is a simple and efficient texture operator which labels the pixels of an image by thresholding the neighborhood of each pixel and considers the result as a binary number. The experiment also uses the Haar Cascade, which is a machine learning object detection algorithm used to identify objects in an image or video and based on the concept of features.


During testing it was noted that all these Haar Cascade classifiers were temperamental and required continually tuning of the parameters scaleFactor and minNeighbors used by detectMultiScale.  The angle of the faces within the images were also a key factor when detecting facial features. Images containing direct frontal faces produced the best results as shown below.
</p>

### Experiment 2:

<p align="justify">
This experiment also uses the Python module cv2 and the LBPH face recognition algorithm.  The previous experiment was designed to do some base level facial prediction. In this experiment a set of 74 images of female actresses was used to create the facial prediction training data, the associated labels and identification numbers. 
</p>

<p align="justify">
The predict feature of LBPHFaceRecognizer was utilized in this experiment. The prediction of the LBPH face recognition algorithm will analyze an image containing face and attempt to identify the face contain in the image against images within the training data. If a probable match is found within the training data then the person name and the confidence score will be overlaid on the image displayed as shown below:
</p>

<p align="center"><br>
<img src="https://github.com/johnbumgarner/image_similarity_experiments/blob/master/aishwarya_rai_confidence_score.jpg">
</p>

<p align="justify">
The algorithm was able to successfully predict that an image contained in the training data matched an image that was used in creating the training data. A confidence_score of '0' is a 100% match. The success rate was less accurate for images not used in the training data. 
</p>

### Experiment 3:

<p align="justify">
Like the previous experiment this one also uses the Python module cv2 and the LBPH face recognition algorithm, but this experiment includes horizontal image flipping. This experiment also used the same set of 74 images of female actresses. These images were used to create the facial prediction training data, the associated labels and identification numbers.
</p>

<p align="justify">
The predict feature of LBPHFaceRecognizer was also utilized in this experiment. The prediction of the LBPH face recognition algorithm will analyze an image containing face and attempt to identify the face contain in the image against images (original and horizontal flipped) within the training data. If a probable match is found within the training data then the person name and the confidence score will be overlaid on the image displayed as shown below:
</p>

<p align="center"><br>
<img src="https://github.com/johnbumgarner/image_similarity_experiments/blob/master/flip_jennifer_aniston_confidence_score.jpg">
</p>

<p align="justify">
The algorithm was able to successfully predict that an image contained in the training data matched either an original or horizontal image that was used in creating the training data. The success rate was again less accurate for images not used in the training data. 
</p>

### Notes:

_The code within this repository is **not** production ready. It was **strictly** designed for experimental testing purposes only._


####
<p align="center"><br>
<img src="https://github.com/johnbumgarner/image_simarility_experiments/blob/master/females_with_earrings_test_images.jpg">
</p>



