<p align="center">
  <img src="https://github.com/johnbumgarner/facial_detection_prediction-/blob/master/graphic/facial_recognition.png">
</p>

# Overview Facial Detection 

<p align="justify">

Facial detection or recognition


face recognition owned significant consideration and appreciated as one of the most promising applications in the field of image analysis. Face detection can consider a substantial part of face recognition operations. According to its strength to focus computational resources on the section of an image holding a face. The method of face detection in pictures is complicated because of variability present across human faces such as pose, expression, position and orientation, skin colour, the presence of glasses or facial hair, differences in camera gain, lighting conditions, and image resolution.


notes: 
https://towardsdatascience.com/face-detection-for-beginners-e58e8f21aad9
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html#face-detection
http://www.willberger.org/cascade-haar-explained/
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html



Most humans can look at two photos and quickly determine if the images are either similarity or dissimilarity in nature. Computers can be programmed to perform a similar task, but the results can vary, because of multiple factors(e.g., lighting conditions, perspectives) that humans can instinctively do automatically.  Humans have little difficulty seeing the subtle differents between a rose and a camellia or a gardenia and a rose. A computer on the other hand will recognize these objects as flowers, but would likely classify all of these images as a single type of flower, roses.   

There are numerous use cases for image similarities technologies. These use cases range from duplicate image detection to domain specific image clustering. Identifying duplicate images in Apple Photo is a common use case for many of us dealing with a large digital image library. Some of us have likey used Google’s Reverse Image Search to look for a specific photo that we want to know more about. Google will scour its massive database for images similar to the one used in your query. 

</p>

## Primary objective of this repository

<p align="justify">
This repository is going to examine various methods and algorithms that can be used to identify specific facial characteristics, such as the eye and mouth areas of a human face. The 3 images used in these tests are of the well-known female actress <i>Natalie Portman</i>.
  
Another objective of this repository is to determine the capabilities and limitations of the Python libraries used to perform these facial characteristics tests.
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

#### Haar Cascade Classifiers - Facial Detection

One of the most basic Haar Cascade classifiers is the one used to detect the facial area of a human face looking directly at the camera. This base-level algorithm comes pretrained, so it is able to identify images that have human face characteristics and their associated parameters and ones that have no human face characteristics, such as an image of a cat.


```python
# This code was extraction from mutiple functions in the script facial_features_haar_cascade_classifiers.py

image_name = 'natalie_portman.jpeg'
photograph = cv2.imread(image_name)
grayscale_image = cv2.cvtColor(photograph, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.3, minNeighbors=5)
for (x_coordinate, y_coordinate, width, height) in faces:
    cv2.rectangle(photograph, (x_coordinate, y_coordinate),
                  (x_coordinate + width, y_coordinate + height), (255, 0, 255), 2)
```

The image of <i>Natalie Portman</i> below has a <i>bounding box</i> drawn around the entire facial area identified by the Haar Cascade classifier  <i>haarcascade_frontalface_default.xml.</i>

<p align="left">
  <img src="https://github.com/johnbumgarner/facial_detection_prediction-/blob/master/graphic/facial_front_detection.jpg">
</p>


#### Haar Cascade Classifiers - Eye Detection

The eye area is another human facial characteristic that can be identified using the Haar Cascade classifier <i>haarcascade_eye.xml</i>, which is used in collaboration with the Haar Cascade classifier for identifing frontal faces. 

```python
# This code was extraction from mutiple functions in the script facial_features_haar_cascade_classifiers.py

image_name = 'natalie_portman.jpeg'
photograph = cv2.imread(image_name)
grayscale_image = cv2.cvtColor(photograph, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.3, minNeighbors=5)

for (x_coordinate, y_coordinate, width, height) in faces:
    roi_gray = grayscale_image[y_coordinate:y_coordinate + height, x_coordinate:x_coordinate + width]
    roi_color = photograph[y_coordinate:y_coordinate + height, x_coordinate:x_coordinate + width]
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE)

    for (eye_x_coordinate, eye_y_coordinate, eye_width, eye_height) in eyes:
        cv2.rectangle(roi_color, (eye_x_coordinate, eye_y_coordinate),
                  (eye_x_coordinate + eye_width, eye_y_coordinate + eye_height), (128, 0, 255), 2)
```

The image of Natalie Portman below has a <i>bounding box</i> drawn around the eye area identified by the Haar Cascade classifier <i>haarcascade_eye.xml.</i>

<p align="left">
  <img src="https://github.com/johnbumgarner/facial_detection_prediction-/blob/master/graphic/eye_detection.jpg">
</p>


#### Haar Cascade Classifiers - Nose Detection

The nose area classifier <i>haarcascade_mcs_nose.xml</i> also works in concert with the Haar Cascade classifier used to identify frontal faces. 

```python
# This code was extraction from mutiple functions in the script facial_features_haar_cascade_classifiers.py

image_name = 'natalie_portman.jpeg'
photograph = cv2.imread(image_name)
grayscale_image = cv2.cvtColor(photograph, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.3, minNeighbors=5)

for (x_coordinate, y_coordinate, width, height) in faces:
    roi_gray = grayscale_image[y_coordinate:y_coordinate + height, x_coordinate:x_coordinate + width]
    roi_color = photograph[y_coordinate:y_coordinate + height, x_coordinate:x_coordinate + width]
    nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=8, flags=cv2.CASCADE_SCALE_IMAGE)

    for (nose_x_coordinate, nose_y_coordinate, nose_width, nose_height) in nose:
        cv2.rectangle(roi_color, (nose_x_coordinate, nose_y_coordinate),
                      (nose_x_coordinate + nose_width, nose_y_coordinate + nose_height), (255, 0, 0), 2)
```

The image of Natalie Portman below has a <i>bounding box</i> drawn around the nose area identified by the Haar Cascade classifier <i>haarcascade_mcs_nose.xml.</i>

<p align="left">
  <img src="https://github.com/johnbumgarner/facial_detection_prediction-/blob/master/graphic/nose_detection.jpg">
</p>

#### Haar Cascade Classifiers - Mouth Detection

This classifier uses the Haar Cascade <i>haarcascade_smile.xml</i> in conjunction with the Haar Cascade classifier used to identify frontal faces.

```python
# This code was extraction from mutiple functions in the script facial_features_haar_cascade_classifiers.py

image_name = 'natalie_portman.jpeg'
photograph = cv2.imread(image_name)
grayscale_image = cv2.cvtColor(photograph, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.3, minNeighbors=5)

for (x_coordinate, y_coordinate, width, height) in faces:
    roi_gray = grayscale_image[y_coordinate:y_coordinate + height, x_coordinate:x_coordinate + width]
    roi_color = photograph[y_coordinate:y_coordinate + height, x_coordinate:x_coordinate + width]
    mouth = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE)

    for (mouth_x_coordinate, mouth_y_coordinate, mouth_width, mouth_height) in mouth:
        cv2.rectangle(roi_color, (mouth_x_coordinate, mouth_y_coordinate),
                      (mouth_x_coordinate + mouth_width, mouth_y_coordinate + mouth_height), (0, 255, 128), 2)
```

The image of Natalie Portman below has a <i>bounding box</i> drawn around the mouth area identified by the Haar Cascade classifier <i>haarcascade_smile.xml.</i>

<p align="left">
  <img src="https://github.com/johnbumgarner/facial_detection_prediction-/blob/master/graphic/mouth_detection.jpg">
</p>

#### Haar Cascade Classifiers - All Facial Characteristics Detection

The image of Natalie Portman below has <i>bounding boxes</i> drawn around the all the facial characteristics previously identified by all the Haar Cascade classifiers listed above.

<p align="left">
  <img src="https://github.com/johnbumgarner/facial_detection_prediction-/blob/master/graphic/all_features_detection.jpg">
</p>


### Haar Cascade Classifiers Detection Issues:

The <i>OpenCV Haar Cascade classifiers</i> have some detection issues. For example, if a person in an image has a hair style that obscures segments of their face from view then the <i>Haar Cascade Frontal Face classifier</i> might have problems detecting the person's face.  Other factors, such as excessive facial hair, sun glasses or heavy makeup can cause both rudimentary and some advanced facical detection algorithms to have errors in detecting various facial features. Some of these Haar Cascade classifiers algorithms were orginally trained using photographics of people with lighter skin tones, so some detection issues can occur with certain skin colors.  

For instance the image below is of <i>Natalie Portman</i> from the movie <i>Black Swan</i>.  As you can see she is wearing heavy makeup for her role as <i>Nina</i>.  

<p align="left">
  <img src="https://github.com/johnbumgarner/facial_features_detection/blob/master/graphic/natalie_portman_black_swan.jpg", width="200" height="200">
</p>

The Haar Cascade classifiers <i>Frontal Face</i> and <i>Nose</i> were able to properly identify these characteristics on this photograph.  The Haar Cascade classifier for <i>Eyes</i>, was not able to identify that facial feature on this image, likey because of the heavy eye makeup being worned by Ms. Portman. The mouth classifier also had difficulties accurately locating the mouth area in this photograph as shown in the image below. 

<p align="left">
  <img src="https://github.com/johnbumgarner/facial_features_detection/blob/master/graphic/mouth_detection_natalie_portman_black_swan.jpg", width="200" height="200">
</p>

The <i>Black Swan</i> transformation of <i>Natalie Portman</i> was an extreme example to showcase the <i>OpenCV Haar Cascade classifiers</i> limitations, so here is another example of the actress wearing dark sunglasses. 

<p align="left">
  <img src="https://github.com/johnbumgarner/facial_features_detection/blob/master/graphic/natalie_portman_sunglasses.jpg", width="200" height="200">
</p>

The Haar Cascade classifier <i>Frontal Face</i> was the only algorithm that was able to properly identify that core facial characteristic on this photograph.  
Both the <i>Eyes</i> and <i>Nose</i> classifiers failed to detect those facial characteristics in the image. Like in the previous <i>Black Swan</i> photo the  
mouth classifier also had difficulties accurately locating the mouth area in the photograph of <i>Natalie Portman</i> wearing sunglasses.  The results of the 
mouth classifier are shown below.

<p align="left">
  <img src="https://github.com/johnbumgarner/facial_features_detection/blob/master/graphic/mouth_detection_natalie_portman_sunglasses.jpg", width="200" height="200">
 
</p>

Both the examples above were very basic levels of facial camouflage, but both were able to foil certain aspects of the <i>Haar Cascade classifiers.</i>. The website [CV Dazzle](https://cvdazzle.com) has more complex camouflage photographs that can be used in testing the capabilities and limitations of facial detection algorithms. 


### Notes:

_The code within this repository is **not** production ready. It was **strictly** designed for experimental testing purposes only._
