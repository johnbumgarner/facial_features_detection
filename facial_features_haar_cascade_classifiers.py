#!/usr/local/bin/python3

##################################################################################
# “AS-IS” Clause
#
# Except as represented in this agreement, all work produced by Developer is
# provided “AS IS”. Other than as provided in this agreement, Developer makes no
# other warranties, express or implied, and hereby disclaims all implied warranties,
# including any warranty of merchantability and warranty of fitness for a particular
# purpose.
##################################################################################

##################################################################################
#
# Date Completed: July 24, 2019
# Author: John Bumgarner
#
# Date Revised: October 7, 2020
# Revised by: John Bumgarner
#
# This Python script is designed to use the OpenCV module to identify human faces
# and specific characteristics commonly associated with a face, including eyes,
# nose and mouth area.
#
##################################################################################

#############################################################################################
# The OS module in provides functions for interacting with the operating system.
#############################################################################################
import os

#############################################################################################
# The OpenCV is a library of programming functions mainly aimed at real-time computer vision.
# The Python package is opencv-python
#
# reference: https://pypi.org/project/opencv-python
#############################################################################################
import cv2

#############################################################################################
# LBPH face recognition algorithm
#
# Local Binary Pattern (LBP) is a simple yet very efficient texture operator
# which labels the pixels of an image by thresholding the neighborhood of each
# pixel and considers the result as a binary number.
#
# Parameters: the LBPH uses 4 parameters:
#
# 1. Radius: the radius is used to build the circular local binary pattern and represents the
# radius around the central pixel. It is usually set to 1.
#
# 2. Neighbors: the number of sample points to build the circular local binary pattern.
# The more sample points you include, the higher the computational cost. Max value is 8.
#
# 3. Grid X: the number of cells in the horizontal direction. The more cells, the finer the grid,
# the higher the dimensionality of the resulting feature vector. Max value is 8
#
# 4. Grid Y: the number of cells in the vertical direction. The more cells, the finer the grid,
# the higher the dimensionality of the resulting feature vector. Max value is 8.
#
#############################################################################################
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=4, grid_x=4, grid_y=4)

#############################################################################################
# A Haar Cascade is a machine learning object detection algorithm used to identify objects
# in an image or video and based on the concept of ​​ features.
#
# reference: https://ieeexplore.ieee.org/document/990517
#
# Haar Cascade files
# 1. https://github.com/opencv/opencv/tree/master/data/haarcascades
# 2. http://alereimondo.no-ip.org/OpenCV/34.version?id=60
#############################################################################################

# obtains the absolute paths to the haar cascade installed
# this path will change based on where the haar cascade files
# are located.
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_frontal_face_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
haar_eye_model = os.path.join(cv2_base_dir, 'data/haarcascade_eye.xml')
haar_mouth_model = os.path.join(cv2_base_dir, 'data/haarcascade_smile.xml')
haar_nose_model = os.path.join(cv2_base_dir, 'data/haarcascade_mcs_nose.xml')

# Load the Haar Cascade Classifier used for frontal face images
face_cascade = cv2.CascadeClassifier(haar_frontal_face_model)

# Load the Haar Cascade Classifier used to identify eyes with face images
eye_cascade = cv2.CascadeClassifier(haar_eye_model)

# Load the Haar Cascade Classifier used to identify noses with face images
nose_cascade = cv2.CascadeClassifier(haar_nose_model)

# Load the Haar Cascade Classifier used to identify mouths with face images
mouth_cascade = cv2.CascadeClassifier(haar_mouth_model)


def detect_single_face(image_array, bounding_box):
    """
    This function is designed to draw a bounding rectangle around the facial area of a single person
    contained in an image.

    :param image_array: this is a numpy.ndarray, which is an array object represents a multidimensional,
                  homogeneous array of fixed-size items
    :param bounding_box: this contains the positions of the facial coordinates produced by face_cascade.detectMultiScale
    :return: image with rectangle draw around facial area
    """
    # (x_coordinate, y_coordinate) are the top-left coordinate of the rectangle
    # (width, height) are the width and height of the rectangle
    for (x_coordinate, y_coordinate, width, height) in bounding_box:
        # Draw bounding rectangle based on parameter dimensions
        # BGR color values (3 parameters)
        # BGR color (255, 0, 255) - https://rgb.to/255,0,255
        # Line width in pixels
        cv2.rectangle(image_array, (x_coordinate, y_coordinate), (x_coordinate + width, y_coordinate + height),
                      (255, 0, 255), 2)

    return image_array


def detect_multiple_faces(image_array, bounding_box):
    """
    This function is designed to draw a bounding rectangle around the facial area of multiple people
    contained in an image.

    :param image_array: this is a numpy.ndarray, which is an array object represents a multidimensional,
                  homogeneous array of fixed-size items
    :param bounding_box: this contains the positions of the facial coordinates produced by face_cascade.detectMultiScale
    :return: image with rectangle draw around facial areas
    """
    # (x_coordinate, y_coordinate) are the top-left coordinate of the rectangle
    # (width, height) are the width and height of the rectangle
    for (x_coordinate, y_coordinate, width, height) in bounding_box:
        # Draw bounding rectangle based on parameter dimensions
        # BGR (Blue, Green, Red) color values (3 parameters)
        # BGR color (255, 0, 255) - https://rgb.to/255,0,255
        # Line width in pixels
        cv2.rectangle(image_array, (x_coordinate, y_coordinate), (x_coordinate + width, y_coordinate + height),
                      (255, 0, 255), 2)

    return image_array


def detect_eyes_single_face(image_array, bounding_box, grayscale_image):
    """
    This function is designed to draw a bounding rectangle around the eye area in the facial area
    of a single person contained in an image.

    :param image_array: this is a numpy.ndarray, which is an array object represents a multidimensional,
                  homogeneous array of fixed-size items
    :param bounding_box: this contains the positions of the facial coordinates produced by face_cascade.detectMultiScale
    :param grayscale_image: image color conversion, which is gray scale
    :return: image with rectangle draw around eye area
   """
    # (x_coordinate, y_coordinate) are the top-left coordinate of the rectangle
    # (width, height) are the width and height of the rectangle
    for (x_coordinate, y_coordinate, width, height) in bounding_box:

        # roi_gray is a numpy.ndarray based on the gray scale of the image
        roi_gray = grayscale_image[y_coordinate:y_coordinate + height, x_coordinate:x_coordinate + width]

        # roi_color is a numpy.ndarray based on the color scale of the image
        roi_color = image_array[y_coordinate:y_coordinate + height, x_coordinate:x_coordinate + width]

        # eyes contains the calculate facial coordinates produced by eyes_cascade.detectMultiScale in relation
        # to face_cascade
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE)

        for (eye_x_coordinate, eye_y_coordinate, eye_width, eye_height) in eyes:
            # Draw bounding rectangle based on parameter dimensions around eyes
            # BGR (Blue, Green, Red) color values (3 parameters)
            # RGB color (255, 0, 128) - https://rgb.to/128,0,255
            # Line width in pixels
            cv2.rectangle(roi_color, (eye_x_coordinate, eye_y_coordinate),
                          (eye_x_coordinate + eye_width, eye_y_coordinate + eye_height), (128, 0, 255), 2)

    return image_array


def detect_nose_single_face(image_array, bounding_box, grayscale_image):
    """
    This function is designed to draw a bounding rectangle around the nose area in the facial area
    of a single person contained in an image.

    :param image_array: this is a numpy.ndarray, which is an array object represents a multidimensional,
                  homogeneous array of fixed-size items
    :param bounding_box: this contains the positions of the facial coordinates produced by face_cascade.detectMultiScale
    :param grayscale_image: image color conversion, which is gray scale
    :return: image with rectangle draw around nose area
   """
    # (x_coordinate, y_coordinate) are the top-left coordinate of the rectangle
    # (width, height) are the width and height of the rectangle
    for (x_coordinate, y_coordinate, width, height) in bounding_box:

        # roi_gray is a numpy.ndarray based on the gray scale of the image
        roi_gray = grayscale_image[y_coordinate:y_coordinate + height, x_coordinate:x_coordinate + width]

        # roi_color is a numpy.ndarray based on the color scale of the image
        roi_color = image_array[y_coordinate:y_coordinate + height, x_coordinate:x_coordinate + width]

        # nose contains the calculate facial coordinates produced by nose_cascade.detectMultiScale in relation
        # to face_cascade
        nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=8, flags=cv2.CASCADE_SCALE_IMAGE)

        for (nose_x_coordinate, nose_y_coordinate, nose_width, nose_height) in nose:
            # Draw bounding rectangle based on parameter dimensions around nose
            # BGR (Blue, Green, Red) color values (3 parameters)
            # RGB color (255, 0, 0) - https://rgb.to/255,0,0
            # Line width in pixels
            cv2.rectangle(roi_color, (nose_x_coordinate, nose_y_coordinate),
                          (nose_x_coordinate + nose_width, nose_y_coordinate + nose_height), (255, 0, 0), 2)

    return image_array


def detect_mouth_single_face(image_array, bounding_box, grayscale_image):
    """
    This function is designed to draw a bounding rectangle around the mouth area in the facial area
    of a single person contained in an image.

    :param image_array: this is a numpy.ndarray, which is an array object represents a multidimensional,
                  homogeneous array of fixed-size items
    :param bounding_box: this contains the positions of the facial coordinates produced by face_cascade.detectMultiScale
    :param grayscale_image: image color conversion, which is gray scale
    :return: image with rectangle draw around mouth area
    """
    # (x_coordinate, y_coordinate) are the top-left coordinate of the rectangle
    # (width, height) are the width and height of the rectangle
    for (x_coordinate, y_coordinate, width, height) in bounding_box:

        # roi_gray is a numpy.ndarray based on the gray scale of the image
        roi_gray = grayscale_image[y_coordinate:y_coordinate + height, x_coordinate:x_coordinate + width]

        # roi_color is a numpy.ndarray based on the color scale of the image
        roi_color = image_array[y_coordinate:y_coordinate + height, x_coordinate:x_coordinate + width]

        # mouth contains the calculate facial coordinates produced by mouth_cascade.detectMultiScale in relation
        # to face_cascade
        mouth = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE)

        for (mouth_x_coordinate, mouth_y_coordinate, mouth_width, mouth_height) in mouth:
            # Draw bounding rectangle based on parameter dimensions around mouth
            # BGR (Blue, Green, Red) color values (3 parameters)
            # RGB color (0, 255, 128) - https://rgb.to/0,255,128
            # Line width in pixels
            cv2.rectangle(roi_color, (mouth_x_coordinate, mouth_y_coordinate),
                          (mouth_x_coordinate + mouth_width, mouth_y_coordinate + mouth_height), (0, 255, 128), 2)

    return image_array


def detect_single_face_multiple_features(image_array, bounding_box, grayscale_image):
    """
    This function is designed to draw a bounding rectangles around the face, eyes, nose and
    mouth areas in the facial area of a single person contained in an image.

    :param image_array: image is a numpy.ndarray, which is an array object represents a multidimensional,
                  homogeneous array of fixed-size items
    :param bounding_box: this contains the positions of the facial coordinates produced by face_cascade.detectMultiScale
    :param grayscale_image:  image color conversion, which is gray scale
    :return: image with rectangle draw around all facial areas
    """
    # (x_coordinate, y_coordinate) are the top-left coordinate of the rectangle
    # (width, height) are the width and height of the rectangle
    for (x_coordinate, y_coordinate, width, height) in bounding_box:
        # Draw bounding rectangle based on parameter dimensions
        # BGR (Blue, Green, Red) color values (3 parameters)
        # BGR color (255, 0, 255) - https://rgb.to/255,0,255
        # Line width in pixels
        cv2.rectangle(image_array, (x_coordinate, y_coordinate), (x_coordinate + width, y_coordinate + height),
                      (255, 0, 255), 2)

        # roi_gray is a numpy.ndarray based on the gray scale of the image
        roi_gray = grayscale_image[y_coordinate:y_coordinate + height, x_coordinate:x_coordinate + width]

        # roi_color is a numpy.ndarray based on the color scale of the image
        roi_color = image_array[y_coordinate:y_coordinate + height, x_coordinate:x_coordinate + width]

        # eyes contains the calculate facial coordinates produced by eyes_cascade.detectMultiScale in relation
        # to face_cascade
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE)

        # nose contains the calculate facial coordinates produced by nose_cascade.detectMultiScale in relation
        # to face_cascade
        nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=8, flags=cv2.CASCADE_SCALE_IMAGE)

        # mouth contains the calculate facial coordinates produced by mouth_cascade.detectMultiScale in relation
        # to face_cascade
        mouth = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE)

        for (eye_x_coordinate, eye_y_coordinate, eye_width, eye_height) in eyes:
            # Draw bounding rectangle based on parameter dimensions
            # BGR (Blue, Green, Red) color values (3 parameters)
            # BGR color (128, 0, 255) - https://rgb.to/128,0,255
            # Line width in pixels
            cv2.rectangle(roi_color, (eye_x_coordinate, eye_y_coordinate),
                          (eye_x_coordinate + eye_width, eye_y_coordinate + eye_height), (128, 0, 255), 2)

        for (nose_x_coordinate, nose_y_coordinate, nose_width, nose_height) in nose:
            # Draw bounding rectangle based on parameter dimensions around nose
            # BGR (Blue, Green, Red) color values (3 parameters)
            # BGR color (255, 0, 0) - https://rgb.to/255,0,0
            # Line width in pixels
            cv2.rectangle(roi_color, (nose_x_coordinate, nose_y_coordinate),
                          (nose_x_coordinate + nose_width, nose_y_coordinate + nose_height), (255, 0, 0), 2)

        for (mouth_x_coordinate, mouth_y_coordinate, mouth_width, mouth_height) in mouth:
            # Draw bounding rectangle based on parameter dimensions
            # BGR (Blue, Green, Red) color values (3 parameters)
            # BGR color (0, 255, 128) - https://rgb.to/0,255,128
            # Line width in pixels
            cv2.rectangle(roi_color, (mouth_x_coordinate, mouth_y_coordinate),
                          (mouth_x_coordinate + mouth_width, mouth_y_coordinate + mouth_height), (0, 255, 128), 2)

    return image_array


def display_facial_detection_results(processed_photo):
    # write image out
    # rename as needed
    # cv2.imwrite('all_features_detection.jpg', processed_photo)

    # Display image with bounding rectangles
    # and title in a window. The window
    # automatically fits to the image size.
    cv2.imshow('Facial features recognized', processed_photo)

    # Displays the window infinitely
    key = cv2.waitKey(0) & 0xFF

    # Shuts down the display window and terminates
    # the Python process when a key is pressed on
    # the window
    if key == ord('q') or key == 113 or key == 27:
        cv2.destroyAllWindows()


def process_single_image(filename):
    """
    This function is designed process a single image and calculate the facial coordinates produced
    by face_cascade.detectMultiScale

    :param filename: the name of the image file to use
    :return: image is a numpy.ndarray, which is an array object represents a multidimensional,
             homogeneous array of fixed-size items

             gray is the image color conversion, which is gray scale

             faces contains the calculate facial coordinates produced by face_cascade.detectMultiScale
    """
    # Read in the image
    photograph = cv2.imread(filename)
    if photograph is not None:

        # cv2.cvtColor(input_image, flag) where flag determines the type of conversion.
        grayscale_image = cv2.cvtColor(photograph, cv2.COLOR_BGR2GRAY)

        # detectMultiScale detects objects of different sizes in the input image.
        # The detected objects are returned as a list of rectangles.
        faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            return photograph, grayscale_image, faces
        else:
            print(f'no faces were detected in the image file: {filename}')
    else:
        print(f'The following image could not be read by OpenCV: {filename}')


def validate_file_extension(filename):
    """
    This function is designed to verify that the file being process has
    a valid file extension for known image files.

    :param filename: name of the file to check
    :return: boolean
    """
    accepted_extensions = ('.bmp', '.gif', '.jpg', '.jpeg', '.png', '.svg', '.tiff')
    if filename.endswith(accepted_extensions):
        return True
    else:
        return False


############################################################
# This code below is running a single face detection test.
#
# Please review the functions above to determine the
# parameters required to execute other detection tests.
###########################################################
base_image_name = 'natalie_portman.jpeg'
image_with_sunglasses = 'natalie_portman_sunglasses.jpg'
image_black_swan_makeup = 'natalie_portman_black_swan.jpg'

valid_file_extension = validate_file_extension(base_image_name)

if valid_file_extension is True:
    face_results = process_single_image(base_image_name)
    if face_results:
        image = face_results[0]
        grayscale = face_results[1]
        facial_area = face_results[2]
        detected_object_bounding_box = detect_single_face(image, facial_area)
        display_facial_detection_results(detected_object_bounding_box)

elif valid_file_extension is False:
    print(f'The following image does not have a valid file_extension {base_image_name}')
