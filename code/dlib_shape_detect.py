import dlib
import cv2
import numpy as np
import os


#get landmarks from image.

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def image_Processing(img_path):
    landmarks = []
    for filename in os.listdir(img_path):
        file_path = os.path.join(img_path, filename)
        img = cv2.imread(file_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector(rgb_img, 0)
        for face in faces:
            shape = predictor(rgb_img, face)
            landmark_points = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            landmarks.append(landmark_points)

    landmark = np.stack(landmarks)

    return landmark




