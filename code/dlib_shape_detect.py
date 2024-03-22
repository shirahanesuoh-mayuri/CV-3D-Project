import dlib
import cv2
import numpy as np

#get landmarks from image.

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def image_Processing(img_path):
    img = cv2.imread(img_path)
    gray_Img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_Img)

    landmarks = []
    for face in faces:
        shape = predictor(gray_Img, face)
        landmark_points = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
        landmarks.append(landmark_points)

    # 将 landmarks 转换为 NumPy 数组
    landmarks_array = np.stack(landmarks)

    return landmarks_array




