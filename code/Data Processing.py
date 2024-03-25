import os
import numpy as np


import dlib_shape_detect as D
import EAR_MAR_compute as C

#use to storage the result of dlib ERV,MRV


img_path_base = os.path.join("..", "datasets", "original_data")
out_path_base = os.path.join("..", "datasets", "processed_data")




#draw landmarks from points
def G_landmark(img_path):
    img_path = img_path_base + img_path
    landMarks = D.image_Processing(img_path)

    return landMarks

#storage landmarks
def S_landmarks(landMarks,img_path):
    out_path = out_path_base + img_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    file_path = os.path.join(out_path + "\\68_lM.npy")
    np.save(file_path, landMarks)

#calculate and storage EAR and MAR
def EAR_MAR_cpt(landMarks,img_path):
    n_faces = landMarks.shape[0]
    EAR_L = np.zeros(n_faces)
    EAR_R = np.zeros(n_faces)
    MAR = np.zeros(n_faces)
    for i in range(n_faces):
        # get landmarks of eyes and mouth
        left_Eye_LM = landMarks[i, 36:42, :]
        right_Eye_LM = landMarks[i, 42:48, :]

        mouth_LM = landMarks[i, 48:68, :]
        EAR_L[i] = C.EAR_cpt(left_Eye_LM)
        EAR_R[i] = C.EAR_cpt(right_Eye_LM)
        MAR[i] = C.MAR_cpt(mouth_LM)
    Features = G_Feature(EAR_L, EAR_R, MAR,img_path)
    return Features

def G_Feature(EAR_L, EAR_R, MAR, img_path): #combine the EAR and MAR to a matrix, add label
    label = os.path.basename(img_path)
    Feature = np.stack((EAR_L, EAR_R, MAR), axis= 1)
    label_vector = np.full((Feature.shape[0], 1), label)
    Feature = np.hstack((Feature, label_vector))
    return Feature

def S_Features(Feature, img_path):
    out_path = out_path_base + img_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    file_path = os.path.join(out_path + "\\EAR&MAR.npy")
    np.save(file_path, Feature)



img_path = input("数据所在目录:")
landMarks = G_landmark(img_path)
Features = EAR_MAR_cpt(landMarks,img_path)
S_Features(Features, img_path)
S_landmarks(landMarks, img_path)






