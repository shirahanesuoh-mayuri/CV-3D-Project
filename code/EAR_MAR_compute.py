import numpy as np


#used to calculate the EAR and MAR based on landmarks

def EAR_cpt(eye_LM):
    #计算垂直距离
    v1 = np.linalg.norm(eye_LM[1] - eye_LM[5])
    v2 = np.linalg.norm(eye_LM[2] - eye_LM[4])
    vertical_dist = (v1 + v2) * 0.5

    #计算水平距离
    horizontal_dist = np.linalg.norm(eye_LM[0] - eye_LM[3])

    EAR = vertical_dist / horizontal_dist

    return EAR

def MAR_cpt(mouth_LM):
    #计算垂直距离
    v_dist_top = np.linalg.norm(mouth_LM[13] - mouth_LM[19])  # 上嘴唇
    v_dist_bottom = np.linalg.norm(mouth_LM[14] - mouth_LM[18])  # 下嘴唇
    vertical_dist = (v_dist_top + v_dist_bottom) / 2.0

    # 计算水平距离（水平）
    horizontal_dist = np.linalg.norm(mouth_LM[12] - mouth_LM[16])  # 嘴角之间的距离

    # 计算MAR
    mar = vertical_dist / horizontal_dist

    return mar

