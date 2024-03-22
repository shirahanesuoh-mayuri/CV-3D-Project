# CV - Proposal

---

**Advisor: Yannick Pauler**

**Team member:** 

Yufan Feng, [yufan.feng@stud.uni-heidelberg.de](mailto:yufan.feng@stud.uni-heidelberg.de)

Yaojie Wang, [yaojie.wang@stud.uni-heidelberg.de](mailto:yaojie.wang@stud.uni-heidelberg.de)

Bingyu Guo, [bingyu.guo@stud.uni-heidelberg.de](mailto:bingyu.guo@stud.uni-heidelberg.de)

Tian Tan, [tan.tan@stud.uni-heidelberg.de](mailto:tan.tan@stud.uni-heidelberg.de)

---

## 1. Concrete Goal of your Project

This project aims to develop a fatigue detection system for drivers. we will employ two principal methods for detecting driver fatigue: facial feature analysis and neural network-based detection using YOLO models.  
The input will be a video of a driver[5]. Our output will be a decision whether or not the driver is drowsy.

1. **Data Acquisition:**  
    
    We intend to use two datasets for our data.  
    1.Driver Drowsiness Dataset (DDD)[https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd/data](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd/data) the size of this dataset is 3GB, more than 4.1k images.  
    2.Drowsiness Prediction Dataset[https://www.kaggle.com/datasets/rakibuleceruet/drowsiness-prediction-dataset/data](https://www.kaggle.com/datasets/rakibuleceruet/drowsiness-prediction-dataset/data) the size of this dataset is 2GB, with 9120 files.  
    We choose the video in this project as a test data[https://github.com/bindujiit/Driver-Drowsiness-Dataset-D3S-/blob/main/README.md](https://github.com/bindujiit/Driver-Drowsiness-Dataset-D3S-/blob/main/README.md)[5]  

2. **Data Process:**  

    We'll use "dlib"[1] package to process data, the "dlib" package is a open-source ML package. The main advantage of ERT is high speed. We will use the pretrained model "shape predictor 68 face landmarks.dat.bz2" to process the image into a 68x2 matrix contain the information of human face[2].  
    
3. **Fatigue Detection Methods**:  

    in this part, we designed two methods to get the decision of fatigue  
    1. we will directly based on the result of facial landmarks dectection. We plan to calculate the EAR(Eye Aspected Ratio)for both eyes, if the value is smaller than 0.3, the eye can be considered as closed[6]. The number of distance anomalies was recorded and used as eye closure to calculate PERCLOS[3].if the PERCLOS(as the time occupied by a certain percentage (70% or 80%) of the eyes closed in a unit time (usually 1 min or 30 s)) is bigger than 0.15.[4] the fatigue is detected. Also we plan to count the frequency of yawning by calculate MAR(Mouth Aspect Ratio), smaller than 20 can be counted as yawning[7]
    2. Also, we will train YOLOv8 with datasets mentioned before, and use trained model to detect fatigue[8]. The train/test is 3/2 for our data.Because the model will feedback the prediction of fatigue, wo plan to count it and calculate the ratio between drowsy and whole video frame, if the value is bigger than 0.15[4], we think the driver is drowsy.
4. **Innovations**:  

    1. We designed a competition of accuarcy between YOLOv8 and our direct method, Depends on the test dataset, the number of drowsiness = closed eye + yawning.

    2. We are also interested in how the YOLOv8 model performs in detecting facial landmarks compared with "dlib" package.  

    3. We want to use the process data(just facial landmarks) to train YOLOv8, but we cannot find a previous work. So we are not sure is that possible.

## 3. Timeline

For this project, we have set the following four milestones:

1. March 4th - March 10th: Data Preparation and Preprocessing
2. March 11th - March 24th: Model Training, Optimization, Testing, and Evaluation
3. March 25th - March 31st: UI Development and Integration
4. April 1st - April 7th: Report Writing

## 4. Reference

[1] Davis E. King. 2009. Dlib-ml: A Machine Learning Toolkit. J. Mach. Learn. Res. 10 (12/1/2009), 1755–1758.

[2] Vahid Kazemi and Josephine Sullivan. 2014. One Millisecond Face Alignment with an Ensemble of Regression Trees. In Proceedings of the 2014 IEEE Conference on Computer Vision and Pattern Recognition (CVPR '14). IEEE Computer Society, USA, 1867–1874. https://doi.org/10.1109/CVPR.2014.241

[3] Drowsiness Detection Based on Facial Landmark and Uniform Local Binary Pattern. Dini Adni Navastara et al 2020 J. Phys.: Conf. Ser. 1529 052015
DOI 10.1088/1742-6596/1529/5/052015

[4] Chang RC, Wang CY, Chen WT, Chiu CD. Drowsiness Detection System Based on PERCLOS and Facial Physiological Signal. Sensors (Basel). 2022 Jul 19;22(14):5380. doi: 10.3390/s22145380. PMID: 35891065; PMCID: PMC9323611.

[5] Gupta, I., Garg, N., Aggarwal, A., Nepalia, N., & Verma, B. (2018, August). Real-time driver's drowsiness monitoring based on dynamically varying threshold. In 2018 Eleventh International Conference on Contemporary Computing (IC3) (pp. 1-6). IEEE

[6] N. N. Pandey and N. B. Muppalaneni, "Real-Time Drowsiness Identification based on Eye State Analysis," 2021 International Conference on Artificial Intelligence and Smart Systems (ICAIS), Coimbatore, India, 2021, pp. 1182-1187, doi: 10.1109/ICAIS50930.2021.9395975.

[7] Li D, Cui Z, Cao F, Cui G, Shen J, Zhang Y. Learning State Assessment in Online Education Based on Multiple Facial Features Detection. Comput Intell Neurosci. 2022 Jan 29;2022:3986470. doi: 10.1155/2022/3986470. PMID: 35132313; PMCID: PMC8817852.

[8] [ultralytics](https://github.com/ultralytics/ultralytics?tab=readme-ov-file)