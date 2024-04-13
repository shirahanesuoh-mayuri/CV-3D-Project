# Documentation

## Install
1. Clone the project repository to your local machine:
```
git clone https://github.com/shirahanesuoh-mayuri/CV-3D-Project.git
```
  
2. Install the project dependencies
```
pip install -r requirements.txt
```

## Usage

<details>
<summary>YOLOv8</summary>
  
1. Navigate to the yolov8 directory
```
cd CV-3D-Project/code/yolov8
```

2. Train the model
```
python main.py
```

3. Evaluate on the test dataset
```
python evaluation.py
```
</details>



# Data Source
The first two datasets are used as our training dataset, the third one is our test dataset:

1. Driver Drowsiness Dataset (DDD): [https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd/data](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd/data) the size of this dataset is 3GB, more than 4.1k images.

2. Drowsiness Prediction Dataset: [https://www.kaggle.com/datasets/rakibuleceruet/drowsiness-prediction-dataset/data](https://www.kaggle.com/datasets/rakibuleceruet/drowsiness-prediction-dataset/data) the size of this dataset is 2GB, with 9120 files.

3. Test data: [https://github.com/bindujiit/Driver-Drowsiness-Dataset-D3S-/blob/main/README.md[5]](https://github.com/bindujiit/Driver-Drowsiness-Dataset-D3S-/blob/main/README.md)
