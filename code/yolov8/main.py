from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

# Split the dataset into train and validation datasets first, then store them under the corresponding folder.
def prepare_datasets(src_folder, dst_folder):
    train_dir = os.path.join(dst_folder, 'train')
    val_dir = os.path.join(dst_folder, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    classes = ['Drowsy', 'Non Drowsy']
    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

        class_dir = os.path.join(src_folder, cls)
        images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
        train_data, val_data = train_test_split(images, test_size=0.2, random_state=42)

        for img in train_data:
            shutil.copy(img, os.path.join(train_dir, cls, os.path.basename(img)))
        for img in val_data:
            shutil.copy(img, os.path.join(val_dir, cls, os.path.basename(img)))

# train the model
def train(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    model.train(data='../images',
                epochs=50,
                patience=25,
                imgsz=256,
                batch=32,
                save=True,
                save_period=10,
                device=device,
                optimizer='Adam')

# A method to plot line graphs
def plot(data, metrics, label):
    folder_path = 'img'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data['epoch'], data[metrics], color='blue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.grid(True)
    fig.savefig('img/' + label + '.png')
    plt.show()

if __name__ == "__main__":
    # split dataset
    source_data_folder = '../images/train1'
    destination_folder = '../images'
    prepare_datasets(source_data_folder, destination_folder)

    # train model
    yolov8n = YOLO('yolov8n-cls.pt')
    train(yolov8n)

    # plot training results
    results = pd.read_csv('runs/classify/train/results.csv')
    results.columns = results.columns.str.strip()
    plot(results, 'metrics/accuracy_top1', 'Training Accuracy')
    plot(results, 'train/loss', 'Training Loss')
    plot(results, 'val/loss', 'Validation Loss')