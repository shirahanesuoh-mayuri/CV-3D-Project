from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm

# perform image classification on the test dataset
def test_img(file_pathname, model, name_dict, save_folder):
    file_dir = os.listdir(file_pathname)

    # create new folders for storing the categorized images.
    for k, v in name_dict.items():
        name_folder = os.path.join(save_folder, v)
        if not os.path.exists(name_folder):
            os.makedirs(name_folder)

    # we assume that Drowsy is positive sample and Non Drowsy is negative sample.
    TP_cur, TN_cur, FP_cur, FN_cur = 0, 0, 0, 0
    for filename in tqdm(file_dir):
        img = cv2.imread(file_pathname + '/' + filename)
        results = model.predict(img)

        for result in results:
            name_dict = result.names
            probs = result.probs.cpu().numpy()
            top1_index = result.probs.top1
            labels_and_probs = {
                name_dict[0]: probs.data[0],
                name_dict[1]: probs.data[1]
            }

            class_name = name_dict[top1_index]

            # save the labels and probs on the image
            x, y = 10, 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            color = (255, 255, 255)
            thickness = 2

            for label, prob in labels_and_probs.items():
                text = f'{label}: {prob:.5f}'
                cv2.putText(img, text, (x, y), font, font_scale, color, thickness)
                y += 30

            save_img_path = os.path.join(save_folder, class_name, filename)
            cv2.imwrite(save_img_path, img)

            # count TP, TN, FP, FN
            if file_pathname == 'test/Neutral':
                if class_name == 'Non Drowsy':
                    TN_cur += 1
                else:
                    FP_cur += 1
            else:
                if class_name == 'Drowsy':
                    TP_cur += 1
                else:
                    FN_cur += 1

    return TP_cur, TN_cur, FP_cur, FN_cur

if __name__ == '__main__':
    test_folder = ['test/Eyeclose', 'test/Neutral', 'test/Yawn']

    name_dict = {0: 'Drowsy', 1: 'Non Drowsy'}
    save_folder = 'runs/classify/predict'
    model = YOLO('runs/classify/train/weights/best.pt')

    TP, TN, FP, FN = 0, 0, 0, 0
    for img_folder in test_folder:
        TP1, TN1, FP1, FN1 = test_img(img_folder, model, name_dict, save_folder)
        TP += TP1
        TN += TN1
        FP += FP1
        FN += FN1

    # metrics
    accuracy = (TP + TN) / (TP + FN + FP + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * TP / (2 * TP + FP + FN)

    print("Accuracy: ", accuracy)       # Accuracy:  0.8766404199475065
    print("Precision: ", precision)     # Precision:  0.9468085106382979
    print("Recall: ", recall)           # Recall:  0.827906976744186
    print("F1: ", F1)                   # F1:  0.8833746898263027


