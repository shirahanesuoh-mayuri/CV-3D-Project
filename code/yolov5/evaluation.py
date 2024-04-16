from tqdm import tqdm
import cv2, torch
import torch.nn.functional as F
def predict(model, name):
  image = cv2.imread(name)
  image = cv2.resize(image, (224, 224))
  image = torch.FloatTensor(image).unsqueeze(0).transpose(1, 3).to(model.device)
  results = model(image)
  return F.softmax(results, dim=1)
def test_img(file_pathname, model, name_dict, save_folder):
    file_dir = os.listdir(file_pathname)
    top1_index = 0

    # create new folders for storing the categorized images.
    for k, v in name_dict.items():
        name_folder = os.path.join(save_folder, v)
        if not os.path.exists(name_folder):
            os.makedirs(name_folder)

    # we assume that Drowsy is positive sample and Non Drowsy is negative sample.
    TP_cur, TN_cur, FP_cur, FN_cur = 0, 0, 0, 0
    for filename in tqdm(file_dir):
        result = predict(model, file_pathname + '/' + filename)[0]
        img = cv2.imread(file_pathname + '/' + filename)
        cls = int(torch.argmax(result).item())
        prod = result[0].item()
        if cls == "1":
          result = [prod, 1-prod]
          top1_index = 0
        elif cls == "0":
          result = [1-prod, prod]
          top1_index = 1
        labels_and_probs = {
            name_dict[0]: result[0],
            name_dict[1]: result[1]
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

        # count TP, TN, FP, FN.
        if file_pathname == '/content/drive/MyDrive/Sub/Neutral':
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
