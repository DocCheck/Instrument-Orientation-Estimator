import os
from datetime import datetime

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from preprocessing import generate_datalist, generate_dataset
import augmentation
from dataset import CustomDataset

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def predict_model(opt):
    # generate the dataset
    datalist = generate_datalist(input_path=opt.input_data)
    X_test, y_test = generate_dataset(opt, datalist, desired_size=144)
    # X_test = X_test[:2000]
    # y_test = y_test[:2000]

    # define the augmentation transforms
    custom_transform_inference = augmentation.custom_augmentation_test()
    # apply the augmentation to the dataset and define the data loader for inference
    test_dataset = CustomDataset(X_test, y_test, transform=custom_transform_inference, num_class=opt.n_class)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=opt.batch_size)

    if not os.path.exists(opt.model_path):
        print("The model does not exist !!!")
        return
    else:
        # load the model if it exists
        loaded_model = torch.load(opt.model_path, map_location=DEVICE)
        # if visualization is enabled, create a new directory for the visualization
        if opt.visualization:
            date_time = datetime.now()
            vis_path = "/".join(list(opt.model_path.split("/")[0:-1])) + "/visualization_" + date_time.strftime(
                format='%Y%m%d_%H%M') + "/"
            os.mkdir(vis_path)

        # start the inference
        correct_test = 0
        total_test = 0
        val_output = []
        with torch.no_grad():
            counter = 0
            for data in test_loader:
                tensor = data[0]
                tensor = tensor.to(DEVICE)
                label = data[1]
                label = label.to(DEVICE)
                outputs = loaded_model(tensor)

                _, predicted = torch.max(outputs.data, 1)
                __, predicted_list = torch.sort(outputs.data, 1, descending=True)
                val_output.append(predicted)
                total_test += tensor.size(0)
                correct_test += (torch.abs(predicted - label) < opt.diff_t).sum().item()

                if opt.visualization:
                    for i in range(tensor.size(0)):
                        if torch.abs(predicted[i] - label[i]) > opt.diff_t:
                            output_sample_name = "img_" + str(counter) + "_" + str(label[i]) + "_" + str(
                                predicted[i]) + ".png"
                            save_image(tensor[i], vis_path + output_sample_name)
                            counter += 1

        print('Accuracy on Test Data :', 100 * (correct_test / total_test), '%')
        print('Test length :', len(val_output), correct_test, total_test)


if __name__ == '__main__':
    predict_model()
