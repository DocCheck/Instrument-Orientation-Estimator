import numpy as np
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchsummary as summary

from preprocessing import generate_datalist, generate_dataset
import augmentation
from dataset import CustomDataset
from model import AlexNet

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_model(opt):
    # if the model doesn't exist, create a new path for it
    if not os.path.exists("data/models"):
        os.makedirs("data/models")
    if not os.path.exists(opt.model_path):
        model_name = opt.model_path.split("/")[-1]
        date_time = datetime.now()
        model_path = "data/models/" + "model_" + date_time.strftime(format='%Y%m%d_%H%M') + "/"
        os.makedirs(model_path)
        model_path += model_name
        opt.model_path = model_path
    else:
        model_path = opt.model_path

    # generate the dataset
    datalist = generate_datalist(input_path=opt.input_data)
    dataset_img, dataset_label = generate_dataset(opt, datalist, desired_size=144)
    # dataset_img = dataset_img[:500]
    # dataset_label = dataset_label[:500]

    # split the dataset into training and validation
    X_train, X_val, y_train, y_val = train_test_split(dataset_img, dataset_label, test_size=opt.train_val[1],
                                                      shuffle=True)
    print("Number of training samples : ", np.shape(X_train))
    print("Number of validation samples : ", np.shape(X_val))

    # define the augmentation transforms
    custom_transform = augmentation.custom_augmentation_train()
    # apply the augmentation to the dataset
    train_dataset = CustomDataset(X_train, y_train, transform=custom_transform, num_class=opt.n_class)
    val_dataset = CustomDataset(X_val, y_val, transform=custom_transform, num_class=opt.n_class)
    # create the dataloaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=opt.batch_size)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=opt.batch_size)

    # load the model if it exists or create a new one
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location=DEVICE)
        print("pre-trained model is loaded successfully")
    else:
        model = AlexNet(num_classes=opt.n_class).to(DEVICE)
    summary.summary(model, (1, opt.imgsz, opt.imgsz))

    # define the optimizer and the loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)

    # start training
    trian_loss = {}
    val_loss = {}
    start_time = time.time()
    for epoch in range(opt.epochs):
        train_losses = []
        val_losses = []
        running_loss = 0
        count = 0
        for i, inp in enumerate(train_loader):
            inputs = inp[0]
            inputs = inputs.to(DEVICE)
            labels = inp[1]
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            count += 1

        # start validation
        if epoch % opt.valid_period == 0:
            for i, inp in enumerate(val_loader):
                inputs = inp[0]
                inputs = inputs.to(DEVICE)
                labels = inp[1]
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
            trian_loss[epoch] = np.mean(train_losses)
            val_loss[epoch] = np.mean(val_losses)
            print('Epoch', epoch, ': train loss =', np.mean(train_losses), ', val loss =', np.mean(val_losses))

    end_time = time.time()
    print("Training Done in", end_time - start_time, "seconds")

    # visualize the training and validation loss
    if opt.visualization:
        fig = plt.figure(figsize=(20, 20))
        plt.plot(list(trian_loss.keys()), list(trian_loss.values()))
        plt.plot(list(val_loss.keys()), list(val_loss.values()))
        plt.savefig("/".join(list(model_path.split("/")[0:-1])) + "/train_val_loss.png")
        plt.close()

    # calculate the accuracy of the model on the training and validation data
    correct_train = 0
    total_train = 0
    correct_val = 0
    total_val = 0
    val_output = []
    with torch.no_grad():
        for data in train_loader:
            tensor = data[0]
            tensor = tensor.to(DEVICE)
            label = data[1]
            label = label.to(DEVICE)
            outputs = model(tensor)
            _, predicted = torch.max(outputs.data, 1)
            total_train += tensor.size(0)
            correct_train += (predicted == label).sum().item()

        for data in val_loader:
            tensor = data[0]
            tensor = tensor.to(DEVICE)
            label = data[1]
            label = label.to(DEVICE)
            outputs = model(tensor)
            _, predicted = torch.max(outputs.data, 1)
            val_output.append(predicted)
            total_val += tensor.size(0)
            correct_val += (predicted == label).sum().item()

    print('Accuracy on the Training Data :', 100 * (correct_train / total_train), '%')
    print('Accuracy on the Validation Data :', 100 * (correct_val / total_val), '%')
    print('Validation length :', len(val_output), correct_val, total_val)

    # save the model
    torch.save(model, model_path)


if __name__ == '__main__':
    train_model()
