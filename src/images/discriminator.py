import glob
import math
import os
import time
from multiprocessing import Pool
import random

import cv2
import torch
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
from torch.autograd import Variable
from torch.nn import Linear, ReLU, L1Loss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, \
    Dropout, Sigmoid, CrossEntropyLoss
from torch.nn.functional import log_softmax
from torch.optim import Adam, SGD


from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize

import numpy as np
import matplotlib.pyplot as plt


SPARSE_LOAD = 2
AS_GRAY = False
USE_CUDA = True and torch.cuda.is_available()
MODEL_PATH = '/home/adwiii/git/perception_fuzzing/src/images/discriminator.model'


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1 if AS_GRAY else 3, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(8192, 4096),
            ReLU(inplace=True),
            Linear(4096, 2048),
            ReLU(inplace=True),
            Linear(2048, 1024),
            ReLU(inplace=True),
            Linear(1024, 512),
            ReLU(inplace=True),
            Linear(512, 256),
            ReLU(inplace=True),
            Linear(256, 128),
            ReLU(inplace=True),
            Linear(128, 2),
            Sigmoid(),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = x.log_softmax(dim=-1)
        # x = x.argmax(dim=-1)
        return x


def load_image(image_file):
    image = imread(image_file, as_gray=AS_GRAY)
    scale_factor = 1
    if scale_factor > 1:
        if AS_GRAY:
            image = resize(image, (image.shape[0] // scale_factor, image.shape[1] // scale_factor))
        else:
            image = resize(image, (image.shape[0] // scale_factor, image.shape[1] // scale_factor, 3))
    return image


class DatasetGenerator(keras.utils.Sequence):
    def __init__(self, batch_size=50, shuffle=True):
        self.batch_size = batch_size
        self.x = []
        self.y = []
        self.img_map = {}
        for img_file in glob.glob("/home/adwiii/git/perception_fuzzing/src/images/fri_*/mutations/*_edit.png"):
            self.x.append(img_file)
            self.y.append(1)  # edit class is 1
        for img_file in glob.glob("/home/adwiii/data/cityscapes/sut_gt_testing/mutations/*.png"):
            self.x.append(img_file)
            self.y.append(0)  # orig class is 0m
        if shuffle:
            shuffler = np.random.permutation(len(self.x))
            self.x = [self.x[index] for index in shuffler]
            self.y = [self.y[index] for index in shuffler]

        with Pool(28) as pool:
            results = {}
            for count, img_file in enumerate(self.x):
                # only half fit in RAM, load every other one so that it is balanced
                # in how long each batch takes to load later on
                if count % 2 == 0:  # TODO: figure out actual logic to compute that
                    results[img_file] = pool.apply_async(load_image, (img_file,))
            for img_file, res in results.items():
                print('loaded', img_file)
                image = res.get()
                self.img_map[img_file] = image

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def get_img(self, file_name):
        if file_name in self.img_map:
            return self.img_map[file_name]
        else:
            return load_image(file_name)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:min((idx + 1) * self.batch_size, len(self.x))]
        batch_y = self.y[idx * self.batch_size:min((idx + 1) * self.batch_size, len(self.y))]
        x_arr = []
        with Pool(28) as pool:
            results = {}
            for img_file in batch_x:
                if img_file in self.img_map:
                    results[img_file] = (True, self.img_map[img_file])
                else:
                    results[img_file] = (False, pool.apply_async(load_image, (img_file,)))
            for img_file, (preloaded, res) in results.items():
                if preloaded:
                    image = res
                else:
                    image = res.get()
                x_arr.append(image)
        x_arr = np.array(x_arr, dtype=np.float32)
        x_arr = np.reshape(x_arr, (-1, 3, x_arr.shape[1], x_arr.shape[2]))
        y_arr = np.array(batch_y)
        return x_arr, y_arr


class CityscapesDiscriminator:
    # Adapted from: https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/
    def __init__(self):
        self.model = Net()
        pass
        # todo figure out how to load in model?

    def evaluate(self):
        self.model = self.model.eval()
        generator = DatasetGenerator(batch_size=50)
        num_batches = len(generator)
        cpu_y = None
        pred_y = None
        pred_probs = None
        image_files = generator.x
        for i in range(num_batches):
            batch_start = time.time()
            train_x, train_y = generator[i]
            train_x = torch.from_numpy(train_x).float()
            # train_y = torch.from_numpy(train_y).long()  # cross entropy needs integer classes
            cpu_y = train_y if cpu_y is None else np.concatenate([cpu_y, train_y])
            if USE_CUDA:
                train_x = train_x.cuda()
            # prediction for training and validation set
            predicted_batch = self.model(train_x)
            pred_y_to_add = predicted_batch.argmax(dim=-1).cpu().detach().numpy()
            pred_y = pred_y_to_add if pred_y is None else np.concatenate([pred_y, pred_y_to_add])
            pred_probs_to_add = predicted_batch.cpu().detach().numpy()
            pred_probs = pred_probs_to_add if pred_probs is None else np.concatenate([pred_probs, pred_probs_to_add])
            # computing the updated weights of all the model parameters
            batch_elapsed = time.time() - batch_start
            print('Evaluating. Finished batch', i + 1, 'of', num_batches,
                  'in %0.2f s, Estimated remaining %0.2f m' %
                  (batch_elapsed, (num_batches - i - 1) * batch_elapsed / 60))

        print(confusion_matrix(cpu_y, pred_y))
        print(classification_report(cpu_y, pred_y, labels=[0, 1]))
        with open('/home/adwiii/git/perception_fuzzing/src/images/good_edits_all_untrained.txt', 'w') as f:
            for index, (actual, pred) in enumerate(zip(cpu_y, pred_y)):
                if actual > 0 and pred < 1:
                    # if actual is 1 (meaning edited) and pred is 0 (meaning unedited) then we wan
                    # print(label_val[index])
                    f.write(str((image_files[index], pred_probs[index])) + '\n')
        all_edits = [(image_files[index], pred_probs[index], cpu_y[index], pred_y[index]) for index in range(len(image_files))]
        all_edits = sorted(all_edits, key=lambda item: item[1][0], reverse=True)
        with open('/home/adwiii/git/perception_fuzzing/src/images/all_images_ranked_untrained.txt', 'w') as f:
            for base_edit in all_edits:
                f.write(str(base_edit) + '\n')
        with open('/home/adwiii/git/perception_fuzzing/src/images/all_edits_ranked_untrained.txt', 'w') as f:
            for base_edit in all_edits:
                if base_edit[2] == 0:
                    continue  # don't include the ones that were not edits
                f.write(str(base_edit) + '\n')

    def train(self, data_root):
        generator = DatasetGenerator()
        num_batches = len(generator)

        # defining the optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.003, momentum=0.9)
        # defining the loss function
        criterion = CrossEntropyLoss()
        print(self.model)

        if USE_CUDA:
            self.model = self.model.cuda()
            criterion = criterion.cuda()
        num_epochs = 75
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            train_loss_epoch = 0
            val_loss_epoch = 0
            self.model.train()
            epoch_start = time.time()
            for i in range(num_batches):
                batch_start = time.time()
                train_x, train_y = generator[i]
                train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1)
                # train_x = np.reshape(train_x, (-1, 3, train_x.shape[1], train_x.shape[2]))
                # val_x = np.reshape(val_x, (-1, 3, val_x.shape[1], val_x.shape[2], -1))
                train_x = torch.from_numpy(train_x).float()
                train_y = torch.from_numpy(train_y).long()  # cross entropy needs integer classes
                val_x = torch.from_numpy(val_x).float()
                val_y = torch.from_numpy(val_y).long()  # cross entropy needs integer classes
                if USE_CUDA:
                    train_x = train_x.cuda()
                    train_y = train_y.cuda()
                    val_x = val_x.cuda()
                    val_y = val_y.cuda()
                # prediction for training and validation set
                output_train = self.model(train_x)
                output_val = self.model(val_x)
                # print('x_val.shape', x_val.shape)
                # print('y_val.shape', y_val.shape)
                # print('output_val.shape', output_val.shape)
                # print('output_val', output_val)
                # computing the training and validation loss
                loss_train = criterion(output_train, train_y)
                loss_val = criterion(output_val, val_y)
                train_loss_epoch += loss_train.cpu().detach().numpy()
                val_loss_epoch += loss_val.cpu().detach().numpy()
                # computing the updated weights of all the model parameters
                batch_elapsed = time.time() - batch_start
                print('Epoch :', epoch + 1, 'Finished batch', i+1, 'of', num_batches,
                      'in %0.2f s, Estimated remaining %0.2f m' %
                      (batch_elapsed, (num_batches - i - 1) * batch_elapsed / 60))
                # clearing the Gradients of the model parameters
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()
            train_losses.append(train_loss_epoch)
            val_losses.append(val_loss_epoch)
            # tr_loss = loss_train.item()
            print('Epoch : ', epoch + 1, '\t', 'loss :', train_loss_epoch, 'Time for epoch:', time.time() - epoch_start, 's')
            torch.save(self.model.state_dict(), MODEL_PATH)
            print('Model Saved')
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.legend()
        plt.show()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        if USE_CUDA:
            self.model = self.model.cuda()


if __name__ == '__main__':
    loaded = torch.load(MODEL_PATH)
    # orig = Net().state_dict()
    # print(type(orig))
    # print(type(loaded))
    # for x, y in zip(orig, loaded):
    #     print('key', x == y)
    #     print('value', orig[x].cpu().detach().numpy() == loaded[y].cpu().detach().numpy())
    # exit()
    discriminator = CityscapesDiscriminator()
    if os.path.exists(MODEL_PATH):
        print('model found, skipping training')
        discriminator.load_state_dict(torch.load(MODEL_PATH))
    # else:
    #     print('no model found, starting training')
    # discriminator.train('/home/adwiii/data/cityscapes')
    discriminator.model = discriminator.model.cuda()
    print('starting evaluation')
    discriminator.evaluate()




