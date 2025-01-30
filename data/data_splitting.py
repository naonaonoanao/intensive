import os
import shutil

dataset = "new_sign_language_dataset/"

dataset_train = "sign_language_dataset_train/"
dataset_val = "sign_language_dataset_val/"
dataset_test = "sign_language_dataset_test/"

if not os.path.isdir(dataset_val):
    os.mkdir(dataset_val)
if not os.path.isdir(dataset_train):
    os.mkdir(dataset_train)
if not os.path.isdir(dataset_test):
    os.mkdir(dataset_test)

labels = os.listdir(dataset)

for label in labels:
    if not os.path.isdir(dataset_val + label):
        os.mkdir(dataset_val + label)
    if not os.path.isdir(dataset_train + label):
        os.mkdir(dataset_train + label)
    if not os.path.isdir(dataset_test + label):
        os.mkdir(dataset_test + label)

    for i in range(1, 3001):
        file_name = label + str(i) + ".jpg"

        if i % 10 < 7:
            shutil.copy(dataset + label + "/" + file_name, dataset_train + label + "/" + file_name)
        elif i % 10 < 9:
            shutil.copy(dataset + label + "/" + file_name, dataset_val + label + "/" + file_name)
        else:
            shutil.copy(dataset + label + "/" + file_name, dataset_test + label + "/" + file_name)