import os
import shutil
import random

dataset = "sign_language_dataset/"
dataset_train = "sign_language_dataset_train/"
dataset_val = "sign_language_dataset_val/"
dataset_test = "sign_language_dataset_test/"

# Создаём каталоги, если они не существуют
if not os.path.isdir(dataset_val):
    os.mkdir(dataset_val)
if not os.path.isdir(dataset_train):
    os.mkdir(dataset_train)
if not os.path.isdir(dataset_test):
    os.mkdir(dataset_test)

labels = os.listdir(dataset)

for label in labels:
    # Создаём каталоги для каждого лейбла
    if not os.path.isdir(dataset_val + label):
        os.mkdir(dataset_val + label)
    if not os.path.isdir(dataset_train + label):
        os.mkdir(dataset_train + label)
    if not os.path.isdir(dataset_test + label):
        os.mkdir(dataset_test + label)

    # Получаем список всех файлов для данного лейбла
    files = os.listdir(dataset + label)

    # Перемешиваем список файлов случайным образом
    random.shuffle(files)

    # Разделяем файлы на тренировочную, валидационную и тестовую выборки
    total_files = len(files)
    train_files = files[:int(0.7 * total_files)]  # 70% на тренировку
    val_files = files[int(0.7 * total_files):int(0.9 * total_files)]  # 20% на валидацию
    test_files = files[int(0.9 * total_files):]  # 10% на тест

    # Копируем файлы в соответствующие каталоги
    for file_name in train_files:
        shutil.copy(dataset + label + "/" + file_name, dataset_train + label + "/" + file_name)

    for file_name in val_files:
        shutil.copy(dataset + label + "/" + file_name, dataset_val + label + "/" + file_name)

    for file_name in test_files:
        shutil.copy(dataset + label + "/" + file_name, dataset_test + label + "/" + file_name)
