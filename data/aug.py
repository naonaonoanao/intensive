import os
import random
from PIL import Image, ImageOps
import albumentations as A
import numpy as np
from torchvision import transforms

# Указываем путь к датасету
dataset_path = 'sign_language_dataset'

# Определяем аугментации
transform = A.Compose([
    A.Rotate(limit=10, p=0.5),  # Уменьшили угол поворота до 10 градусов
    A.RandomBrightnessContrast(p=0.2),  # Яркость и контраст
    A.RandomGamma(p=0.2),  # Изменение гаммы
    A.ToGray(p=0.1),  # Преобразование в чб
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.3),  # Цветовые изменения
])


def augment_and_save_image(img_path, save_path):
    # Открываем изображение
    image = Image.open(img_path)
    image = np.array(image)

    # Применяем аугментацию
    augmented = transform(image=image)
    augmented_image = augmented['image']

    # Сохраняем аугментированное изображение
    augmented_img = Image.fromarray(augmented_image)
    augmented_img.save(save_path)


def process_and_augment(dataset_path):
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                if img_name.endswith('.jpg'):
                    # Генерируем новое имя для аугментированного изображения
                    augmented_img_name = f"aug_{random.randint(1000, 9999)}_{img_name}"
                    augmented_img_path = os.path.join(label_path, augmented_img_name)

                    # Аугментируем и сохраняем
                    augment_and_save_image(img_path, augmented_img_path)


# Запускаем обработку и аугментацию
process_and_augment(dataset_path)
