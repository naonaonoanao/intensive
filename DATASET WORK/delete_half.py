import os
import random


def delete_random_files(directory):
    print('Храни тебя Аллах...')
    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            files_in_folder = os.listdir(folder_path)

            # Сохраняем только файлы, игнорируя директории
            files_in_folder = [f for f in files_in_folder if os.path.isfile(os.path.join(folder_path, f))]

            # Определяем половину файлов для удаления
            num_files_to_delete = len(files_in_folder) // 2

            # Выбираем случайные файлы
            files_to_delete = random.sample(files_in_folder, num_files_to_delete)

            # Удаляем файлы
            for file_name in files_to_delete:
                file_path = os.path.join(folder_path, file_name)
                try:
                    os.remove(file_path)
                    print(f"Удалён файл: {file_path}")
                except Exception as e:
                    print(f"Ошибка при удалении {file_path}: {e}")


# Укажите путь к вашей папке с набором данных
data_dir = "sign_language_dataset"

while True:
    print("Точно хочешь нахуй снести половину датасета? (y/n) - ", end='')
    ans = input()
    if ans == 'y':
        break
    elif ans == 'n':
        print('Ну и похуй')
        exit()

# Запуск скрипта
delete_random_files(data_dir)

