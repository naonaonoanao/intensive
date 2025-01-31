import os
import random


def limit_files_in_folder(directory, limit):
    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            files_in_folder = os.listdir(folder_path)

            # Сохраняем только файлы, игнорируя директории
            files_in_folder = [f for f in files_in_folder if os.path.isfile(os.path.join(folder_path, f))]

            # Проверяем количество файлов
            if len(files_in_folder) > limit:
                num_files_to_delete = len(files_in_folder) - limit

                # Выбираем случайные файлы для удаления
                files_to_delete = random.sample(files_in_folder, num_files_to_delete)

                # Удаляем файлы
                for file_name in files_to_delete:
                    file_path = os.path.join(folder_path, file_name)
                    try:
                        os.remove(file_path)
                        print(f"Удалён файл: {file_path}")
                    except Exception as e:
                        print(f"Ошибка при удалении {file_path}: {e}")


# Укажите путь к вашей папке с набором данных и лимит файлов
data_dir = "sign_language_dataset"
file_limit = 2250

# Запуск скрипта
limit_files_in_folder(data_dir, file_limit)