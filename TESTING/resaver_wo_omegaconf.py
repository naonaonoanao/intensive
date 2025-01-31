import torch


def resave_model_without_omegaconf(input_model_path: str, output_model_path: str):
    """
    Пересохранение модели, исключая зависимости от omegaconf.

    :param input_model_path: Путь к исходному файлу модели (.pth)
    :param output_model_path: Путь для сохранения новой модели (.pth)
    """
    # Загрузка модели
    checkpoint = torch.load(input_model_path, map_location=torch.device('cpu'))

    # Проверяем наличие state_dict
    if 'state_dict' not in checkpoint:
        raise KeyError("Файл модели не содержит 'state_dict'. Убедитесь, что загружается правильный файл.")

    # Извлекаем state_dict
    state_dict = checkpoint['state_dict']

    # Сохраняем только state_dict
    torch.save({'state_dict': state_dict}, output_model_path)
    print(f"Модель успешно сохранена в '{output_model_path}' без зависимостей от omegaconf.")


# Пример использования
if __name__ == "__main__":
    input_path = "model_aug.pth"  # Путь к исходной модели
    output_path = "model_aug_cleaned.pth"  # Путь для новой модели

    resave_model_without_omegaconf(input_path, output_path)
