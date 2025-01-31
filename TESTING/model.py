import os
from typing import List

import cv2
import numpy as np
import torch
import torchvision

TEMPERATURE = 2.344858


def get_convnexttiny_model(num_classes: int = 29):
    model = torchvision.models.mobilenet_v3_small()
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


class ImageClassifier:
    def __init__(self):
        self.model_ = None
        model_fname = os.path.join(os.path.dirname(__file__), 'model_sec_best_cleaned.pth')
        # Check if the model file exists
        if not os.path.isfile(model_fname):
            raise IOError(f'The file "{model_fname}" does not exist!')

        # Load the model
        checkpoint = torch.load(model_fname, map_location=torch.device('cpu'))
        # checkpoint = torch.load(model_fname)
        self.model_ = get_convnexttiny_model()
        self.model_.load_state_dict(checkpoint['state_dict'])
        # self.model_.load_state_dict(checkpoint)

        # Set up device and model
        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_.eval().to(self.device_)

    def _preprocess(self, img):
        img = cv2.resize(img, (200, 200))
        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img).float().to(self.device_)

    def predict(self, image: np.ndarray) -> torch.Tensor:
        """Predict the class of a single image."""
        image_tensor = self._preprocess(image)
        image_tensor = torch.unsqueeze(image_tensor, 0)  # Add batch dimension
        with torch.no_grad():
            outputs = self.model_(image_tensor)
        return outputs / TEMPERATURE

    def predict_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """Predict the class of a batch of images."""
        image_tensors = torch.stack([self._preprocess(image) for image in images])
        with torch.no_grad():
            outputs = self.model_(image_tensors)
        return outputs / TEMPERATURE

    def get_top_n_predictions(self, outputs: torch.Tensor, n: int = 3):
        label_encode = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10,
                        'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20,
                        'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'del': 26, 'nothing': 27, 'space': 28}
        label_encode = {v: k for k, v in label_encode.items()}

        top_n_probs, top_n_indices = torch.topk(outputs, n)
        top_n_probs = top_n_probs.squeeze().cpu().numpy()
        top_n_indices = top_n_indices.squeeze().cpu().numpy()

        top_n_labels = [label_encode[idx] for idx in top_n_indices]

        return list(zip(top_n_labels, top_n_probs))


if __name__ == '__main__':
    classifier = ImageClassifier()
    image = cv2.imread('img.png')
    prediction = classifier.predict(image)  # For a single image

    # Get top-3 predictions and their probabilities
    top_3_predictions = classifier.get_top_n_predictions(prediction, n=3)

    # Output top-3 predictions with their probabilities
    for label, prob in top_3_predictions:
        print(f'Predicted label: {label}, Probability: {prob:.4f}')
