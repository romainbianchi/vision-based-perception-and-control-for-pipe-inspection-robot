import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

class DirectionDetectionModel():

    def __init__(self, state_dict_path, class_names):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model = models.mobilenet_v2(pretrained=False)
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),])
        self.class_names = class_names

        model.classifier[1] = nn.Linear(model.last_channel, 3)  # 2 classes: left and right
        model.load_state_dict(torch.load(state_dict_path, map_location=self.device))
        model.eval()
        model.to(self.device)


    def predict(self, img, conf_threshold):

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        if img.mode != "RGB":
            img = img.convert("RGB")

        img_tensor = self.transform(img).unsqueeze(0).to(self.device)  # Add batch dimension

        # Make prediction and compute confidence
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = F.softmax(output, dim=1)
            _, predicted = output.max(1)
            predicted_class = self.class_names[predicted.item()]
            confidence = probabilities[0][predicted.item()].item() * 100

            if confidence < conf_threshold:
                predicted_class = 'no known'

        return predicted_class, confidence