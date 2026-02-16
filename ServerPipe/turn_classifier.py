import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

class TurnClassifier:

    def __init__(self, model_path, class_names):
        # Define the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the trained model architecture
        self.model = models.convnext_tiny(pretrained=False)
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, len(class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

        # Define the same transform used during training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.class_names = class_names

    def predict(self, image):

        img = Image.fromarray(image.astype('uint8'))
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Make prediction and compute confidence
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = F.softmax(output, dim=1)
            _, predicted = output.max(1)
            predicted_class = self.class_names[predicted.item()]
            confidence = probabilities[0][predicted.item()].item() * 100

        return predicted_class, confidence