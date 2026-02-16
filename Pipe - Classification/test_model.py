import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model architecture
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 3)  # 2 classes: left and right
model.load_state_dict(torch.load("turn_classifier.pth", map_location=device))
model.eval()
model.to(device)

# Define the same transform used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define class names (in the same order as folders in ImageFolder)
class_names = ['left', 'right', 'up']  # Make sure this matches your folder names

# Load and preprocess the image
image_path = "test_image_flash_right.jpg"  # <-- Replace with your image path
img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension

# Make prediction and compute confidence
with torch.no_grad():
    output = model(img_tensor)
    probabilities = F.softmax(output, dim=1)
    _, predicted = output.max(1)
    predicted_class = class_names[predicted.item()]
    confidence = probabilities[0][predicted.item()].item() * 100

# Print and visualize result
print(f"Predicted class: {predicted_class} ({confidence:.2f}% confidence)")

plt.imshow(img)
plt.title(f"Predicted: {predicted_class} ({confidence:.1f}%)")
plt.axis('off')
plt.show()