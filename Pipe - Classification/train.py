from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

# Image transformations with minimal augmentation
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(5),           # More rotation
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomPerspective(distortion_scale=0.3),  # Add perspective changes
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Add blur
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load datasets
train_dataset = datasets.ImageFolder("turn_dataset_45_90", transform=transform_train)
val_dataset = datasets.ImageFolder("turn_dataset_45_90", transform=transform_val)

# Split into train/val with fixed seed for reproducibility
torch.manual_seed(42)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_indices, val_indices = torch.utils.data.random_split(range(len(train_dataset)), [train_size, val_size])

train_ds = torch.utils.data.Subset(train_dataset, train_indices.indices)
val_ds = torch.utils.data.Subset(val_dataset, val_indices.indices)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)  # Smaller batch size
val_loader = DataLoader(val_ds, batch_size=32)

model = models.convnext_tiny(pretrained=True)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(train_dataset.classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
# Add weight decay for regularization
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# Early stopping variables
best_val_acc = 0.0
patience_counter = 0
early_stop_patience = 3

print(f"Training with {len(train_ds)} samples, validating with {len(val_ds)} samples")
print(f"Classes: {train_dataset.classes}")

# Training loop with validation monitoring
for epoch in range(30):
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_acc = 100. * correct / total
    avg_loss = running_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_acc = 100. * val_correct / val_total
    val_avg_loss = val_loss / len(val_loader)
    
    print(f"Epoch {epoch+1:2d} | Train: {train_acc:5.1f}% (loss: {avg_loss:.3f}) | Val: {val_acc:5.1f}% (loss: {val_avg_loss:.3f})")
    
    # Learning rate scheduling
    scheduler.step(val_acc)
    
    # Early stopping logic
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), "best_turn_classifier.pth")
        print(f"  → New best validation accuracy: {val_acc:.1f}%")
    else:
        patience_counter += 1
        
    # Stop if validation performance hasn't improved
    if patience_counter >= early_stop_patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break
    
    # Warning if overfitting detected
    if train_acc - val_acc > 10:
        print(f"Overfitting detected: {train_acc - val_acc:.1f}% gap")

print(f"\nBest validation accuracy: {best_val_acc:.1f}%")
print("Best model saved as 'best_turn_classifier.pth'")

# Load best model for final evaluation
model.load_state_dict(torch.load("best_turn_classifier.pth"))
model.eval()

# Final detailed validation
class_correct = [0] * len(train_dataset.classes)
class_total = [0] * len(train_dataset.classes)

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        
        for i in range(labels.size(0)):
            label = labels[i].item()
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

print("\nPer-class accuracy:")
for i, class_name in enumerate(train_dataset.classes):
    if class_total[i] > 0:
        acc = 100 * class_correct[i] / class_total[i]
        print(f"  {class_name}: {acc:.1f}% ({class_correct[i]}/{class_total[i]})")