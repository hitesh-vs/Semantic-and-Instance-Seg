import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import models
from dataloader import WindowDataset
from network import UNetResNet18Scratch
# from network import UNetMobileNetV2Scratch
from loadParam import IMG_DIR, MASK_DIR, BATCH_SIZE, LR, NUM_WORKERS
import os

# Function to calculate accuracy (binary segmentation)
def calculate_accuracy(output, target):
    output = torch.sigmoid(output)
    predicted = (output > 0.5).float()
    correct = (predicted == target).float().sum()
    return correct / target.numel()

# focal loss for imbalanced
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')  # Use 'none' to keep per-element loss

    def forward(self, logits, targets):
        # Binary Cross-Entropy Loss
        bce_loss = self.bce_loss(logits, targets)
        
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)
        
        # Focal loss weights based on probability and gamma
        focal_weight = self.alpha * (1 - probs) ** self.gamma
        
        # Multiply the focal weight with BCE loss
        loss = focal_weight * bce_loss
        
        return loss.mean()  # Return the average loss


# Set up the device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Model initialization
model = UNetResNet18Scratch(num_classes=1).to(device)

model = model.to(device)

# Dataset loading and splitting (80% train, 10% val, 10% test)
dataset = WindowDataset(img_dir=IMG_DIR, mask_dir=MASK_DIR)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# DataLoader initialization
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
# loss_fn = FocalLoss(alpha=1.0, gamma=2.0)  # You can adjust alpha and gamma based on your dataset
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Learning Rate Scheduler: Reduces learning rate by 0.10 after every 10 epochs
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** (epoch // 10))

# Store the metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training function
def train(loader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    total_acc = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        # loss = loss_fn(output.squeeze(1), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += calculate_accuracy(output, target).item()
    return total_loss / len(loader), total_acc / len(loader)

# Validation function
def validate(loader, model, loss_fn):
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            # loss = loss_fn(output.squeeze(1), target)

            total_loss += loss.item()
            total_acc += calculate_accuracy(output, target).item()
    return total_loss / len(loader), total_acc / len(loader)

# Training loop
epochs = 100

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # Train the model
    train_loss, train_acc = train(train_loader, model, loss_fn, optimizer)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validate the model
    val_loss, val_acc = validate(val_loader, model, loss_fn)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # Save model weights after every epoch
    torch.save(model.state_dict(), f'resnet18_unet/model_mobNet_epoch_{epoch + 1}.pth')
    print(f"Model weights saved for epoch {epoch + 1}")

    # Step the scheduler
    scheduler.step()

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Save final model
torch.save(model.state_dict(), 'resnet18_unet/final_ResNet_model.pth')
print("Final model saved as final_model.pth")

# Plot training and validation loss
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('./resnet18_unet/loss_curve.png')

# Plot training and validation accuracy
plt.figure()
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('./resnet18_unet/accuracy_curve.png')

# Export the metrics to a CSV file
metrics_data = {
    'Train Loss': train_losses,
    'Validation Loss': val_losses,
    'Train Accuracy': train_accuracies,
    'Validation Accuracy': val_accuracies
}
df = pd.DataFrame(metrics_data)
df.to_csv('./resnet18_unet/training_metrics.csv', index=False)

# Evaluate on the test set
test_loss, test_acc = validate(test_loader, model, loss_fn)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
