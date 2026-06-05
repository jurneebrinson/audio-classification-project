# Imports
import os
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from audio_classification.data.dataset import SpectrogramDataset
from audio_classification.models.cnn import SpectrogramCNN

# Config
TRAIN_DIR = "data/data_split/train"
VAL_DIR = "data/data_split/val"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
NUM_CLASSES = 6

best_epoch = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

MODEL_OUTPUT_DIR = "outputs/models"
PLOT_OUTPUT_DIR = "outputs/plots"
LOG_OUTPUT_DIR = "outputs/logs"

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_OUTPUT_DIR, exist_ok=True)

best_model_path = os.path.join(MODEL_OUTPUT_DIR, f"best_model.pth")
run_model_path = os.path.join(MODEL_OUTPUT_DIR, f"run_{timestamp}.pth")

loss_path = os.path.join(PLOT_OUTPUT_DIR, f"loss_{timestamp}.png")
acc_path = os.path.join(PLOT_OUTPUT_DIR, f"acc_{timestamp}.png")

# Datasets & Loaders
train_dataset = SpectrogramDataset(root_dir = TRAIN_DIR)
val_dataset = SpectrogramDataset(root_dir = VAL_DIR)

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

# Model
model = SpectrogramCNN(num_classes = NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# Metrics Storage
train_losses, val_losses = [], []
train_accs, val_accs = [], []

best_val_acc = 0.0

# Training Loop
for epoch in range(EPOCHS):
    
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        preds = torch.argmax(outputs, dim = 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            preds = torch.argmax(outputs, dim = 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    
    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    # Log
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    print("-" * 40)

    if val_acc > best_val_acc:
        best_epoch = epoch + 1
        best_val_acc = val_acc

        torch.save({
            "epoch": best_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
        }, best_model_path)

        print(f"New best model saved! " f"Epoch {best_epoch} | Validation accuracy: {val_acc:.4f}")

torch.save({
    "epoch": EPOCHS,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_val_acc": best_val_acc,
}, run_model_path)

# Plots
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Loss Curve")
plt.savefig(loss_path)
plt.close()

plt.figure()
plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Val Accuracy")
plt.legend()
plt.title("Accuracy Curve")
plt.savefig(acc_path)
plt.close()

history = pd.DataFrame({
    "epoch": range(1, EPOCHS + 1),
    "train_loss": train_losses,
    "val_loss": val_losses,
    "train_acc": train_accs,
    "val_acc": val_accs
})

history_path = os.path.join(
    LOG_OUTPUT_DIR,
    f"training_history_{timestamp}.csv"
)

history.to_csv(history_path, index=False)

leaderboard_path = os.path.join(
    LOG_OUTPUT_DIR,
    "model_leaderboard.csv"
)

new_entry = pd.DataFrame({
    "timestamp": [timestamp],
    "best_val_acc": [best_val_acc],
    "best_epoch": [best_epoch],
    "epochs": [EPOCHS],
    "batch_size": [BATCH_SIZE],
    "learning_rate": [LR],
    "weight_decay": [1e-4],
    "label_smoothing": [0.1]
})

if os.path.exists(leaderboard_path):
    leaderboard = pd.read_csv(leaderboard_path)
    leaderboard = pd.concat(
        [leaderboard, new_entry],
        ignore_index=True
    )
else:
    leaderboard = new_entry

leaderboard = leaderboard.sort_values(
    by="best_val_acc",
    ascending=False
)

leaderboard.to_csv(
    leaderboard_path,
    index=False
)

print("Training history saved to:", history_path)

print("\nTraining complete.")
print(f"\nBest validation accuracy: {best_val_acc:.4f}")
print(f"Best epoch: {best_epoch}")
print("Best model saved to:", best_model_path)
print("Run model saved to:", run_model_path)
print("Plots saved to:", PLOT_OUTPUT_DIR)
print("Logs saved to:", LOG_OUTPUT_DIR)