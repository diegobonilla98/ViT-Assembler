import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from simple_vit import ViT


image_size = 224
patch_size = 16
dim = 384
depth = 10
heads = 12
mlp_dim = 1536
channels = 3
dim_head = 32
batch_size = 128
num_classes = 257
log_interval = 10

transform_train = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset_root = r'D:\256_ObjectCategories'
full_dataset = datasets.ImageFolder(root=dataset_root, transform=transform_train)

# Split dataset
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = transform_val

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


# Define ViT Classifier
class ViTClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super(ViTClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(dim, num_classes)

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        embeddings = self.encoder(x, evaluation=True).mean(dim=1)
        logits = self.classifier(embeddings)
        return logits


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ViT(
    image_size=image_size,
    patch_size=patch_size,
    dim=dim,
    depth=depth,
    heads=heads,
    mlp_dim=mlp_dim,
    channels=channels,
    dim_head=dim_head
)
model = model.to(device)
model.load_state_dict(torch.load("checkpoints\\model_epoch_48.pth", weights_only=True))
model.eval()

for param in model.parameters():
    param.requires_grad = False

# Instantiate classifier
classifier_model = ViTClassifier(encoder=model, num_classes=num_classes).to(device)

# Define loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = Adam(classifier_model.classifier.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Setup TensorBoard and checkpointing
log_dir = Path('logs') / 'classification_experiment'
writer = SummaryWriter(log_dir=log_dir)

checkpoint_dir = Path('checkpoints_256_objects_classification')
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Training Loop
num_epochs = 20
best_val_accuracy = 0.0
total_steps = 0

for epoch in range(1, num_epochs + 1):
    # Training Phase
    classifier_model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader, 1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = classifier_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Log training status
        if batch_idx % log_interval == 0 or batch_idx == len(train_loader):
            step_loss = running_loss / (batch_idx * batch_size)
            step_accuracy = correct / total
            print(f"Epoch [{epoch}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], "
                  f"Loss: {step_loss:.4f}, Accuracy: {step_accuracy:.4f}")

            # Log to TensorBoard
            writer.add_scalar('Loss/Train_step', step_loss, total_steps)
            writer.add_scalar('Accuracy/Train_step', step_accuracy, total_steps)

        total_steps += 1

    # Calculate full epoch metrics for validation after each epoch
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    classifier_model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = classifier_model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_epoch_loss = val_loss / val_size
    val_epoch_accuracy = val_correct / val_total

    print(f"Validation after Epoch [{epoch}]: Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_accuracy:.4f}")
    writer.add_scalar('Loss/Validation', val_epoch_loss, epoch)
    writer.add_scalar('Accuracy/Validation', val_epoch_accuracy, epoch)

    # Scheduler step and checkpointing continue as before
    scheduler.step()

    checkpoint_path = checkpoint_dir / f'model_epoch_{epoch}.pth'
    torch.save(classifier_model.state_dict(), checkpoint_path)

    # Save best model
    if val_epoch_accuracy > best_val_accuracy:
        best_val_accuracy = val_epoch_accuracy
        best_model_path = checkpoint_dir / 'best_model.pth'
        torch.save(classifier_model.state_dict(), best_model_path)
        print(f"Best model updated at epoch {epoch} with accuracy {best_val_accuracy:.4f}")

writer.close()
