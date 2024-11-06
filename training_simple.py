import os
import time

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from PIL import Image

from torch.utils.tensorboard import SummaryWriter

from simple_vit import ViT


data_dir = r"D:\coco\train2017"
checkpoint_dir = "checkpoints"
log_dir = "logs"

batch_size = 128
patch_size = 16
image_size = 224
dim = 384
depth = 10
heads = 12
mlp_dim = 1536
channels = 3
dim_head = 32
learning_rate = 1e-4
weight_decay = 1e-2
epochs = 50
log_interval = 10
save_interval = 10

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output directories
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# TensorBoard writer
writer = SummaryWriter(log_dir=log_dir)

# Data transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Custom Dataset
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._load_image_paths(self.root_dir)

    def _load_image_paths(self, root_dir):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


# Load dataset
train_dataset = ImageDataset(root_dir=data_dir, transform=transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True
)

# Model initialization
model = ViT(
    image_size=image_size,
    patch_size=patch_size,
    dim=dim,
    depth=depth,
    heads=heads,
    mlp_dim=mlp_dim,
    channels=channels,
    dim_head=dim_head
).to(device)

# Optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Training loop
global_step = 0
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    start_time = time.time()

    for batch_idx, images in enumerate(train_loader):
        images = images.to(device)

        optimizer.zero_grad()

        # Forward pass and loss computation
        loss, _ = model(images)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Logging
        epoch_loss += loss.item()
        writer.add_scalar('Loss/train', loss.item(), global_step)

        if batch_idx % log_interval == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] "
                  f"Step [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}")

        global_step += 1

    # Scheduler step
    scheduler.step()

    # Epoch summary
    avg_loss = epoch_loss / len(train_loader)
    epoch_time = time.time() - start_time
    print(f"Epoch [{epoch + 1}/{epochs}] completed in {epoch_time:.2f}s, "
          f"Average Loss: {avg_loss:.4f}")

    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

writer.close()
