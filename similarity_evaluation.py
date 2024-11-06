import torch
from PIL import Image
from torch.nn.functional import cosine_similarity
from torchvision import transforms
import random
import glob
from simple_vit import ViT
import matplotlib.pyplot as plt
from torch import nn


image_size = 224
patch_size = 16
dim = 384
depth = 10
heads = 12
mlp_dim = 1536
channels = 3
dim_head = 32

device = "cuda"

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

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset images
dataset_path = r"D:\DogsCats\data\train\*"
image_paths = glob.glob(dataset_path)
N = 100

# random.seed(42)
image_paths = random.sample(image_paths, N)

# Prepare images
imageA = Image.open(image_paths[0]).convert('RGB')
imagesB = [Image.open(image_path).convert('RGB') for image_path in image_paths[1:]]

# Convert images to tensors
input_tensor = transform(imageA).unsqueeze(0).to(device)
compare_tensors = torch.stack([transform(image).to(device) for image in imagesB])

# Generate embeddings
with torch.no_grad():
    embeddingA = model(input_tensor, evaluation=True).mean(dim=1)
    embeddingsB = model(compare_tensors, evaluation=True).mean(dim=1)

# Calculate similarities
similarities = cosine_similarity(embeddingA, embeddingsB).squeeze(0)

# Get the top 5 similar images
top_indices = similarities.topk(5).indices

# Display the reference image and top 5 similar images
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.imshow(imageA)
plt.title("Reference Image")
plt.axis('off')

for idx, top_idx in enumerate(top_indices, start=1):
    plt.subplot(2, 3, idx + 1)
    plt.imshow(imagesB[top_idx])
    plt.title(f"Similarity: {similarities[top_idx]:.2f}")
    plt.axis('off')

plt.show()



