import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViT(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            patch_size,
            dim,
            depth,
            heads,
            mlp_dim,
            channels=3,
            dim_head=64
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        self.num_patches_h = image_height // patch_height
        self.num_patches_w = image_width // patch_width
        self.num_patches = self.num_patches_h * self.num_patches_w
        patch_dim = channels * patch_height * patch_width
        num_classes = self.num_patches  # Number of classes equals number of patches

        self.to_patch_embedding = Rearrange(
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=patch_height,
            p2=patch_width,
        )

        self.embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h=self.num_patches_h,
            w=self.num_patches_w,
            dim=dim,
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.linear_head = nn.Linear(dim, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, img, evaluation=False):
        device = img.device

        # Step 1: Patchify the images
        x = self.to_patch_embedding(img)  # Shape: (batch_size, num_patches, patch_dim)

        batch_size, num_patches, patch_dim = x.shape

        if not evaluation:
            # Step 2: Generate random permutations for each image in the batch
            permuted_indices = torch.stack(
                [torch.randperm(num_patches, device=device) for _ in range(batch_size)]
            )  # Shape: (batch_size, num_patches)

            # Step 3: Permute the patches
            x = torch.stack(
                [x[i, permuted_indices[i]] for i in range(batch_size)], dim=0
            )  # Shape: (batch_size, num_patches, patch_dim)

            # Step 4: Store the labels (original positions of each permuted patch)
            labels = permuted_indices  # Shape: (batch_size, num_patches)

        # Step 5: Apply patch embedding and positional embedding
        x = self.embedding(x)  # Shape: (batch_size, num_patches, dim)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        # Step 6: Pass through the transformer
        x = self.transformer(x)  # Shape: (batch_size, num_patches, dim)

        if not evaluation:
            logits = self.linear_head(x)  # Shape: (batch_size, num_patches, num_classes)
            # Step 7: Compute the loss
            loss = self.loss_fn(
                logits.view(-1, self.num_patches), labels.view(-1)
            )  # Flattened to (batch_size * num_patches, num_classes)

            return loss

        return x


if __name__ == '__main__':
    model = ViT(
        image_size=224,
        patch_size=16,
        dim=256,
        depth=4,
        heads=4,
        mlp_dim=512,
        channels=3,
        dim_head=64
    )

    x = torch.rand((10, 3, 224, 224))
    out = model(x)
    print()
