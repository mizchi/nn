"""
PyTorch ViT reference for MNIST - same architecture as MoonBit implementation.
Config: image=28, patch=7, d_model=64, heads=4, layers=2, d_ff=128, classes=10
SGD (no momentum), lr=0.001, batch_size=64, seed=42
"""
import struct, time, math, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── MNIST raw loader (same files as MoonBit) ──
def load_mnist_images(path):
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read()
    images = torch.frombuffer(bytearray(data), dtype=torch.uint8).float() / 255.0
    return images.view(n, rows * cols)

def load_mnist_labels(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        data = f.read()
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).long()

# ── ViT (manual, matching MoonBit structure exactly) ──
class ViT(nn.Module):
    def __init__(self, image_size=28, patch_size=7, d_model=64,
                 num_heads=4, num_layers=2, d_ff=128, num_classes=10):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size * patch_size
        seq_len = num_patches + 1

        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = num_patches
        self.d_model = d_model

        # Patch embedding
        self.patch_proj = nn.Linear(patch_dim, d_model)
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(TransformerBlock(d_model, num_heads, d_ff))

        # Final LN + classification head
        self.ln_final = nn.LayerNorm(d_model)
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B = x.shape[0]
        ps = self.patch_size
        img = self.image_size
        # Extract patches: [B, 784] -> [B, num_patches, patch_dim]
        x = x.view(B, img // ps, ps, img // ps, ps)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.num_patches, ps * ps)
        # Patch embedding
        x = self.patch_proj(x)
        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        # Add pos embedding
        x = x + self.pos_embedding
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        # Final LN
        x = self.ln_final(x)
        # CLS token -> classification
        cls_out = x[:, 0]
        return self.cls_head(cls_out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        h = self.ln2(x)
        h = self.ff(h)
        x = x + h
        return x

def main():
    limit = 1000
    epochs = 5
    batch_size = 64
    lr = 0.001

    for arg in sys.argv[1:]:
        if arg.startswith("--limit="):
            limit = int(arg.split("=")[1])
        elif arg.startswith("--epochs="):
            epochs = int(arg.split("=")[1])
        elif arg.startswith("--batch-size="):
            batch_size = int(arg.split("=")[1])
        elif arg.startswith("--lr="):
            lr = float(arg.split("=")[1])

    base = "data/mnist"
    train_images = load_mnist_images(f"{base}/train-images-idx3-ubyte")
    train_labels = load_mnist_labels(f"{base}/train-labels-idx1-ubyte")
    test_images = load_mnist_images(f"{base}/t10k-images-idx3-ubyte")
    test_labels = load_mnist_labels(f"{base}/t10k-labels-idx1-ubyte")

    if limit and limit < len(train_images):
        train_images = train_images[:limit]
        train_labels = train_labels[:limit]

    print(f"Train: {len(train_images)} samples, Test: {len(test_images)} samples")
    print(f"Config: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print(f"ViT: image=28, patch=7, d_model=64, heads=4, layers=2, d_ff=128")
    print()

    torch.manual_seed(42)
    model = ViT()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print()

    total_train_time = 0.0
    for epoch in range(epochs):
        model.train()
        # Shuffle
        perm = torch.randperm(len(train_images))
        train_images = train_images[perm]
        train_labels = train_labels[perm]

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        t0 = time.time()
        for offset in range(0, len(train_images), batch_size):
            end = min(offset + batch_size, len(train_images))
            xb = train_images[offset:end]
            yb = train_labels[offset:end]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = end - offset
            epoch_loss += loss.item() * bs
            epoch_correct += (logits.argmax(1) == yb).sum().item()
            epoch_samples += bs

        train_time = time.time() - t0
        total_train_time += train_time
        avg_loss = epoch_loss / epoch_samples
        avg_acc = epoch_correct / epoch_samples

        # Test accuracy
        model.eval()
        with torch.no_grad():
            test_correct = 0
            for offset in range(0, len(test_images), 256):
                end = min(offset + 256, len(test_images))
                xb = test_images[offset:end]
                yb = test_labels[offset:end]
                logits = model(xb)
                test_correct += (logits.argmax(1) == yb).sum().item()
            test_acc = test_correct / len(test_images)

        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} train_acc={avg_acc:.4f} test_acc={test_acc:.4f} time={train_time:.2f}s")

    print(f"\nTotal training time: {total_train_time:.2f}s")

if __name__ == "__main__":
    main()
