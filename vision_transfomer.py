import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

np.random.seed(0)
torch.manual_seed(0)


# Multi-head attention block
class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads
    
        # The dimensionality for each head
        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                # Get the linear mappings for the current head
                q_mapping = self.q_mappings(head)
                k_mapping = self.k_mappings(head)
                v_mapping = self.v_mappings(head)

                # Get a subvector of the sequence
                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q = q_mapping(seq)
                k = k_mapping(seq)
                v = v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU,
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )



class MyViT(nn.Module):
    def __init__(self, chw=(1, 28, 28), n_patches=7, hidden_d=8, n_heads=2, n_blocks=2):
        super(MyViT, self).__init__()
        
        self.chw = chw
        self.n_patches = n_patches
        self.hidden_d = hidden_d

        assert chw[1] % n_patches == 0, "Input shape not entirely div by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely div by number of patches"

        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d)) # [1, 8]

        # Create positional embeddings for tokens (7x7 + 1 for class token)
        self.pos_embed = nn.Parameter(torch.tensor(self.get_positional_embeddings(
            self.n_patches ** 2 + 1, self.hidden_d)))
        
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

    def get_positional_embeddings(self, sequence_length, d):
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                if j % 2 == 0:
                    result[i][j] = np.sin(i / (10000 ** (j / d)))
                else:
                    result[i][j] = np.cos(i / (10000 ** ((j - 1) / d)))

        return result

    def patchify(self, images, n_patches):
        n, c, h, w = images.shape

        assert h == w, "Patchify method is implemented for square images only"

        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2).to(images.device)
        patch_size = h // n_patches

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                    patches[idx, i * n_patches + j] = patch.flatten()
        return patches

    def forward(self, images):
        # Creates n_patches x n_patches flattened patches: [N, 49, 16]
        patches =  self.patchify(images, self.n_patches) 
        tokens = self.linear_mapper(patches) # Last dimension are the features [N, 49, 8]

        # Add a classification token to the beginning of the sequence
        tokens_w_class = torch.stack([
            torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))
        ]) # [N, 50, 8]

        n = tokens_w_class.shape[0]
        pos_embed = self.pos_embed.repeat(n, 1, 1) # [N, 50, 8]
        out = tokens_w_class + pos_embed

        for block in self.blocks:
            out = block(out)

        return out


def main():
    # Loading data
    transform = ToTensor()

    train_set = MNIST(
        root="./../datasets", train=True, download=True, transform=transform
    )
    test_set = MNIST(
        root="./../datasets", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device)
    device_status = device_name if torch.cuda.is_available() else ""
    print(f"Using device: {device} - {device_status}")
    model = MyViT(
        (1, 28, 28), n_patches=7, hidden_d=8, n_blocks=2,  n_heads=2, #out_d=10
    ).to(device)
    N_EPOCHS = 5
    LR = 0.005

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1} in training", leave=False
        ):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")


if __name__ == "__main__":
    main()
