import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class DigitDataset(Dataset):
    def __init__(self, root_dir):
        self.paths = []
        self.labels = []

        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

        for d in range(10):
            folder = os.path.join(root_dir, str(d))
            for f in os.listdir(folder):
                if f.endswith(".png"):
                    self.paths.append(os.path.join(folder, f))
                    self.labels.append(d)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        img = self.transform(img)
        label = torch.tensor(self.labels[idx])
        return img, label


def generate_prototypes_7x12(digit_root):
    prototypes = []

    for d in range(10):
        folder = os.path.join(digit_root, str(d))
        imgs = []

        for f in os.listdir(folder):
            if f.endswith(".png"):
                path = os.path.join(folder, f)
                img = Image.open(path).convert("L")

                img = transforms.ToTensor()(img) 
                img = F.interpolate(img.unsqueeze(0), size=(128,128), mode="bilinear", align_corners=False)
                img = img.squeeze(0)  

                imgs.append(img)

        mean_img = torch.stack(imgs).mean(dim=0).unsqueeze(0) 

        down = F.interpolate(mean_img, size=(7,12), mode="bilinear", align_corners=False)

        binary = torch.where(down > 0.5, torch.tensor(1.0), torch.tensor(-1.0))

        prototypes.append(binary.view(-1)) 

    return torch.stack(prototypes) 




def squash(x, S=2/3):
    return 1.7159 * torch.tanh(S * x)

class LeNet5_RBF(nn.Module):
    def __init__(self, rbf_prototypes):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)   
        self.pool1 = nn.AvgPool2d(2)      
        self.conv2 = nn.Conv2d(6, 16, 5) 
        self.pool2 = nn.AvgPool2d(2) 
        self.conv3 = nn.Conv2d(16, 120, 5) 

        self.fc1 = nn.Linear(120, 84)    

        self.num_classes = 10
        self.register_buffer("rbf_centers", rbf_prototypes)  

    def forward(self, x):
        x = squash(self.conv1(x))
        x = self.pool1(x)

        x = squash(self.conv2(x))
        x = self.pool2(x)

        x = squash(self.conv3(x))
        x = x.view(x.size(0), -1)

        x = squash(self.fc1(x))

        x_expanded = x.unsqueeze(1)                 
        centers = self.rbf_centers.unsqueeze(0)     
        dist2 = ((x_expanded - centers) ** 2).sum(dim=2)
        output = dist2                 
        return output



def rbf_map_loss(output, target):

    y_d = output[torch.arange(len(output)), target] 

    concat = torch.cat([
        torch.full((output.size(0),1), -1.0, device=output.device),
        -output
    ], dim=1)

    log_term = torch.logsumexp(concat, dim=1)
    return (y_d + log_term).mean()



def train_lenet5(data_path, digit_path, epochs=10, batch_size=32, lr=1e-3):

    print("7×12 RBF prototypes")
    prototypes = generate_prototypes_7x12(digit_path)    

    dataset = DigitDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cpu")

    model = LeNet5_RBF(prototypes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            out = model(x)
            loss = rbf_map_loss(out, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}  Loss={total_loss/len(loader):.4f}")
    return prototypes


if __name__ == "__main__":
    train_lenet5(
        data_path="digits updated",  
        digit_path="digits updated",  
        epochs=30,
        batch_size=32,
        lr=1e-3
    )
