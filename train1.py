import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np
from LeNet5 import LeNet5_RBF, rbf_map_loss  
import matplotlib.pyplot as plt

def load_mnist(batch_size=1):
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((32,32)),
        transforms.ToTensor(),                     
        transforms.Lambda(lambda x: x*255.0)   
    ])

    trainset = MNIST(root="./mnist", train=True, download=True, transform=transform)
    testset  = MNIST(root="./mnist", train=False, download=True, transform=transform)

    trainset.data = trainset.data[:5000]
    trainset.targets = trainset.targets[:5000]

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader  = DataLoader(testset,  batch_size=batch_size, shuffle=False)

    return trainloader, testloader



def evaluate(model, loader, device):
    correct = 0
    total = 0
    confusion = torch.zeros(10,10, dtype=torch.int32)

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmin(dim=1)   
            correct += (pred == y).sum().item()
            total += y.size(0)

            confusion[y.item(), pred.item()] += 1

    return correct/total, confusion



def find_most_confusing(model, loader, device):

    model.eval()
    worst = {i: {"score": -1e9, "true": i, "pred": None, "img": None} for i in range(10)}

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            true_penalty = out[:, y.item()].item()

            wrong_penalties = out.clone()
            wrong_penalties[:, y.item()] = 1e9        
            pred_wrong = wrong_penalties.argmin(dim=1).item()
            wrong_penalty = wrong_penalties[0, pred_wrong].item()

            score = true_penalty - wrong_penalty

            c = y.item()
            if score > worst[c]["score"]:
                worst[c]["score"] = score
                worst[c]["pred"] = pred_wrong
                worst[c]["img"] = x.cpu().squeeze(0)  

    return worst


def train_model_with_eval(prototypes, epochs=20, lr=1e-3):

    trainloader, testloader = load_mnist(batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5_RBF(prototypes).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_errors = []
    test_errors  = []
    confusion_final = None

    for epoch in range(1, epochs+1):
        model.train()
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = rbf_map_loss(out, y)
            loss.backward()
            optimizer.step()

        train_acc, _ = evaluate(model, trainloader, device)
        test_acc, confusion_final = evaluate(model, testloader, device)

        train_errors.append(1 - train_acc)
        test_errors.append(1 - test_acc)

        print(f"Epoch {epoch}/{epochs}  Train Err={1-train_acc:.4f}  Test Err={1-test_acc:.4f}")

    confusing = find_most_confusing(model, testloader, device)
    torch.save(model, "lenet1.pth")

    return train_errors, test_errors, confusion_final, confusing


def plot_errors(train_err, test_err):
    plt.plot(train_err, label="Train Error")
    plt.plot(test_err, label="Test Error")
    plt.xlabel("Epoch")
    plt.ylabel("Error Rate")
    plt.title("Training & Test Error")
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    from LeNet5 import train_lenet5
    prototypes =     train_lenet5(
        data_path="digits updated",    
        digit_path="digits updated",   
        epochs=30,
        batch_size=32,
        lr=1e-3
    )
    train_err, test_err, confusion, confusing = train_model_with_eval(prototypes)


    print("\nConfusion Matrix:")
    print(confusion)

    plot_errors(train_err, test_err)

    print("\nMost confusing examples (test set):")
    for c in range(10):
        print(f"Digit {c} misclassified as {confusing[c]['pred']} (score={confusing[c]['score']:.3f})")