from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import mnist
import torch
import numpy as np
import torchvision

 
def test(dataloader,model):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for image, label in dataloader:

            image = image.float()
            output = model(image)
            pred = torch.argmax(output, dim=1)
            correct += (pred == label).sum().item()
            total += 1

    test_accuracy = correct / total                                                                                                                                                                            

    print("test accuracy:", test_accuracy)

 

def main():

    pad=torchvision.transforms.Pad(2,fill=0,padding_mode='constant')

    mnist_test=mnist.MNIST(split="test",transform=pad)

    test_dataloader= DataLoader(mnist_test,batch_size=1,shuffle=False)

    model = torch.load("LeNet1.pth")

    test(test_dataloader,model)

 

if __name__=="__main__":

    main()
