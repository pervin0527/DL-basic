import torch
import torchvision.models as models
import torchvision.transforms as transforms

from torch import nn 
from tqdm import tqdm
from vgg import VGG11
from load_data import CustomDataset
from torch.utils.data import DataLoader
from preprocessing import get_mean_rgb, get_std_rgb, Scale_Jitter

def train_loop(dataloader, model, loss_fn, optimizer):
    total_loss, total_correct, total_sampels = 0, 0, 0
    for idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        ## prediction & loss
        pred = model(X)
        loss = loss_fn(pred, y)

        ## Backprop
        optimizer.zero_grad() ## 각 epoch마다 gradient를 측정하기 위해 0으로 초기화한다.
        loss.backward() ## 현재 loss값에 대한 backpropagation을 시작한다.
        optimizer.step() ## parameter를 update한다.

        total_loss += loss.item()
        _, pred = torch.max(pred, 1)
        _, y = torch.max(y, 1)        
        total_correct += (pred == y).sum().item()
        total_sampels += y.size(0)

    epoch_loss = total_loss / len(dataloader)
    epoch_accuracy = total_correct / len(dataloader.dataset)
    print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}")



def test_loop(dataloader, model, loss_fn):
    total_loss, total_correct, total_sampels = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            total_loss += loss_fn(pred, y).item()
            _, pred = torch.max(pred, 1)
            _, y = torch.max(y, 1)        
            total_correct += (pred == y).sum().item()
            total_sampels += y.size(0)

    test_loss = total_loss / len(dataloader)
    test_accuracy = total_correct / len(dataloader.dataset)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    ## Hyper-parameters
    epochs = 10
    batch_size = 64
    weight_decay = 0.0005
    learning_rate = 1e-2

    ## Dataset Processing
    calc_mean = True
    if calc_mean:
        dataset_path = "/home/pervinco/Datasets/sports_ball"
        mean_rgb = get_mean_rgb(f"{dataset_path}/train")
        std_rgb = get_std_rgb(f"{dataset_path}/train", mean_rgb)
    else:
        mean_rgb = (0.485, 0.456, 0.406)
        std_rgb = (0.229, 0.224, 0.225)

    print(mean_rgb, std_rgb)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(224, 224), antialias=False),
        transforms.Normalize(mean=mean_rgb, std=std_rgb),
    ])

    train_dataset = CustomDataset(dataset_path, "train", train_transform)
    test_dataset = CustomDataset(dataset_path, "test", train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4)

    ## Load model
    classes = train_dataset.get_classes()
    model = VGG11(num_classes=len(classes), init_weights=True).to(device)

    # model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).to(device)
    # for param in model.parameters():
    #     param.requires_grad = True
    # model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(classes)).to(device)

    ## Loss func & Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate, 
                                momentum=0.9,
                                weight_decay=0.0005)
    
    for epoch in range(epochs):
        print(f"\n----------------------------- Epoch {epoch+1} -----------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
        print("-----------------------------------------------------------------\n")
    print("Done!")