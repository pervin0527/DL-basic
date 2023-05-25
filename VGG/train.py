import torch
import torchvision.models as models
import torchvision.transforms as transforms

from torch import nn 
from tqdm import tqdm
from vgg import VGG
from load_data import CustomDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from preprocessing import get_mean_rgb, get_std_rgb, Scale_Jitter


def train(dataloader, model, loss_fn, optimizer):
    current_lr = optimizer.param_groups[0]["lr"]
    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_correct = 0, 0
        pbar = tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1} / {epochs}', unit='step')

        for iter_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            ## prediction & loss
            pred = model(X)
            loss = loss_fn(pred, y)

            ## Backprop
            optimizer.zero_grad() ## 각 epoch마다 gradient를 측정하기 위해 0으로 초기화한다.
            loss.backward() ## 현재 loss값에 대한 backpropagation을 시작한다.
            optimizer.step() ## parameter를 update한다.

            _, pred = torch.max(pred, 1)
            _, y = torch.max(y, 1)
            correct = (pred == y).sum().item()

            epoch_loss += loss.item() * X.size(0)
            epoch_correct += correct

            pbar.set_postfix({"Loss" : loss.item(), "Acc" : correct / X.size(0)})
            pbar.update(1)

        pbar.close()
        epoch_loss /= len(dataloader.dataset)
        epoch_correct /= len(dataloader.dataset)
        print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_correct:.4f}")

        valid_loss = valid(valid_dataloader, model, loss_fn)
        if optimizer.param_groups[0]['lr'] != current_lr:
            early_stop_patience -=1
            if early_stop_patience == 0:
                break

    torch.save(model.state_dict(), save_path)
    print(f"{model_name} is saved {save_path}")


def valid(dataloader, model, loss_fn):
    model.eval()
    valid_loss, valid_correct = 0, 0
    with torch.no_grad():
        for iter_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y).item()

            valid_loss += loss * X.size(0)
            _, pred = torch.max(pred, 1)
            _, y = torch.max(y, 1)        
            valid_correct += (pred == y).sum().item()

    valid_loss /= len(dataloader.dataset)
    valid_correct /= len(dataloader.dataset)
    print(f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_correct:.4f} \n")
    scheduler.step(valid_loss)

    return valid_loss


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    ## Hyper-parameters
    model_name = "vgg13"
    epochs = 100
    batch_size = 64
    weight_decay = 0.0005
    learning_rate = 1e-2
    early_stop_patience = 5

    ## Dir
    dataset_path = "/home/pervinco/Datasets/sports_ball"
    save_path = f"/home/pervinco/Models/VGG/{model_name}.pth"
    load_path = f"/home/pervinco/Models/VGG/vgg11.pth"

    ## Dataset Processing
    calc_mean = False
    if calc_mean:
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

    ## Define Dataloader
    train_dataset = CustomDataset(dataset_path, "train", train_transform)
    valid_dataset = CustomDataset(dataset_path, "test", train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    ## dataloader : totals / batch size, dataset : total
    print(f"Total train data : {len(train_dataloader.dataset)}, step numbers : {len(train_dataloader)}")
    print(f"Total test data : {len(valid_dataloader.dataset)}, step numbers : {len(valid_dataloader)}")

    ## build model
    classes = train_dataset.get_classes()
    model = VGG(model_name=model_name, num_classes=len(classes), init_weights=True).to(device)
    print(model)

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
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    """
    weight load하고 필요한 layer만 초기화하는 거 만들기.
    """
    state_dict = torch.load(load_path)
    print(state_dict)

    
    layers = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
    with torch.no_grad():
        model.layers[1].weight.copy_(state_dict["features.0.weight"])

    # train(train_dataloader, model, loss_fn, optimizer)