import torch
import torchvision.models as models
import torchvision.transforms as transforms

from torch import nn 
from tqdm import tqdm
from vgg import VGG
from load_data import CustomDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from preprocessing import get_mean_rgb, get_std_rgb, ScaleJitterTransform, data_visualize

def load_torch_weight(pretrained_model_name, train_model_name):   
    if train_model_name == "vgg11":
        pretrained_model = models.vgg11(weights="IMAGENET1K_V1").to(device)
    elif train_model_name == "vgg13":
        pretrained_model = models.vgg13(weights="IMAGENET1K_V1").to(device)
    elif train_model_name == "vgg16":
        pretrained_model = models.vgg16(weights="IMAGENET1K_V1").to(device)
    else:
        pretrained_model = models.vgg19(weights="IMAGENET1K_V1").to(device)
    
    pretrained_model.classifier[6] = nn.Linear(pretrained_model.classifier[6].in_features, len(classes)).to(device)
    
    for param in pretrained_model.parameters():
        param.requires_grad = False
        
    train_model = VGG(model_name=train_model_name, num_classes=len(classes), init_weights=True).to(device)
    pretrained_model_layers = dict(pretrained_model.named_modules())
    train_model_layers = dict(train_model.named_modules())

    ## Conv2d(3, 64), Conv2d(64, 128), Conv2d(128, 256), Conv2d(256, 256)
    layers_to_transfer = {"vgg11" : ["features.0", "features.3", "features.6", "features.8", "classifier.0", "classifier.3", "classifier.6"],
                          "vgg13" : ["features.0", "features.5", "features.10", "features.12", "classifier.0", "classifier.3", "classifier.6"],
                          "vgg16" : ["features.0", "features.5", "features.10", "features.12", "classifier.0", "classifier.3", "classifier.6"],
                          "vgg19" : ["features.0", "features.5", "features.10", "features.12", "classifier.0", "classifier.3", "classifier.6"]}
    
    for train_layer_name, pretrained_layer_name in zip(layers_to_transfer[train_model_name], layers_to_transfer[pretrained_model_name]):
        train_model_layers[train_layer_name].weight.data.copy_(pretrained_model_layers[pretrained_layer_name].weight.data)
        train_model_layers[train_layer_name].bias.data.copy_(pretrained_model_layers[pretrained_layer_name].bias.data)

    check_list = []
    for train_layer_name, pretrained_layer_name in zip(layers_to_transfer[train_model_name], layers_to_transfer[pretrained_model_name]):
        train_layer_params = train_model_layers[train_layer_name].weight
        pretrained_layer_params = pretrained_model_layers[pretrained_layer_name].weight
        check_list.append(torch.allclose(train_layer_params, pretrained_layer_params))

        if all(map(lambda x: x == True, check_list)):
            print("Torch weight is applied \n")
    
        return train_model


def load_weight_apply(pretrained_model_name, train_model_name):
    pretrained_model = VGG(model_name=pretrained_model_name, num_classes=len(classes), init_weights=False).to(device)
    print(f"{pretrained_model_name} \n {pretrained_model}")
    pretrained_model.load_state_dict(torch.load(load_path))
    pretrained_model_layers = dict(pretrained_model.named_modules())

    train_model = VGG(model_name=train_model_name, num_classes=len(classes), init_weights=True).to(device)
    print(f"{train_model_name} \n {train_model}")
    train_model_layers = dict(train_model.named_modules())

    ## Conv2d(3, 64), Conv2d(64, 128), Conv2d(128, 256), Conv2d(256, 256)
    layers_to_transfer = {"vgg11" : ["features.0", "features.3", "features.6", "features.8", "classifier.0", "classifier.3", "classifier.6"],
                          "vgg13" : ["features.0", "features.5", "features.10", "features.12", "classifier.0", "classifier.3", "classifier.6"],
                          "vgg16" : ["features.0", "features.5", "features.10", "features.12", "classifier.0", "classifier.3", "classifier.6"],
                          "vgg19" : ["features.0", "features.5", "features.10", "features.12", "classifier.0", "classifier.3", "classifier.6"]}
    
    for train_layer_name, pretrained_layer_name in zip(layers_to_transfer[train_model_name], layers_to_transfer[pretrained_model_name]):
        train_model_layers[train_layer_name].weight.data.copy_(pretrained_model_layers[pretrained_layer_name].weight.data)
        train_model_layers[train_layer_name].bias.data.copy_(pretrained_model_layers[pretrained_layer_name].bias.data)

    check_list = []
    for train_layer_name, pretrained_layer_name in zip(layers_to_transfer[train_model_name], layers_to_transfer[pretrained_model_name]):
        train_layer_params = train_model_layers[train_layer_name].weight
        pretrained_layer_params = pretrained_model_layers[pretrained_layer_name].weight
        check_list.append(torch.allclose(train_layer_params, pretrained_layer_params))

        if all(map(lambda x: x == True, check_list)):
            print("Pretrained weight is applied \n")
    
        return train_model


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
    valid_accuracy = valid_correct / len(dataloader.dataset)
    print(f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f} \n")

    return valid_loss, valid_accuracy


def train(dataloader, model, loss_fn, optimizer):
    global lr_patience, early_stop_patience
    best_loss = 0
    writer = SummaryWriter(log_dir=log_path)

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0, 0
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

            train_loss += loss.item() * X.size(0)
            train_correct += correct

            pbar.set_postfix({"Loss" : loss.item(), "Acc" : correct / X.size(0)})
            pbar.update(1)

        pbar.close()
        train_loss /= len(dataloader.dataset)
        train_accuracy = train_correct / len(dataloader.dataset)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        ## Validation step
        valid_loss, valid_accuracy = valid(valid_dataloader, model, loss_fn)

        ## Record train & valid & lr log to tensorboard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", valid_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
        writer.add_scalar("Accuracy/Validation", valid_accuracy, epoch)

        writer.add_scalars("Loss", {"train_loss" : train_loss, "valid_loss" : valid_loss}, epoch)
        writer.add_scalars("Accuracy", {"train_accuracy" : train_accuracy, "valid_accuracy" : valid_accuracy}, epoch)

        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)


        ## LR scheduler & Early Stopping
        if epoch == 0:
            best_loss = valid_loss
        else:
            if valid_loss <= best_loss:
                print(f"Valid loss decreased. The minimum valid loss updated {best_loss:.4f} to {valid_loss:.4f}.")
                best_loss = valid_loss
                lr_patience = 10
            else:
                lr_patience -= 1
                print(f"Valid loss did not decrease. patience : {lr_patience} | best : {best_loss:.4f} | current : {valid_loss:.4f}")

                if lr_patience == 0:
                    lr_patience = 10
                    early_stop_patience -= 1
                    new_lr = optimizer.param_groups[0]["lr"] * 0.1

                    print(f"Early Stop patience : {early_stop_patience}, learning rate changed {new_lr * 10} to {new_lr}")
                    optimizer.param_groups[0]['lr'] = new_lr

                if early_stop_patience == 0:
                    print("Early stopping patience is 0. Train stopped.")
                    break

    writer.close()
    torch.save(model.state_dict(), save_path)
    print(f"{train_model}_{dataset_name} is saved {save_path}")


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    ## Hyper-parameters
    train_model = "vgg11"
    pretrain_model = "vgg11"
    use_pretrained = False
    use_torch_weight = True

    epochs = 1000
    batch_size = 128
    img_size = 224
    learning_rate = 1e-2
    weight_decay = 0.0005
    calc_mean = True

    lr_patience = 10
    early_stop_patience = 5

    ## Dir
    root = "/home/pervinco"
    dataset_name = "sports"
    dataset_path = f"{root}/Datasets/{dataset_name}"
    save_path = f"{root}/Models/VGG/{train_model}_{dataset_name}.pth"
    load_path = f"{root}/Models/VGG/{pretrain_model}_{dataset_name}.pth"
    log_path = f"{root}/Models/VGG/{train_model}_{dataset_name}"

    ## Dataset Processing
    if calc_mean:
        mean_rgb = get_mean_rgb(f"{dataset_path}/train")
        std_rgb = get_std_rgb(f"{dataset_path}/train", mean_rgb)
    else:
        mean_rgb = (0.485, 0.456, 0.406)
        std_rgb = (0.229, 0.224, 0.225)

    print(mean_rgb, std_rgb, "\n")
    train_transform = transforms.Compose([
        ScaleJitterTransform(),
        transforms.ToTensor(),
        transforms.Resize(size=(img_size, img_size), antialias=False),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=(-0.5, 0.5)),
        transforms.Normalize(mean=mean_rgb, std=std_rgb),
    ])

    ## Define Dataloader
    train_dataset = CustomDataset(dataset_path, "train", train_transform)
    valid_dataset = CustomDataset(dataset_path, "valid", train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    ## dataloader : totals / batch size, dataset : total
    print(f"Total train data : {len(train_dataloader.dataset)}, step numbers : {len(train_dataloader)}")
    print(f"Total test data : {len(valid_dataloader.dataset)}, step numbers : {len(valid_dataloader)} \n")

    ## data visualize
    # data_visualize(train_dataloader, mean_rgb, std_rgb, img_size)

    ## build model
    classes = train_dataset.get_classes()
    if use_pretrained and not use_torch_weight:
        model = load_weight_apply(pretrain_model, train_model)
    elif use_torch_weight and not use_pretrained:
        model = load_torch_weight(train_model, pretrain_model)
    else:
        model = VGG(model_name=train_model, num_classes=len(classes), init_weights=True).to(device)
        print(f"{train_model} \n {model}")

    ## Loss func & Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate, 
                                momentum=0.9,
                                weight_decay=0.0005)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    train(train_dataloader, model, loss_fn, optimizer)