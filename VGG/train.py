import torch
from torch import nn 
from vgg import VGG11
from load_data import CustomDataset
from torch.utils.data import DataLoader

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    print(size)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        ## 예측(prediction)과 손실(loss) 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        ## 역전파
        optimizer.zero_grad() ## 각 epoch마다 gradient를 측정하기 위해 0으로 초기화한다.
        loss.backward() ## 현재 loss값에 대한 backpropagation을 시작한다.
        optimizer.step() ## parameter를 update한다.

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    ## Hyper-parameters
    epochs = 5
    batch_size = 32
    weight_decay = 0.0005
    learning_rate = 1e-2

    ## Set Dataloader & model
    train_dataset = CustomDataset("/home/pervinco/Datasets/sports_ball", "train")
    test_dataset = CustomDataset("/home/pervinco/Datasets/sports_ball", "test")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4)

    # for (X, y) in train_dataloader:
    #     print(X.shape)
    #     print(y.shape, y)

    #     break

    classes = train_dataset.get_classes()
    model = VGG11(num_classes=len(classes)).to(device)

    ## Loss func & Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate, 
                                momentum=0.9,
                                weight_decay=0.0005)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")