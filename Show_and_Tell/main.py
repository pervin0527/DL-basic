import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import random
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.utils.data import DataLoader

from model.encoder import Encoder
from model.decoder import Decoder
from data.dataset import CocoDataset, collate_fn

def train(encoder, decoder, dataloader, criterion, optimizer, device):
    encoder.train()
    decoder.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, captions, lengths in tqdm(dataloader, desc="Training"):
        images, captions = images.to(device), captions.to(device) ## torch.Size([32, 3, 224, 224]) torch.Size([32, 28])

        optimizer.zero_grad()
        features = encoder(images)
        outputs = decoder(features, captions[:, :-1]) ## torch.Size([32, 28, 27314])

        outputs = outputs[:, :captions.size(1) - 1, :] ## torch.Size([32, 27, 27314])
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, predicted = outputs.max(2)
        correct += (predicted == captions[:, 1:]).sum().item()
        total += captions[:, 1:].numel()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def valid(encoder, decoder, dataloader, criterion, device):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, captions, lengths in tqdm(dataloader, desc="Validation"):
            images, captions = images.to(device), captions.to(device)

            features = encoder(images)
            outputs = decoder(features, captions[:, :-1])
            outputs = outputs[:, :captions.size(1) - 1, :]

            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions[:, 1:].reshape(-1))
            total_loss += loss.item()

            _, predicted = outputs.max(2)
            correct += (predicted == captions[:, 1:]).sum().item()
            total += captions[:, 1:].numel()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(encoder, decoder, vocab, image, device):
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        feature = encoder(image)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()
        
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.get_itos()[word_id]
            sampled_caption.append(word)
            if word == '<eos>':
                break
        return ' '.join(sampled_caption)


def main():
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = CocoDataset(data_dir=data_dir, ds_type='train', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    vocab = train_dataset.vocab
    vocab_size = len(vocab)
    print(vocab_size)

    valid_dataset = CocoDataset(data_dir, 'val', vocab=vocab, transform=transform)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

    encoder = Encoder(embed_dim).to(device)
    decoder = Decoder(embed_dim, hidden_dim, vocab_size, num_layers).to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    optimizer = torch.optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    best_valid_loss = float('inf')
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train(encoder, decoder, train_dataloader, criterion, optimizer, device)
        valid_loss, valid_acc = valid(encoder, decoder, valid_dataloader, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, f'{save_path}/best.pth')
            print(f"Best model saved at epoch {epoch} with valid_loss: {valid_loss:.4f}")

        if epoch % 5 == 0:
            idx = random.randint(0, len(valid_dataset) - 1)
            image, _ = valid_dataset[idx]
            caption = evaluate(encoder, decoder, vocab, image, device)
            print(f"Sampled caption at epoch {epoch}: {caption}")

        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = '/home/pervinco/Datasets/COCO2017'
    save_path = './weights'
    epochs = 100
    batch_size = 64
    img_size = 224
    num_workers = 4

    embed_dim = 512
    hidden_dim = 512
    num_layers = 1
    learning_rate = 0.001

    main()