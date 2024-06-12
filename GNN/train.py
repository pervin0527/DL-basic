import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch

from tqdm import tqdm
from datetime import datetime
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.model import GAT
from utils.util import load_config
from data.utils import generate_datasets
from data.dataset import LeashBioDataset, collate_fn


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total = 0
    correct = 0
    total_loss = 0.0
    for batch in tqdm(dataloader, desc='Train', leave=False):
        graphs, labels, protein_names = batch
        graphs = graphs.to(device)
        labels = labels.to(device)
        protein_names = protein_names.to(device)

        optimizer.zero_grad()
        outputs = model(graphs, protein_names).squeeze()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def valid(model, dataloader, criterion, device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Valid', leave=False):
            graphs, labels, protein_names = batch
            graphs = graphs.to(device)
            labels = labels.to(device)
            protein_names = protein_names.to(device)

            outputs = model(graphs, protein_names).squeeze()
            loss = criterion(outputs, labels.float())

            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Save path
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_dir = os.path.join(cfg['save_dir'], timestamp)
    os.makedirs(f"{save_dir}/weights", exist_ok=True)
    os.makedirs(f"{save_dir}/logs", exist_ok=True)
    print(f"Save directory: {save_dir}")

    ## Tensorboard
    writer = SummaryWriter(log_dir=f"{save_dir}/logs")

    ## Dataset & DataLoader
    train_dataset = LeashBioDataset(cfg['train_parquet'], cfg['num_train_data'])
    valid_dataset = LeashBioDataset(cfg['train_parquet'], cfg['num_valid_data'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], collate_fn=collate_fn)

    if cfg['debug']:
        smiles, labels, target_protein = train_dataset[0]
        print(smiles, labels, target_protein)

        for batch in train_dataloader:
            graphs, labels, target_protein = batch
            print(graphs, labels, target_protein)
            break

    ## Model & Optimizer & Criterion
    num_node_features, num_edge_features, num_proteins = 29, 6, len(cfg['target_proteins'])
    model = GAT(
        initial_node_dim=num_node_features, 
        initial_edge_dim=num_edge_features,
        num_layers=cfg['num_layers'], 
        num_heads=cfg['num_heads'], 
        hidden_dim=cfg['hidden_dim'], 
        drop_prob=cfg['drop_prob'], 
        readout='sum', 
        activation=F.relu,
        mlp_bias=False,
        num_proteins=num_proteins).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
    criterion = torch.nn.BCELoss()

    best_valid_loss = float('inf')
    early_stop_counter = 0

    ## Training
    for epoch in range(1, cfg['epochs']+1):
        print(f"Epoch : [{epoch}/{cfg['epochs']}]")
        train_loss, train_accuracy = train(model, train_dataloader, optimizer, criterion, device)
        print(f"Train Loss : {train_loss:.4f}, Train Accuracy : {train_accuracy:.4f}")
        valid_loss, valid_accuracy = valid(model, valid_dataloader, criterion, device)
        print(f"Valid Loss : {valid_loss:.4f}, Valid Accuracy : {valid_accuracy:.4f}\n")

        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Accuracy/valid', valid_accuracy, epoch)

        # Save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f"{save_dir}/weights/best.pth")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Early stopping
        if early_stop_counter >= cfg['early_stop_patience']:
            print(f"Early stopping at epoch {epoch}")
            break

    # Save the last model
    torch.save(model.state_dict(), f"{save_dir}/weights/last.pth")

    writer.close()

if __name__ == "__main__":
    config_path = 'config.yaml'
    cfg = load_config(config_path)
    main(cfg)
