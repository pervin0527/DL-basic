import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch

from tqdm import tqdm
from datetime import datetime
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.util import load_config
from models.model import GAT, GCN
from data.dataset import LeashBioDataset, collate_fn, get_protein_sequences, precompute_embeddings

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total = 0
    correct = 0
    total_loss = 0.0
    for batch in tqdm(dataloader, desc='Train', leave=False):
        main_graphs, buildingblock_graphs, labels, proteins = batch
        main_graphs = main_graphs.to(device)
        buildingblock_graphs = [g.to(device) for g in buildingblock_graphs]
        labels = labels.to(device)
        proteins = proteins.to(device)

        optimizer.zero_grad()
        outputs = model(main_graphs, buildingblock_graphs, proteins).squeeze()
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
            main_graphs, buildingblock_graphs, labels, proteins = batch
            main_graphs = main_graphs.to(device)
            buildingblock_graphs = [g.to(device) for g in buildingblock_graphs]
            labels = labels.to(device)
            proteins = proteins.to(device)

            outputs = model(main_graphs, buildingblock_graphs, proteins).squeeze()
            loss = criterion(outputs, labels.float())

            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy

def train_and_evaluate(cfg, save_dir, writer, device):
    ## Dataset & DataLoader
    train_dataset = LeashBioDataset(cfg['train_parquet'], f"{cfg['data_dir']}/precomputed_embeddings.json", cfg['num_train_data'])
    valid_dataset = LeashBioDataset(cfg['valid_parquet'], f"{cfg['data_dir']}/precomputed_embeddings.json", cfg['num_valid_data'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], collate_fn=collate_fn)

    if cfg['debug']:
        smiles, labels, target_protein = train_dataset[0]
        print(smiles, labels, target_protein)

        for batch in train_dataloader:
            main_graphs, buildingblock_graphs, labels, target_protein = batch
            print(main_graphs, buildingblock_graphs, labels, target_protein)
            break

    ## Model & Optimizer & Criterion
    buildingblock_embedding_dim = cfg.get('buildingblock_embedding_dim', 64)  # 기본값 설정

    if cfg['model'] == "GAT":
        model = GAT(initial_node_dim=cfg['num_node_features'], 
                    initial_edge_dim=cfg['num_edge_features'],
                    num_layers=cfg['num_layers'], 
                    num_heads=cfg['num_heads'], 
                    hidden_dim=cfg['hidden_dim'], 
                    drop_prob=cfg['drop_prob'], 
                    protein_embedding_dim=cfg['protein_embedding_dim'],
                    buildingblock_embedding_dim=buildingblock_embedding_dim,
                    readout='sum', 
                    activation=F.relu,
                    mlp_bias=False).to(device)
    elif cfg['model'] == "GCN":
        model = GCN(initial_node_dim=cfg['num_node_features'],
                    initial_edge_dim=cfg['num_edge_features'],
                    num_layers=cfg['num_layers'],
                    hidden_dim=cfg['hidden_dim'],
                    drop_prob=cfg['drop_prob'],
                    protein_embedding_dim=cfg['protein_embedding_dim'],
                    buildingblock_embedding_dim=buildingblock_embedding_dim,
                    readout='sum',
                    activation=F.relu).to(device)
    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg['T_0'], T_mult=cfg['T_mult'], eta_min=cfg['min_lr'])

    ## Training
    early_stop_counter = 0
    best_valid_loss = float('inf')
    for epoch in range(1, cfg['epochs']+1):
        ## Log learning rate
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            break
        writer.add_scalar('Learning_Rate/train', current_lr, epoch)

        print(f"\nEpoch : [{epoch}/{cfg['epochs']}], LR : {current_lr}")
        train_loss, train_accuracy = train(model, train_dataloader, optimizer, criterion, device)
        print(f"Train Loss : {train_loss:.4f}, Train Accuracy : {train_accuracy:.4f}")
        valid_loss, valid_accuracy = valid(model, valid_dataloader, criterion, device)
        print(f"Valid Loss : {valid_loss:.4f}, Valid Accuracy : {valid_accuracy:.4f}")
        scheduler.step()

        ## TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Accuracy/valid', valid_accuracy, epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f"{save_dir}/weights/best.pth")
            early_stop_counter = 0
            print(f"Validation loss improved. Reset early_stop_counter to {early_stop_counter}.")
        else:
            early_stop_counter += 1
            print(f"No improvement in validation loss. Increase early_stop_counter to {early_stop_counter}.")

        if early_stop_counter >= cfg['early_stop_patience']:
            print(f"Early stopping at epoch {epoch} due to no improvement for {cfg['early_stop_patience']} consecutive epochs.")
            break

    # Save the last model
    torch.save(model.state_dict(), f"{save_dir}/weights/last.pth")
    writer.close()


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(f"{cfg['data_dir']}/protein_sequence.json"):
        get_protein_sequences(cfg['uniprot_dicts'], cfg['data_dir'])
        precompute_embeddings(seq_path=f"{cfg['data_dir']}/protein_sequence.json", output_path=f"{cfg['data_dir']}/precomputed_embeddings.json")

    for iteration in range(cfg['n_iter']):
        ## Save path
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        save_dir = os.path.join(cfg['save_dir'], f"{timestamp}_iter_{iteration}")
        os.makedirs(f"{save_dir}/weights", exist_ok=True)
        os.makedirs(f"{save_dir}/logs", exist_ok=True)
        print(f"Save directory: {save_dir}")

        ## Tensorboard
        writer = SummaryWriter(log_dir=f"{save_dir}/logs")

        # Train and evaluate the model
        train_and_evaluate(cfg, save_dir, writer, device)


if __name__ == "__main__":
    config_path = 'config.yaml'
    cfg = load_config(config_path)
    main(cfg)