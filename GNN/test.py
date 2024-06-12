import os
import torch
import pandas as pd

from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.model import GAT
from utils.util import load_config
from data.dataset import TestDataset, test_collate_fn

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_node_features, num_edge_features, num_proteins = 29, 6, len(cfg['target_proteins'])

    test_dataset = TestDataset(cfg['test_parquet'])
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'], num_workers=4, collate_fn=test_collate_fn)

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
    model.load_state_dict(torch.load(f"{cfg['ckpt_dir']}/weights/best.pth", map_location=device))
    model.eval()
    
    results = []
    with torch.no_grad():
        for graphs, proteins, ids in tqdm(test_dataloader):
            graphs, proteins = graphs.to(device), proteins.to(device)
            outputs = model(graphs, proteins)
            probabilities = torch.sigmoid(outputs).cpu().numpy()

            for id_, prob in zip(ids, probabilities):
                results.append((id_.item(), prob.item()))

    results.sort(key=lambda x: x[0])
    submission_df = pd.DataFrame(results, columns=['id', 'binds'])
    submission_df.to_csv('my_submission.csv', index=False)


if __name__ == "__main__":
    config_path = 'config.yaml'
    cfg = load_config(config_path)
    main(cfg)