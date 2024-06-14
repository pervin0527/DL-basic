## kaggle competitions submit -c leash-BELKA -f my_submission.csv -m "My submission"

import os
import torch
import pandas as pd

from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.model import GAT, GCN
from utils.util import load_config
from data.dataset import TestDataset, test_collate_fn

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = TestDataset(cfg['test_parquet'], f"{cfg['data_dir']}/precomputed_embeddings.json")
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], collate_fn=test_collate_fn)

    model = GCN(
        initial_node_dim=cfg['num_node_features'], 
        initial_edge_dim=cfg['num_edge_features'],
        num_layers=cfg['num_layers'], 
        hidden_dim=cfg['hidden_dim'], 
        buildingblock_embedding_dim=cfg['buildingblock_embedding_dim'],
        drop_prob=cfg['drop_prob'], 
        readout='sum', 
        activation=F.relu).to(device)
    model.load_state_dict(torch.load(f"{cfg['ckpt_dir']}/weights/best.pth", map_location=device))
    model.eval()
    
    results = []
    with torch.no_grad():
        for main_graphs, buildingblock_graph_batches, proteins, ids in tqdm(test_dataloader):
            main_graphs, proteins = main_graphs.to(device), proteins.to(device)
            buildingblock_graph_batches = [batch.to(device) for batch in buildingblock_graph_batches]
            outputs = model(main_graphs, buildingblock_graph_batches, proteins)
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