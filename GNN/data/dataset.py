import os
import dgl
import json
import torch
import duckdb
import requests
import numpy as np
import pandas as pd

from rdkit import Chem
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer


def get_protein_sequences(uniprot_dicts, output_path):
    def fetch_sequence(uniprot_id):
        url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
        response = requests.get(url)
        if response.status_code == 200:
            response_text = response.text
            lines = response_text.splitlines()
            seq = "".join(lines[1:])
            return seq
        else:
            return None

    protein_seq_dicts = {}
    for protein_name, uniprot_id in uniprot_dicts.items():
        protein_sequence = fetch_sequence(uniprot_id)
        if protein_sequence:
            protein_seq_dicts[protein_name] = protein_sequence
        else:
            print(f"Failed to retrieve sequence for {protein_name} ({uniprot_id})")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    with open(f"{output_path}/protein_sequence.json", 'w') as file:
        json.dump(protein_seq_dicts, file)

    print(f"Protein Sequence \n {protein_seq_dicts}")
    print(f"Protein Sequence Saved at {output_path}/protein_sequence.json \n")


def precompute_embeddings(seq_path, output_path):
    with open(seq_path, 'r') as file:
        protein_seq_dicts = json.load(file)

    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert')
    model = BertModel.from_pretrained('Rostlab/prot_bert')
    
    embeddings = {}
    for protein_name, sequence in protein_seq_dicts.items():
        inputs = tokenizer(sequence, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings[protein_name] = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    
    with open(output_path, 'w') as file:
        json.dump(embeddings, file)


def get_combined_graphs(molecule_smiles, buildingblock_smiles_list):
    main_graph = get_molecular_graph(molecule_smiles)
    buildingblock_graphs = [get_molecular_graph(smiles) for smiles in buildingblock_smiles_list if smiles]
    return main_graph, buildingblock_graphs


ATOM_VOCAB = [
	'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
	'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
	'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
	'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
	'Pt', 'Hg', 'Pb', 'Dy',
	#'Unknown'
] ## 44

HYBRIDIZATION_TYPE = [
	Chem.rdchem.HybridizationType.S,
	Chem.rdchem.HybridizationType.SP,
	Chem.rdchem.HybridizationType.SP2,
	Chem.rdchem.HybridizationType.SP3,
	Chem.rdchem.HybridizationType.SP3D
]

def one_of_k_encoding(x, allowable_set, allow_unk=False):
	if x not in allowable_set:
		if allow_unk:
			x = allowable_set[-1]
		else:
			raise Exception(f'input {x} not in allowable set{allowable_set}!!!')
	return list(map(lambda s: x == s, allowable_set))


def get_atom_feature(atom):
    feature = (
        one_of_k_encoding(atom.GetSymbol(), ATOM_VOCAB)
        + one_of_k_encoding(atom.GetHybridization(), HYBRIDIZATION_TYPE)
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        + one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5])
        + one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
        + [atom.IsInRing()]
        + [atom.GetIsAromatic()]
        + [atom.GetFormalCharge()]  # 추가 특성: 원자의 전하
    )
    feature = np.array(feature, dtype=np.float32)
    return feature


def get_bond_feature(bond):
    bond_type = bond.GetBondType()
    feature = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
        bond.GetBondDir() == Chem.rdchem.BondDir.ENDUPRIGHT  # 추가 특성: 결합 방향성
    ]
    feature = np.array(feature, dtype=np.float32)
    return feature


def get_molecular_graph(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    
    ## 원자 수(노드 수)
    atom_list = molecule.GetAtoms()
    num_atoms = len(atom_list)
    
    ## 원자의 특징(Atom Features)
    atom_features = [get_atom_feature(atom) for atom in atom_list]
    atom_feature_list = torch.tensor(np.array(atom_features), dtype=torch.float32)
    
    ## 연결의 특징(Edge Features)
    src_list, dst_list, bond_features = [], [], []
    bond_list = molecule.GetBonds() ## 분자가 가진 bond들을 구함.
    for bond in bond_list:
        bond_feature = get_bond_feature(bond)
        
        src = bond.GetBeginAtom().GetIdx() ## 결합의 시작 원자의 인덱스
        dst = bond.GetEndAtom().GetIdx() ## 결합의 끝 원자의 인덱스
        
        src_list.append(src)
        dst_list.append(dst)
        bond_features.append(bond_feature)
        
        src_list.append(dst)
        dst_list.append(src)
        bond_features.append(bond_feature)

    bond_feature_list = torch.tensor(np.array(bond_features), dtype=torch.float32)
    
    graph = dgl.graph((src_list, dst_list), num_nodes=num_atoms)
    graph.ndata['h'] = atom_feature_list
    graph.edata['e_ij'] = bond_feature_list

    return graph


class LeashBioDataset(Dataset):
    def __init__(self, parquet_path, embedding_path, limit):
        con = duckdb.connect()
        data_0 = con.query(f"""
            SELECT molecule_smiles, buildingblock1_smiles, buildingblock2_smiles, buildingblock3_smiles, protein_name, binds
            FROM parquet_scan('{parquet_path}')
            WHERE binds = 0
            ORDER BY random()
            LIMIT {limit}
        """).df()

        data_1 = con.query(f"""
            SELECT molecule_smiles, buildingblock1_smiles, buildingblock2_smiles, buildingblock3_smiles, protein_name, binds
            FROM parquet_scan('{parquet_path}')
            WHERE binds = 1
            ORDER BY random()
            LIMIT {limit * 0.1}
        """).df()

        self.data = pd.concat([data_0, data_1])
        binds_0_count = self.data[self.data['binds'] == 0].shape[0]
        binds_1_count = self.data[self.data['binds'] == 1].shape[0]
        print(f"Dataset shape: {self.data.shape}, binds=0 count: {binds_0_count}, binds=1 count: {binds_1_count}")

        with open(embedding_path, 'r') as file:
            self.protein_embeddings = json.load(file)

        self.train_smiles = list(self.data['molecule_smiles'])
        self.building_block_smiles = self.data[['buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles']].values.tolist()
        self.train_labels = list(self.data['binds'])
        self.train_proteins = list(self.data['protein_name'])

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):
        molecule_smiles = self.data.iloc[idx]['molecule_smiles']
        building_block_smiles_list = self.building_block_smiles[idx]
        label = self.data.iloc[idx]['binds']
        protein_name = self.data.iloc[idx]['protein_name']
        
        main_graph, buildingblock_graphs = get_combined_graphs(molecule_smiles, building_block_smiles_list)

        protein_embedding = torch.tensor(self.protein_embeddings[protein_name])

        return main_graph, buildingblock_graphs, torch.tensor(label, dtype=torch.float), protein_embedding


def collate_fn(batch):
    main_graph_list, buildingblock_graph_lists, label_list, protein_list = [], [], [], []

    for item in batch:
        main_graph, buildingblock_graphs, label, protein = item
        main_graph_list.append(main_graph)
        buildingblock_graph_lists.append(buildingblock_graphs)
        label_list.append(label)
        protein_list.append(protein)

    main_graph_batch = dgl.batch(main_graph_list)
    buildingblock_graph_batches = [dgl.batch(graph_list) for graph_list in zip(*buildingblock_graph_lists)]
    label_list = torch.tensor(label_list, dtype=torch.float32)
    protein_list = torch.stack(protein_list)

    return main_graph_batch, buildingblock_graph_batches, label_list, protein_list


class TestDataset(Dataset):
    def __init__(self, parquet_path, embedding_path):
        con = duckdb.connect()
        self.data = con.query(f"""
            SELECT id, molecule_smiles, buildingblock1_smiles, buildingblock2_smiles, buildingblock3_smiles, protein_name
            FROM parquet_scan('{parquet_path}')
        """).df()

        self.test_ids = list(self.data['id'])
        self.test_smiles = list(self.data['molecule_smiles'])
        self.building_block_smiles = self.data[['buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles']].values.tolist()
        self.test_proteins = list(self.data['protein_name'])

        # Load precomputed embeddings
        with open(embedding_path, 'r') as file:
            self.protein_embeddings = json.load(file)

    def __len__(self):
        return len(self.test_smiles)

    def __getitem__(self, idx):
        molecule_smiles = self.data.iloc[idx]['molecule_smiles']
        building_block_smiles_list = self.building_block_smiles[idx]
        protein_name = self.data.iloc[idx]['protein_name']
        main_graph, buildingblock_graphs = get_combined_graphs(molecule_smiles, building_block_smiles_list)

        # Load precomputed protein embedding
        protein_embedding = torch.tensor(self.protein_embeddings[protein_name])

        return main_graph, buildingblock_graphs, protein_embedding, self.data.iloc[idx]['id']


def test_collate_fn(batch):
    main_graph_list, buildingblock_graph_lists, protein_list, id_list = [], [], [], []
    for item in batch:
        main_graph, buildingblock_graphs, protein, id_ = item
        main_graph_list.append(main_graph)
        buildingblock_graph_lists.append(buildingblock_graphs)
        protein_list.append(protein)
        id_list.append(id_)

    main_graph_batch = dgl.batch(main_graph_list)
    buildingblock_graph_batches = [dgl.batch(graph_list) for graph_list in zip(*buildingblock_graph_lists)]
    protein_list = torch.stack(protein_list)
    id_list = torch.tensor(id_list)

    return main_graph_batch, buildingblock_graph_batches, protein_list, id_list
