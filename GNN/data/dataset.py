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
from PyBioMed.Pyprotein import PyProtein
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler

def get_protein_sequences(uniprot_dicts, output_path):
    """
    표적 단백질에 대한 sequence 계산.
    """
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


def save_protein_ctd_to_parquet(protein_seq_dicts, output_path):
    """
    표적 단백질에 대한 CTD를 계산하고 저장.
    """
    ctd_features = []
    for protein_name, sequence in protein_seq_dicts.items():
        protein_class = PyProtein(sequence)
        ctd = protein_class.GetCTD()
        ctd = {'protein_name': protein_name, **ctd}
        ctd_features.append(ctd)

    ctd_df = pd.DataFrame(ctd_features)
    ctd_df = ctd_df[['protein_name'] + [col for col in ctd_df.columns if col != 'protein_name']]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ctd_df.to_parquet(f"{output_path}/ctd.parquet", index=False)
    ctd_df.to_csv(f"{output_path}/ctd.csv", index=False)
    print(f"Target Proteins CTD Saved at {output_path}/ctd.parquet \n")


def normalize_ctd(ctd_df):
    min_max_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    polarizability_columns = [col for col in ctd_df.columns if '_Polarizability' in col]
    solvent_accessibility_columns = [col for col in ctd_df.columns if '_SolventAccessibility' in col]
    ctd_df[polarizability_columns] = min_max_scaler.fit_transform(ctd_df[polarizability_columns])
    ctd_df[solvent_accessibility_columns] = min_max_scaler.fit_transform(ctd_df[solvent_accessibility_columns])

    secondary_str_columns = [col for col in ctd_df.columns if '_SecondaryStr' in col]
    hydrophobicity_columns = [col for col in ctd_df.columns if '_Hydrophobicity' in col]
    ctd_df[secondary_str_columns] = standard_scaler.fit_transform(ctd_df[secondary_str_columns])
    ctd_df[hydrophobicity_columns] = standard_scaler.fit_transform(ctd_df[hydrophobicity_columns])
    
    return ctd_df


ATOM_VOCAB = ['C', 'S', 'N', 'Dy', 'I', 'B', 'Br', 'F', 'Si', 'O', 'Cl']

def one_of_k_encoding(x, vocab):
	if x not in vocab:
		x = vocab[-1]
	return list(map(lambda s: float(x==s), vocab))


def get_atom_feature(atom):
    atom_feature = one_of_k_encoding(atom.GetSymbol(), ATOM_VOCAB) ## 11
    atom_feature += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) ## 6
    atom_feature += one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) ## 6
    atom_feature += one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) ## 6
    atom_feature += [atom.GetIsAromatic()]
    return atom_feature


def get_bond_feature(bond):
    bt = bond.GetBondType()
    bond_feature = [
        bt == Chem.rdchem.BondType.SINGLE, ## 단일
        bt == Chem.rdchem.BondType.DOUBLE, ## 이중
        bt == Chem.rdchem.BondType.TRIPLE, ## 삼중
        bt == Chem.rdchem.BondType.AROMATIC, ## 방향족
        bond.GetIsConjugated(), ## 공액
        bond.IsInRing() ## 고리
    ]
    return bond_feature ## 6


def get_molecular_graph(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    
    ## 원자 수(노드 수)
    atom_list = molecule.GetAtoms()
    num_atoms = len(atom_list)
    
    ## 원자의 특징(Atom Features)
    atom_feature_list = torch.tensor([get_atom_feature(atom) for atom in atom_list], dtype=torch.float64)
    
    ## 연결의 특징(Edge Features)
    src_list, dst_list, bond_feature_list = [], [], []
    bond_list = molecule.GetBonds() ## 분자가 가진 bond들을 구함.
    for bond in bond_list:
        bond_feature = get_bond_feature(bond)
        
        src = bond.GetBeginAtom().GetIdx() ## 결합의 시작 원자의 인덱스
        dst = bond.GetEndAtom().GetIdx() ## 결합의 끝 원자의 인덱스
        
        src_list.append(src)
        dst_list.append(dst)
        bond_feature_list.append(bond_feature)
        
        src_list.append(dst)
        dst_list.append(src)
        bond_feature_list.append(bond_feature)

    bond_feature_list = torch.tensor(bond_feature_list, dtype=torch.float64)
    
    graph = dgl.graph((src_list, dst_list), num_nodes=num_atoms)
    graph.ndata['h'] = atom_feature_list
    graph.edata['e_ij'] = bond_feature_list

    return graph


class LeashBioDataset(Dataset):
    def __init__(self, parquet_path, ctd_path, limit):
        con = duckdb.connect()
        data_0 = con.query(f"""
            SELECT molecule_smiles, protein_name, binds
            FROM parquet_scan('{parquet_path}')
            WHERE binds = 0
            ORDER BY random()
            LIMIT {limit} 
        """).df()

        data_1 = con.query(f"""
            SELECT molecule_smiles, protein_name, binds
            FROM parquet_scan('{parquet_path}')
            WHERE binds = 1
            ORDER BY random()
            LIMIT {limit}
        """).df()

        self.data = pd.concat([data_0, data_1])
        binds_0_count = self.data[self.data['binds'] == 0].shape[0]
        binds_1_count = self.data[self.data['binds'] == 1].shape[0]
        print(f"Dataset shape: {self.data.shape}, binds=0 count: {binds_0_count}, binds=1 count: {binds_1_count}")

        self.label_encoder = LabelEncoder()
        self.data['protein_name_encoded'] = self.label_encoder.fit_transform(self.data['protein_name'])
        self.num_proteins = len(self.label_encoder.classes_)

        self.onehot_encoder = OneHotEncoder(sparse_output=False)
        self.protein_onehot = self.onehot_encoder.fit_transform(self.data[['protein_name_encoded']])

        self.train_smiles = list(self.data['molecule_smiles'])
        self.train_labels = list(self.data['binds'])
        self.train_proteins = list(self.data['protein_name_encoded'])

        ctd_df = pd.read_parquet(ctd_path, engine='pyarrow')
        self.ctd_df = normalize_ctd(ctd_df)

        # Create a dictionary for quick lookup
        self.protein_feature_dict = {name: features.drop(columns=['protein_name']).values.flatten()
                                     for name, features in self.ctd_df.groupby('protein_name')}

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):
        molecule_smiles = self.data.iloc[idx]['molecule_smiles']
        label = self.data.iloc[idx]['binds']
        protein_name_encoded = self.data.iloc[idx]['protein_name_encoded']
        graph = get_molecular_graph(molecule_smiles)

        protein_name = self.label_encoder.inverse_transform([protein_name_encoded])[0]
        protein_features = self.protein_feature_dict[protein_name]
        protein_onehot = self.protein_onehot[idx]
        protein_combined = np.concatenate((protein_onehot, protein_features))

        return graph, torch.tensor(label, dtype=torch.float), torch.tensor(protein_combined, dtype=torch.float)


def collate_fn(batch):
    graph_list, label_list, protein_list = [], [], []

    for item in batch:
        graph, label, protein = item
        graph_list.append(graph)
        label_list.append(label)
        protein_list.append(protein)

    graph_list = dgl.batch(graph_list)
    label_list = torch.stack(label_list)
    protein_list = torch.stack(protein_list)

    return graph_list, label_list, protein_list

class TestDataset(Dataset):
    def __init__(self, parquet_path, ctd_path):
        con = duckdb.connect()
        self.data = con.query(f"""
            SELECT id, molecule_smiles, protein_name
            FROM parquet_scan('{parquet_path}')
        """).df()

        self.label_encoder = LabelEncoder()
        self.data['protein_name_encoded'] = self.label_encoder.fit_transform(self.data['protein_name'])

        self.test_ids = list(self.data['id'])
        self.test_smiles = list(self.data['molecule_smiles'])
        self.test_proteins = list(self.data['protein_name_encoded'])

        ctd_df = pd.read_parquet(ctd_path, engine='pyarrow')
        self.ctd_df = normalize_ctd(ctd_df)

        self.onehot_encoder = OneHotEncoder(sparse_output=False)
        self.protein_onehot = self.onehot_encoder.fit_transform(self.data[['protein_name_encoded']])

        self.protein_feature_dict = {name: features.drop(columns=['protein_name']).values.flatten()
                                     for name, features in self.ctd_df.groupby('protein_name')}

    def __len__(self):
        return len(self.test_smiles)

    def __getitem__(self, idx):
        molecule_smiles = self.data.iloc[idx]['molecule_smiles']
        protein_name_encoded = self.data.iloc[idx]['protein_name_encoded']
        graph = get_molecular_graph(molecule_smiles)

        protein_name = self.label_encoder.inverse_transform([protein_name_encoded])[0]
        protein_features = self.protein_feature_dict[protein_name]
        protein_onehot = self.protein_onehot[idx]
        protein_combined = np.concatenate((protein_onehot, protein_features))

        return graph, torch.tensor(protein_combined, dtype=torch.float), self.data.iloc[idx]['id']
    

def test_collate_fn(batch):
    graph_list, protein_list, id_list = [], [], []
    for item in batch:
        graph, protein, id_ = item
        graph_list.append(graph)
        protein_list.append(protein)
        id_list.append(id_)

    graph_list = dgl.batch(graph_list)
    protein_list = torch.stack(protein_list)
    id_list = torch.tensor(id_list)

    return graph_list, protein_list, id_list