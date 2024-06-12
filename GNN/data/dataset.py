import dgl
import torch
import duckdb
import pandas as pd


from rdkit import Chem
from sklearn.preprocessing import LabelEncoder

from torch.utils.data import Dataset

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
    def __init__(self, parquet_path, limit):
        con = duckdb.connect()
        data_0 = con.query(f"""
            SELECT molecule_smiles, protein_name, binds
            FROM parquet_scan('{parquet_path}')
            WHERE binds = 0
            ORDER BY random()
            LIMIT {limit} 
        """).df()

        data_1 = con.query(f"""
            SELECT id, molecule_smiles, protein_name, binds
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
        self.data['protein_name'] = self.label_encoder.fit_transform(self.data['protein_name'])

        self.train_smiles = list(self.data['molecule_smiles'])
        self.train_labels = list(self.data['binds'])
        self.train_proteins = list(self.data['protein_name'])

    def __len__(self):
        return len(self.train_labels)
    
    def __getitem__(self, idx):
        molecule_smiles = self.data.iloc[idx]['molecule_smiles']
        label = self.data.iloc[idx]['binds']
        protein_name = self.data.iloc[idx]['protein_name']
        graph = get_molecular_graph(molecule_smiles)

        return graph, torch.tensor(label, dtype=torch.float), torch.tensor(protein_name, dtype=torch.long)


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
    def __init__(self, parquet_path):
        con = duckdb.connect()
        self.data = con.query(f"""
            SELECT id, molecule_smiles, protein_name
            FROM parquet_scan('{parquet_path}')
        """).df()

        self.label_encoder = LabelEncoder()
        self.data['protein_name'] = self.label_encoder.fit_transform(self.data['protein_name'])

        self.test_ids = list(self.data['id'])
        self.test_smiles = list(self.data['molecule_smiles'])
        self.test_proteins = list(self.data['protein_name'])

    def __len__(self):
        return len(self.test_smiles)

    def __getitem__(self, idx):
        molecule_smiles = self.data.iloc[idx]['molecule_smiles']
        protein_name = self.data.iloc[idx]['protein_name']
        graph = get_molecular_graph(molecule_smiles)

        return graph, torch.tensor(protein_name, dtype=torch.long), self.data.iloc[idx]['id']
    
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