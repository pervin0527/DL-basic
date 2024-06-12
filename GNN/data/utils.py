import os
import gc
import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from rdkit import Chem
from multiprocessing import Pool, cpu_count

def process_smiles(smiles):
    unique_atoms = set()
    molecule = Chem.MolFromSmiles(smiles)
    if molecule:
        for atom in molecule.GetAtoms():
            unique_atoms.add(atom.GetSymbol())
    return unique_atoms

def extract_unique_atoms(smiles_list):
    unique_atoms = set()
    
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_smiles, smiles_list), total=len(smiles_list), leave=False))
    
    for result in results:
        unique_atoms.update(result)
    
    return unique_atoms

def generate_datasets(parquet_path, save_path, chunk_size=10000):
    os.makedirs(save_path, exist_ok=True)
    print(save_path)

    all_unique_atoms = set()
    try:
        con = duckdb.connect()
        
        # 전체 데이터 수를 계산하기 위한 쿼리
        total_query = f"""
            SELECT COUNT(*) AS total_count
            FROM parquet_scan('{parquet_path}')
        """
        total_data_count = con.execute(total_query).fetchone()[0]
        print(f"Total data count: {total_data_count}")
        
        offset = 0
        processed_data_count = 0

        while True:
            query = f"""
                SELECT id, molecule_smiles, protein_name, binds
                FROM parquet_scan('{parquet_path}')
                LIMIT {chunk_size} OFFSET {offset}
            """
            
            try:
                data = con.execute(query).fetchdf()
            except Exception as e:
                print(f"Query failed: {e}")
                break

            if data.empty:
                break

            binds_0_count = data[data['binds'] == 0].shape[0]
            binds_1_count = data[data['binds'] == 1].shape[0]
            print(f"Dataset shape : {data.shape}, binds=0 count : {binds_0_count}, binds=1 count : {binds_1_count}")
            
            smiles_list = data['molecule_smiles'].tolist()
            unique_atoms = extract_unique_atoms(smiles_list)
            all_unique_atoms.update(unique_atoms)
            print(f"Unique atoms found so far: {len(all_unique_atoms)}\n")
            
            processed_data_count += data.shape[0]
            remaining_data_count = total_data_count - processed_data_count
            print(f"Processed data count: {processed_data_count}, Remaining data count: {remaining_data_count}")
            
            offset += chunk_size

    finally:
        con.close()

    with open(f"{save_path}/unique_atoms.txt", 'w') as file:
        for atom in all_unique_atoms:
            file.write(f"{atom}\n")
        file.write("\n")
    print(f"Unique Atom List Saved {save_path}/unique_atoms.txt")
