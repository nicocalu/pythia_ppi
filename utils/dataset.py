import torch
import pickle
import random
import pandas as pd
import os
from Bio.SeqUtils import seq1

from torch.utils.data import  DataLoader
from utils.pythia.model import * 
from utils.pythia.pdb_utils import *

alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
non_standard_residue_substitutions = {
    '2AS':'ASP', '3AH':'HIS', '5HP':'GLU', 'ACL':'ARG', 'AGM':'ARG', 'AIB':'ALA', 'ALM':'ALA', 'ALO':'THR', 'ALY':'LYS', 'ARM':'ARG',
    'ASA':'ASP', 'ASB':'ASP', 'ASK':'ASP', 'ASL':'ASP', 'ASQ':'ASP', 'AYA':'ALA', 'BCS':'CYS', 'BHD':'ASP', 'BMT':'THR', 'BNN':'ALA',
    'BUC':'CYS', 'BUG':'LEU', 'C5C':'CYS', 'C6C':'CYS', 'CAS':'CYS', 'CCS':'CYS', 'CEA':'CYS', 'CGU':'GLU', 'CHG':'ALA', 'CLE':'LEU', 'CME':'CYS',
    'CSD':'ALA', 'CSO':'CYS', 'CSP':'CYS', 'CSS':'CYS', 'CSW':'CYS', 'CSX':'CYS', 'CXM':'MET', 'CY1':'CYS', 'CY3':'CYS', 'CYG':'CYS',
    'CYM':'CYS', 'CYQ':'CYS', 'DAH':'PHE', 'DAL':'ALA', 'DAR':'ARG', 'DAS':'ASP', 'DCY':'CYS', 'DGL':'GLU', 'DGN':'GLN', 'DHA':'ALA',
    'DHI':'HIS', 'DIL':'ILE', 'DIV':'VAL', 'DLE':'LEU', 'DLY':'LYS', 'DNP':'ALA', 'DPN':'PHE', 'DPR':'PRO', 'DSN':'SER', 'DSP':'ASP',
    'DTH':'THR', 'DTR':'TRP', 'DTY':'TYR', 'DVA':'VAL', 'EFC':'CYS', 'FLA':'ALA', 'FME':'MET', 'GGL':'GLU', 'GL3':'GLY', 'GLZ':'GLY',
    'GMA':'GLU', 'GSC':'GLY', 'HAC':'ALA', 'HAR':'ARG', 'HIC':'HIS', 'HIP':'HIS', 'HMR':'ARG', 'HPQ':'PHE', 'HTR':'TRP', 'HYP':'PRO',
    'IAS':'ASP', 'IIL':'ILE', 'IYR':'TYR', 'KCX':'LYS', 'LLP':'LYS', 'LLY':'LYS', 'LTR':'TRP', 'LYM':'LYS', 'LYZ':'LYS', 'MAA':'ALA', 'MEN':'ASN',
    'MHS':'HIS', 'MIS':'SER', 'MLE':'LEU', 'MPQ':'GLY', 'MSA':'GLY', 'MSE':'MET', 'MVA':'VAL', 'NEM':'HIS', 'NEP':'HIS', 'NLE':'LEU',
    'NLN':'LEU', 'NLP':'LEU', 'NMC':'GLY', 'OAS':'SER', 'OCS':'CYS', 'OMT':'MET', 'PAQ':'TYR', 'PCA':'GLU', 'PEC':'CYS', 'PHI':'PHE',
    'PHL':'PHE', 'PR3':'CYS', 'PRR':'ALA', 'PTR':'TYR', 'PYX':'CYS', 'SAC':'SER', 'SAR':'GLY', 'SCH':'CYS', 'SCS':'CYS', 'SCY':'CYS',
    'SEL':'SER', 'SEP':'SER', 'SET':'SER', 'SHC':'CYS', 'SHR':'LYS', 'SMC':'CYS', 'SOC':'CYS', 'STY':'TYR', 'SVA':'SER', 'TIH':'ALA',
    'TPL':'TRP', 'TPO':'THR', 'TPQ':'ALA', 'TRG':'LYS', 'TRO':'TRP', 'TYB':'TYR', 'TYI':'TYR', 'TYQ':'TYR', 'TYS':'TYR', 'TYY':'TYR', 
    'MLY':'LYS', 'M3L':'LYS', 'CMT':'CYS'
}
class Datasets(torch.utils.data.Dataset):
    def __init__(self, csv_path, pdb_dir, val_fold, split, feature_path="None"):
        self.df = pd.read_csv(csv_path)
        self.entries = {
            "train":[],
            "val": []
        }
        if os.path.exists(feature_path):
            with open(feature_path, 'rb') as file:
                self.feature = pickle.load(file)
        else:
            self.feature = {}
            for _, row in self.df.iterrows():
                pdb_id = row['pdb_id']
                wt = row['wt']
                mt = row['mt']
                mt_position = row['mt_position']
                ddG = row['ddG']
                pdbid_identify = row['pdbid_identify']

                wt_id, mt_id, node_in, edge_in = extarct_feature(pdb_id, pdb_dir, wt, mt, mt_position)

                self.feature[pdbid_identify] = {
                    'pdbid_identify': pdbid_identify,
                    'ddG': ddG,
                    'wt_id': wt_id,
                    'mt_id': mt_id,
                    'node_in': node_in,
                    'edge_in': edge_in
                    }
            with open(feature_path, 'wb') as file: 
                pickle.dump(self.feature, file)
        for _,row in self.df.iterrows():
            if row['protein_level_group'] == val_fold:
                self.entries['val'].append(row['pdbid_identify'])
            else: 
                self.entries['train'].append(row['pdbid_identify'])
        self.dataset = self.entries[split]
        if split == 'train': random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset) 
    
    def __getitem__(self, index):

        pdbid_identify = self.dataset[index]
        sample_feature = self.feature[pdbid_identify]

        return sample_feature
    
def extarct_feature(pdb_id, pdb_dir, wt, mt, mt_position, device='cuda'):
    pdb_path = os.path.join(pdb_dir, '{}.pdb'.format(pdb_id))
    protbb, _ = read_pdb_to_protbb(pdb_path, return_chain_dict=True)
    seq_index = protbb.seq.detach().numpy()
   
    # print('________')
    # seq = ''
    # for x in seq_index:
    #     seq += alphabet[int(x[0])]
    # print(pdb_id, wt, mt_position, mt)
    # print(len(seq_index))
    # print(seq)
    # print(seq_index[int(mt_position)][0])
    # print(alphabet.index(wt))

    assert alphabet.index(wt) == int(seq_index[int(mt_position)][0])

    node, edge, seq = get_neighbor(protbb, noise_level=0, mask=False)
    protbb.seq[int(mt_position)] = alphabet.index(mt)
    node_mt, edge, seq = get_neighbor(protbb, noise_level=0, mask=False)
    
    node = torch.stack([node[:,int(mt_position),:], node_mt[:,int(mt_position),:]],dim=1).unsqueeze(0)
    edge = torch.stack([edge[:,int(mt_position),:], edge[:,int(mt_position),:]],dim=1).unsqueeze(0)
    
    wt_index = alphabet.index(wt)
    mt_index = alphabet.index(mt)

    wt_id = torch.nn.functional.one_hot(torch.tensor(wt_index).long(), num_classes=21)
    mt_id = torch.nn.functional.one_hot(torch.tensor(mt_index).long(), num_classes=21)

    # node: [1, 32, 2, 28]
    # batch_size = node.shape[0]
    node_in = torch.cat([node[:,:,0,:], node[:,:,1,:]], dim=0).transpose(0,1).to(device)
    edge_in = torch.cat([edge[:,:,0,:], edge[:,:,1,:]], dim=0).transpose(0,1).to(device)
    
    return wt_id, mt_id, node_in, edge_in

class CombineDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_sizes = [0] + [len(d) for d in datasets] # train:[0, 3459, 3436] val:[0, 617]
        self.total_size = sum(len(d) for d in datasets)
    def __len__(self):
        return self.total_size 
    def __getitem__(self, index):
        dataset_index = 0
        cumulative = self.cumulative_sizes[dataset_index + 1] + self.cumulative_sizes[dataset_index]
        if index >= cumulative:
            dataset_index += 1
        return self.datasets[dataset_index][index - self.cumulative_sizes[dataset_index]]

def train_dataset_dataloader(dataset_names, batch_size, csv_path, pdb_dir, feature_path, val_fold, train=True, shuffle=True):                                                    
    datasets_list = []
    if train:
        split = 'train'
        for dataset_name in dataset_names:
            if dataset_name == 'Skempi':
                skempi_csv_path = csv_path.Skempi
                skempi_pdb_dir = pdb_dir.Skempi
                skempi_feature_path = feature_path.Skempi
                dataset = Datasets(skempi_csv_path, skempi_pdb_dir, val_fold, split, skempi_feature_path)
            elif dataset_name == 'FireProt':
                fireprot_csv_path = csv_path.FireProt
                fireprot_pdb_dir = pdb_dir.FireProt
                fireprot_feature_path = feature_path.FireProt
                dataset = Datasets(fireprot_csv_path, fireprot_pdb_dir, val_fold, split, fireprot_feature_path)
            else:
                raise ValueError(f'Unknown dataset: {dataset_name}')
            datasets_list.append(dataset)
    else: 
        split = 'val'
        skempi_csv_path = csv_path.Skempi
        skempi_pdb_dir = pdb_dir.Skempi
        skempi_feature_path = feature_path.Skempi
        dataset = Datasets(skempi_csv_path, skempi_pdb_dir, val_fold, split, skempi_feature_path)
        datasets_list.append(dataset)
    combined_dataset = CombineDataset(datasets_list)
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def test_dataset_dataloader(batch_size, csv_path, pdb_dir, feature_path, val_fold=0, shuffle=False):
    split = 'val'
    dataset = Datasets(csv_path, pdb_dir, val_fold, split, feature_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def preprocessing(pdb_path, mt_chain, mutation):
    chain_position_residue_dict = {}
    position_residue_dict = {}
    i = 0
    wt_pdb_sequence = ''
    with open(pdb_path, 'r') as file: 
        for line in file: 
            lines = line.rstrip()
            if lines[:6] == "HETATM" and lines[17:17 + 3] == "MSE":
                lines = lines.replace("HETATM", "ATOM  ")
                lines = lines.replace("MSE", "MET")

            if lines[:4] == 'ATOM':
                if lines[17:17+3] in non_standard_residue_substitutions:
                    residue = non_standard_residue_substitutions[lines[17:17+3]]
                else:
                    residue = lines[17:17+3]

                residue = seq1(residue)
                residue_position = lines[22:22+5].rstrip()
                chain = lines[21:22]

                if i == 0: chain_letter=chain

                if chain_letter != chain :
                    chain_position_residue_dict[chain_letter] = position_residue_dict
                    chain_letter = chain
                    position_residue_dict = {}
                else:
                    i = 1
                
                position_residue_dict[residue_position] = residue
    chain_position_residue_dict[chain] = position_residue_dict
    pdb_mutation_position = int(mutation[1:-1])

    length_chain = 0
    j = 0
    for chain in chain_position_residue_dict:
        g = 0
        h = 0
        init = 0
        record = 0
        for position in chain_position_residue_dict[chain]:
            if g == 0: 
                g = int(position) -1
                init = int(position)
            if int(position) - g == 1:
                wt_pdb_sequence += chain_position_residue_dict[chain][position]
            else:
                a = int(position) - g - 1
                wt_pdb_sequence += "" * a
                wt_pdb_sequence += chain_position_residue_dict[chain][position]
                h += 1*a
                if pdb_mutation_position >= int(position):
                    record = h      
            g = int(position)

        if chain == mt_chain and j == 0: 
            pdb_position = length_chain + pdb_mutation_position - init - record
            j += 1

        length_chain += len(chain_position_residue_dict[chain]) 
    assert wt_pdb_sequence[pdb_position] == mutation[0]
    
    wt = mutation[0]
    mt = mutation[-1]
    k = pdb_path.rfind('/')
    pdb_dir = pdb_path[:k+1]
    pdb_id = pdb_path[k+1:].split('.')[0]

    return extarct_feature(pdb_id, pdb_dir, wt, mt, pdb_position)
    

def inference_process(pdb_path, pdb_name, device="cuda"):
    protbb, _ = read_pdb_to_protbb(pdb_path, return_chain_dict=True)
    seq_index = protbb.seq.detach().numpy()
    node, edge, _ = get_neighbor(protbb, noise_level=0, mask=False)
    
    wt_list, mt_list, mt_position_list, res_position_list, pdb_id_list, chain_list = [], [], [], [], [], []
    wt_id_list, mt_id_list, node_in_list, edge_in_list = [], [], [], []

    with open(pdb_path, "r") as f:
        mt_position = -1
        for line in f:
            line = line.rstrip()
            if line[:4] == "ATOM" and line[13:16].rstrip() == "CA":
                mt_position += 1
                wt = seq1(line[17:20])
                chain = line[21:22]
                position = line[22:27].strip()
                wt_ids, mt_ids, node_ins, edge_ins, mutations = inference_extarct_feature(
                    wt, mt_position, seq_index=seq_index, node=node, edge=edge
                )

                wt_id_list.extend(wt_ids)
                mt_id_list.extend(mt_ids)
                node_in_list.extend(node_ins)
                edge_in_list.extend(edge_ins)

                for mt in mutations:
                    wt_list.append(wt)
                    chain_list.append(chain)
                    mt_position_list.append(mt_position)
                    res_position_list.append(position)
                    pdb_id_list.append(pdb_name)
                    mt_list.append(mt)

    wt_ids = torch.stack(wt_id_list).to(device)
    mt_ids = torch.stack(mt_id_list).to(device)
    node_ins = torch.stack(node_in_list).to(device)
    edge_ins = torch.stack(edge_in_list).to(device)
    return (
        {
            "wt_id": wt_ids,
            "mt_id": mt_ids,
            "node_in": node_ins,
            "edge_in": edge_ins,
        },
        {
            "pdb_id": pdb_id_list,
            "wt": wt_list,
            "mt": mt_list,
            "chain": chain_list,
            "res_position": res_position_list,
            "mt_position": mt_position_list,
        },
    )

def inference_extarct_feature(wt, mt_position, seq_index, node, edge, device="cuda"):
    try:
        assert alphabet.index(wt) == int(seq_index[int(mt_position)][0])
    except AssertionError:
        print(f"Assertion failed: wt = {wt}, mt_position = {mt_position}")

    wt_id_list, mt_id_list, node_in_list, edge_in_list, mutations = [], [], [], [], []

    edge = torch.stack(
        [edge[:, int(mt_position), :], edge[:, int(mt_position), :]], dim=1
    ).unsqueeze(0)
    edge_in = (
        torch.cat([edge[:, :, 0, :], edge[:, :, 1, :]], dim=0)
        .transpose(0, 1)
        .to(device)
    )
    wt_index = alphabet.index(wt)
    wt_id = torch.nn.functional.one_hot(torch.tensor(wt_index).long(), num_classes=21)

    for mt in alphabet:
        if mt != wt and mt != "X":
            mutations.append(mt)
            node_mt = node.clone()
            node_r = node.clone()

            node_mt[0, int(mt_position), alphabet.index(wt)] = 0
            node_mt[0, int(mt_position), alphabet.index(mt)] = 1

            node_r = torch.stack(
                [node_r[:, int(mt_position), :], node_mt[:, int(mt_position), :]], dim=1
            ).unsqueeze(0)
            mt_index = alphabet.index(mt)
            mt_id = torch.nn.functional.one_hot(
                torch.tensor(mt_index).long(), num_classes=21
            )
            node_in = (
                torch.cat([node_r[:, :, 0, :], node_r[:, :, 1, :]], dim=0)
                .transpose(0, 1)
                .to(device)
            )

            wt_id_list.append(wt_id)
            mt_id_list.append(mt_id)
            node_in_list.append(node_in)
            edge_in_list.append(edge_in)

    return wt_id_list, mt_id_list, node_in_list, edge_in_list, mutations


