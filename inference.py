import argparse
import os
import torch
import pandas as pd
from utils.pythia.model import *
from utils.pythia.pdb_utils import *
from utils.model import Pythia_PPI, get_torch_model
from utils.dataset import inference_process
import warnings
from Bio import BiopythonWarning
from Bio.SeqUtils import seq1

warnings.simplefilter("ignore", BiopythonWarning)
alphabet = "ACDEFGHIKLMNPQRSTVWYX"

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('pdb_path', type=str, help='Path to a single PDB file')
    argparser.add_argument('--path_to_pretrained_model_weights', type=str, default='./utils/pythia/pythia-p.pt', help='Path to pretrained model weights')
    argparser.add_argument('--path_to_model_weights', type=str, default='./train_model', help='Path to model weights folder')
    argparser.add_argument('--out_folder', type=str, default='./output', help='Folder to output results')
    argparser.add_argument('--model_name', type=str, default='ppi_affinity', help='Model name: ppi_affinity, protein_stability')   
    argparser.add_argument('--device', type=str, default='cuda')
    args = argparser.parse_args() 

    pdb_name = os.path.basename(args.pdb_path).split(".pdb")[0]
    model_weight_path = os.path.join(args.path_to_model_weights, f'{args.model_name}.pth')
    output_file = os.path.join(args.out_folder, f"{pdb_name}.csv")

    os.makedirs(args.out_folder, exist_ok=True)

    pythia_ppi = Pythia_PPI(get_torch_model(args.path_to_pretrained_model_weights, args.device))
    model_dict = torch.load(model_weight_path, map_location=args.device)
    pythia_ppi.load_state_dict(model_dict)
    pythia_ppi.to(args.device)
    pythia_ppi.eval()

    feats, info = inference_process(args.pdb_path, pdb_name, device=args.device)
    with torch.no_grad():
        pred_1, pred_2 = pythia_ppi(feats["wt_id"], feats["mt_id"], feats["node_in"], feats["edge_in"])
    
    if args.model_name == 'ppi_affinity':
        predictions = pred_1.cpu().tolist()
    else: 
        predictions = pred_2.cpu().tolist()
    results = {
        f"{info['wt'][i]}_{info['chain'][i]}_{info['mt_position'][i]}_{info['mt'][i]}": predictions[i]
        for i in range(len(info["pdb_id"]))
    }
    df = pd.DataFrame(list(results.items()), columns=['mutation', 'ddG_pred'])
    df.to_csv(output_file, index=False)
