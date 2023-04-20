import sys, os
PROJECT_DIR = os.environ['PROJECT_DIR'] if 'PROJECT_DIR' in os.environ \
    else ".."
sys.path.append(PROJECT_DIR)

import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.data.buckets import MakeBuckets
from src.features.tokenizer import Tokenizer
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--result_dir", default="result")
    parser.add_argument("--val_ratio", type=float, default=0.03)
    args = parser.parse_args()
    
    os.makedirs(args.result_dir, exist_ok=True)

    # create tokenizer
    PAD = 0
    START = 1
    END = 2
    UNK = 3
    smiles_voc_set = [f'{i}' for i in range(10)]+['(', ')', '[', ']', ':', '=', '@', '@@', '+', '/', '\\', '.',
        '-', '#', '%', 'c', 'i', 'o', 'n', 'p', 's', 'b',
        'H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    voc2tok = {}
    voc2tok.update({'<pad>':PAD, '<s>':START, '</s>':END})
    voc2tok.update({voc:i+4 for i, voc in enumerate(smiles_voc_set)})

    n_voc = max([tok for tok in voc2tok.values()])+1
    tok2voc = np.ndarray(n_voc, dtype='<U5')
    for voc, tok in voc2tok.items():
        tok2voc[tok] = voc
    tok2voc[UNK] = '<unk>'
    tokenizer = Tokenizer(tok2voc, PAD, START, END, UNK, 2)

    tokenizer = Tokenizer(tok2voc, PAD, START, END, UNK, 2)
    with open(os.path.join(args.result_dir, "tokenizer.pkl"), mode='wb') as f:
        pickle.dump(tokenizer, f)


    smiles_df = pd.read_csv(args.input_file, sep='\t', usecols=["randomized_smiles", "smiles"])
    rstate = np.random.RandomState(seed=0)
    n_comp = len(smiles_df)
    val_mask = rstate.choice(np.arange(n_comp),size=int(n_comp*args.val_ratio), replace=False)
    train_mask = np.setdiff1d(np.arange(n_comp), val_mask)

    tokens_random_train = [tokenizer.tokenize(smiles) for smiles in tqdm(smiles_df['randomized_smiles'].iloc[train_mask])]
    tokens_canonical_train = [tokenizer.tokenize(smiles) for smiles in tqdm(smiles_df['smiles'].iloc[train_mask])]
    buckets = MakeBuckets([tokens_random_train, tokens_canonical_train], tokenizer.pad_token, 
        [],[], datas=[], min_len=10, max_len=120, min_bucket_step=10, min_n_token=500,
        residue='include')
    buckets.shuffle_batch(n_tokens=12500)
    buckets.save(os.path.join(args.result_dir, "train"))
    del buckets, tokens_random_train, tokens_canonical_train
    
    tokens_random_val = [tokenizer.tokenize(smiles) for smiles in smiles_df["randomized_smiles"].iloc[val_mask]]
    tokens_canonical_val = [tokenizer.tokenize(smiles) for smiles in smiles_df["smiles"].iloc[val_mask]]
    buckets = MakeBuckets([tokens_random_val, tokens_canonical_val], tokenizer.pad_token, 
        [],[], datas=[], min_len=10, max_len=120, min_bucket_step=10, min_n_token=500,
        residue='include')
    buckets.shuffle_batch(n_tokens=12500)
    buckets.save(os.path.join(args.result_dir, "val"))
    del buckets, tokens_random_val, tokens_canonical_val


if __name__ == '__main__':
    main()

