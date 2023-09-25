import sys, os
os.environ.setdefault('PROJECT_DIR', '.')
sys.path.append(os.environ['PROJECT_DIR'])
import argparse
import concurrent.futures as cf
import pandas as pd
from src.data.buckets import MakeBuckets
from src.features.tokenizer import Tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", required=True, help="Input csv file path (.csv)")
    parser.add_argument("--input_col", default='random', help="Column of input SMILES")
    parser.add_argument("--target_col", default='canonical', help="Column of target SMILES")
    parser.add_argument("--output", required=True, help="Output data directory")
    parser.add_argument("--vocabulary", default="data/smiles_vocabulary.txt")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of processes to use.")
    args = parser.parse_args()

    # create tokenizer
    with open(args.vocabulary, 'r') as f:
        vocabulary = f.read().splitlines()
    tokenizer = Tokenizer(vocabulary)

    # Load SMILES
    print("Loading SMILES...")
    smiles_df = pd.read_csv(args.smiles, usecols=[args.input_col, args.target_col])

    # Tokenize SMILES for each column
    print(f"Tokenizing input SMILES...")
    with cf.ProcessPoolExecutor(max_workers=args.max_workers) as e:
        input_tokens = list(e.map(tokenizer.tokenize, smiles_df[args.input_col].values, chunksize=10000))
    print(f"Tokenizing target SMILES...")
    with cf.ProcessPoolExecutor(max_workers=args.max_workers) as e:
        target_tokens = list(e.map(tokenizer.tokenize, smiles_df[args.target_col].values, chunksize=10000))
    
    # Make bucket dataset and save
    buckets = MakeBuckets([input_tokens, target_tokens], tokenizer.pad_token, 
        [],[], datas=[], min_len=10, max_len=120, min_bucket_step=10, min_n_token=500,
        residue='include')
    buckets.shuffle_batch(n_tokens=12500)
    buckets.save(args.output)
    del buckets, input_tokens, target_tokens

if __name__ == '__main__':
    main()

