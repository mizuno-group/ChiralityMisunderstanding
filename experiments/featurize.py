import sys, os
import argparse
import concurrent.futures as cf
import numpy as np
import pandas as pd
import torch
os.environ.setdefault('PROJECT_DIR', ".")
sys.path.append(os.environ['PROJECT_DIR'])
from src.models.model_whole import prepare_model, add_model_args
from src.models.components import MeanPooler, StartPooler, StartMeanEndMaxPooler, \
    StartMeanEndMaxMinStdPooler
from src.features.tokenizer import load_tokenizer
from src.data.buckets import MakeBuckets

poolers = {
    'mean': MeanPooler,
    'start': StartPooler,
    'startmeanendmax': StartMeanEndMaxPooler,
    'startmeanendmaxminstd': StartMeanEndMaxMinStdPooler
}

def load_model(args, DEVICE):
    model = prepare_model(args, DEVICE)
    if args.pretrained_model_file is not None:
        print(f"Loading {args.pretrained_model_file} ...")
        model.load_state_dict(torch.load(args.pretrained_model_file))
    return model

def main():

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--col", required=True, default='smiles')
    parser.add_argument("--output", required=True, help='path to output csv file.')
    parser.add_argument("--vocabulary", default='data/smiles_vocabulary.txt')
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--pooler", default='mean', help="Name of pooling method. ",
        choices=['mean', 'start', 'startmeanendmax', 'startmeanendmaxminstd'])

    add_model_args(parser)
    args = parser.parse_args()

    # device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"DEVICE: {DEVICE}", flush=True)

    # prepare tokenizer, model, optimizer
    print(f"Loading model...")
    tokenizer = load_tokenizer(args.vocabulary)
    args.n_tok = tokenizer.n_tok()
    model = prepare_model(args, DEVICE)
    model.load_state_dict(torch.load(args.model_file))
    model.eval()

    print(f"Loading model...")
    smiles = pd.read_csv(args.smiles, usecols=[args.col])[args.col].values
    pooler = poolers[args.pooler](tokenizer.end_token, tokenizer.pad_token)
    pooler.to(DEVICE)
    pooler.eval()
    print(f"Tokenizing...")
    with cf.ProcessPoolExecutor(max_workers=args.max_workers) as e:
        tokens = list(e.map(tokenizer.tokenize, smiles, chunksize=10000))
    buckets = MakeBuckets([tokens], tokenizer.pad_token,  datas=[np.arange(len(tokens))], min_len=10,
        max_len=120, min_bucket_step=10, min_n_token=500, residue='include')
    buckets.shuffle_batch(n_tokens=12500, n_epoch=1)

    print(f"Featurizing...")
    indices = []
    features = []
    with torch.no_grad():
        for batch_src, batch_indices in buckets.iterbatches():
            batch_src = torch.tensor(batch_src, device=DEVICE, dtype=torch.long)
            memory = model.encode(batch_src, pad_token=tokenizer.pad_token)[0]
            indices.append(batch_indices)
            
            features.append
            features.append(pooler(memory, batch_src).cpu().numpy())
            del batch_src, memory
            torch.cuda.empty_cache()

    indices = np.concatenate(indices, axis=0)
    features = np.concatenate(features, axis=0)[np.argsort(indices)]
    out_dir = os.path.dirname(args.output)
    if len(out_dir) > 0:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(data=features).to_csv(args.output, index=False)
if __name__ == '__main__':
    main()