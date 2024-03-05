import sys, os
import argparse
import concurrent.futures as cf
import numpy as np
import pandas as pd
import torch
os.environ.setdefault('PROJECT_DIR', ".")
sys.path.append(os.environ['PROJECT_DIR'])
from src.models.model_whole import prepare_model, add_model_args
from src.features.tokenizer import load_tokenizer
from src.data.buckets import MakeBuckets
def main():

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--input_col", default='random')
    parser.add_argument("--target_col", default='canonical')
    parser.add_argument("--output")
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--vocabulary", default='data/smiles_vocabulary.txt')
    parser.add_argument("--decode_type", default='greedydecode', help='Method of decoding',
        choices=['greedydecode', 'teacherforcing'])
    parser.add_argument("--max_workers", type=int, default=1)
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

    print(f"Loading data...")
    df = pd.read_csv(args.smiles, usecols=[args.input_col, args.target_col])
    print(f"Tokenizing...")
    with cf.ProcessPoolExecutor(max_workers=args.max_workers) as e:
        input_tokens = list(e.map(tokenizer.tokenize, df[args.input_col].values, chunksize=10000))
    with cf.ProcessPoolExecutor(max_workers=args.max_workers) as e:
        target_tokens = list(e.map(tokenizer.tokenize, df[args.target_col].values, chunksize=10000))
    buckets = MakeBuckets([input_tokens, target_tokens], tokenizer.pad_token,  datas=[np.arange(len(df))], min_len=10,
        max_len=120, min_bucket_step=10, min_n_token=500, residue='include')
    buckets.shuffle_batch(n_tokens=12500, n_epoch=1)

    print(f"Evaluating...")
    indices = []
    preds = []
    perfect = partial = 0
    with torch.no_grad():
        for batch_src, batch_tgt, batch_indices in buckets.iterbatches():
            batch_src = torch.tensor(batch_src, dtype=torch.long, device=DEVICE)
            batch_tgt = torch.tensor(batch_tgt, dtype=torch.long, device=DEVICE)
            if args.decode_type == 'greedydecode':
                pred = model.greedy_decode(*model.encode(batch_src, 
                    pad_token=tokenizer.pad_token), batch_tgt.size(1), 
                    start_token=tokenizer.start_token, pad_token=tokenizer.pad_token)
            else:
                pred = torch.argmax(model.generator(model(batch_src, batch_tgt[:,:-1],
                    pad_token=tokenizer.pad_token)), dim=-1)
                
            pad_mask = (batch_tgt != tokenizer.pad_token).to(torch.int)
            perfect += torch.all((batch_tgt*pad_mask) == (pred*pad_mask), dim=1).sum().detach().cpu().numpy()
            partial += (torch.sum((batch_tgt == pred)*pad_mask, dim=1) / torch.sum(pad_mask, dim=1)).sum().detach().cpu().numpy()
            
            indices.append(batch_indices)
            preds += pred.cpu().numpy().tolist()
            del batch_src, batch_tgt, pred, pad_mask
            torch.cuda.empty_cache()
    perfect /= len(buckets)
    partial /= len(buckets)
    print(f"Perfect accuracy: {perfect:.3f}")
    print(f"Partial accuracy: {partial:.3f}")
    
    if args.output is not None:
        out_dir = os.path.dirname(args.output)
        if len(out_dir) > 0:
            os.makedirs(out_dir, exist_ok=True)
        indices = np.concatenate(indices, axis=0)
        with open(args.output, 'w') as f:
            f.write('prediction\n')
            for idx in np.argsort(indices):
                f.write(tokenizer.detokenize(preds[idx]))
                f.write('\n')

if __name__ == '__main__':
    main()