import sys, os
import argparse
import pickle
import datetime
import time
import yaml
import shutil
import numpy as np
import pandas as pd
import torch
os.environ.setdefault('PROJECT_DIR', ".")
sys.path.append(os.environ['PROJECT_DIR'])
RESULT_DIR = os.path.join(os.environ['PROJECT_DIR'], "training_results")
from src.models.model_whole import prepare_model, add_model_args
from src.models.components import NoamOpt, LabelSmoothing
from src.data.buckets import Buckets
from src.features.tokenizer import load_tokenizer

def load_data(file):
    print(f"Loading {file} ...", end='', flush=True)
    data = Buckets.load(file)
    print("loaded.", flush=True)
    return data
    
class data_iterator:
    def __init__(self, files):
        self.epoch = 0
        self.n_file = len(files)
        if self.n_file == 1:
            self.file = load_data(files[0])
        else:
            self.file = None
            self.files = files
    def __iter__(self):
        return self
    def __next__(self):
        self.epoch += 1
        if self.n_file == 1:
            return self.file
        else:
            del self.file
            self.file = load_data(self.files[(self.epoch-1)%self.n_file])
            return self.file

def load_model(args, DEVICE):
    model = prepare_model(args, DEVICE)
    if args.pretrained_model_file is not None:
        print(f"Loading {args.pretrained_model_file} ...")
        model.load_state_dict(torch.load(args.pretrained_model_file))
    return model

def prepare_optimizer(args, parameters):
    if args.optimizer.lower() in ["warmup_adam", "warmup_adamw"]:
        if args.warmup is None:
            args.warmup = 4000
        if args.optimizer.lower() == "warmup_adam":
            base_opt = torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9)
        else:
            base_opt = torch.optim.AdamW(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9)
        optimizer = NoamOpt(args.d_model, warmup=args.warmup, optimizer=base_opt)
        optimizer._step = args.optimizer_start_step
    elif args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    else:
        raise ValueError(f"Unknown type of args.optimizer: {args.optimizer}")
    return optimizer

def val_model(model, tokenizer, val_data, DEVICE):
    perfect = 0
    partial = 0
    for batch_src, batch_tgt in val_data.iterbatches():
        batch_src = torch.tensor(batch_src, dtype=torch.long, device=DEVICE)
        batch_tgt = torch.tensor(batch_tgt, dtype=torch.long, device=DEVICE)

        pred = model.greedy_decode(*model.encode(batch_src, pad_token=tokenizer.pad_token),
            batch_tgt.size(1), start_token=tokenizer.start_token, pad_token=tokenizer.pad_token)
        pad_mask = (batch_tgt != tokenizer.pad_token).to(torch.int)
        perfect += torch.all((batch_tgt*pad_mask) == (pred*pad_mask), dim=1).sum().detach().cpu().numpy()
        partial += (torch.sum((batch_tgt == pred)*pad_mask, dim=1) / torch.sum(pad_mask, dim=1)).sum().detach().cpu().numpy()
        del batch_src, batch_tgt, pred, pad_mask
        torch.cuda.empty_cache()
    perfect /= len(val_data)
    partial /= len(val_data)
    return perfect, partial

def main():

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--studyname", default="default_study")
    parser.add_argument("--duplicate", default='ask', choices=['ask', 'overwrite'])
    parser.add_argument("--pretrained_model_file")
    parser.add_argument("--vocabulary", default='data/smiles_vocabulary.txt')
    parser.add_argument("--train_file", nargs='+')
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", type=bool, default=True)
    add_model_args(parser)
    parser.add_argument("--optimizer", default="warmup_adam")
    parser.add_argument("--lr", default=0.01)
    parser.add_argument("--optimizer_start_step", default=0)
    parser.add_argument("--warmup", default=None)
    parser.add_argument("--minnstep", type=int, default=0)
    parser.add_argument("--maxnstep", type=int, default=200000)
    parser.add_argument("--minnepoch", type=int, default=0)
    parser.add_argument("--maxnepoch", type=int, default=np.inf)
    parser.add_argument("--minvalperfect",default=1, type=float, help="Abort learing when perfect accuracy of validation set exceeded the specified value.")
    parser.add_argument("--valstep", type=int, default=2000)
    parser.add_argument("--savestep", type=int, default=2000)
    parser.add_argument("--logstep", type=int, default=1000)
    parser.add_argument("--keepseed", action="store_true")
    parser.add_argument("--saveallopt", action="store_true")
    parser.add_argument("--optstep", type=int, default=2)
    args = parser.parse_args()

    # make directory
    dt_now = datetime.datetime.now()
    result_dirname = RESULT_DIR+f"/{args.studyname}/"
    if os.path.exists(result_dirname):
        if args.duplicate == 'ask':
            answer = None
            while answer not in ['y', 'n']:
                answer = input(f"Study '{args.studyname}' already exists. Will you overwrite this study? (y/n)")
            if answer == 'n':
                return
        else:
            pass
        shutil.rmtree(result_dirname)
    os.makedirs(result_dirname, exist_ok=True)
    os.makedirs(result_dirname+"models/", exist_ok=True)

    # device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"DEVICE: {DEVICE}", flush=True)

    # prepare tokenizer, model, optimizer, data
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = not args.deterministic
    tokenizer = load_tokenizer(args.vocabulary)
    args.n_tok = tokenizer.n_tok()
    train_data_iterator = data_iterator(args.train_file)
    val_data = load_data(args.val_file)
    model = load_model(args, DEVICE)
    criterion = LabelSmoothing(size=args.n_tok, padding_idx=tokenizer.pad_token, smoothing=0.1)
    optimizer = prepare_optimizer(args, model.parameters())
    with open(result_dirname+"args.yaml", mode='w') as f:
        yaml.dump(vars(args), f)

    loss_steps = []
    loss_hist = []
    val_steps = []
    perfect_hist =  []
    partial_hist = []
    log_total_tokens = 0
    log_total_loss = 0
    batch_step = 0
    optimizer_step = 0

    # validation at initial step
    model.eval()
    print("Validating...", flush=True)
    perfect, partial = val_model(model, tokenizer, val_data, DEVICE)
    print(f"Step {optimizer_step:5} Validation Perfect: {perfect:.3f}, Partial: {partial:.3f}", flush=True)
    perfect_hist.append(perfect)
    partial_hist.append(partial)
    val_steps.append(optimizer_step)
    model.train()
    torch.save(model.state_dict(), result_dirname+f"models/{optimizer_step}.pth")

    # training
    optimizer.zero_grad()
    print("Training started.", flush=True)
    try:
        for train_data in train_data_iterator:
            start = time.time()
            for batch_src, batch_tgt in train_data.iterbatches():
                batch_step += 1
                batch_src = torch.tensor(batch_src, dtype=torch.long, device=DEVICE)
                batch_tgt = torch.tensor(batch_tgt, dtype=torch.long, device=DEVICE)
                tgt_n_tokens = (batch_tgt[:,1:] != tokenizer.pad_token).data.sum()
                pred = model.generator(model(batch_src, batch_tgt[:,:-1], pad_token=tokenizer.pad_token))
                loss = criterion(pred.contiguous().view(-1, pred.size(-1)), batch_tgt[:,1:].contiguous().view(-1)) / tgt_n_tokens

                loss.backward()
                loss = loss.detach().cpu().numpy()
                loss_hist.append(loss)
                loss_steps.append(batch_step)
                loss *= tgt_n_tokens.detach().cpu().numpy()
                log_total_loss += loss
                log_total_tokens += tgt_n_tokens
                del batch_src, batch_tgt, loss
                torch.cuda.empty_cache()
                if (batch_step+1) % args.optstep != 0:
                    continue
                optimizer_step += 1

                optimizer.step()
                optimizer.zero_grad()
                val_start = time.time()

                if optimizer_step % args.logstep == 0:
                    elapsed = time.time()-start
                    loss_per_token = log_total_loss / log_total_tokens
                    print(f"Step {optimizer_step:5} Loss: {loss_per_token:2.2f} Elapsed time: {elapsed:>4.1f}s", flush=True)
                    start = time.time()
                    log_total_loss = 0
                    log_total_tokens = 0

                if optimizer_step % args.valstep == 0:
                    val_steps.append(optimizer_step)
                    model.eval()
                    print(f"Validating ...", flush=True)
                    perfect, partial = val_model(model, tokenizer, val_data, DEVICE)
                    perfect_hist.append(perfect)
                    partial_hist.append(partial)
                    print(f"Step {optimizer_step:5} Validation Perfect: {perfect:.3f}, Partial: {partial:.3f}", flush=True)
                    pd.DataFrame(data=np.array(perfect_hist)[:, np.newaxis], index=val_steps).to_csv(result_dirname+"perfect_val.csv")
                    pd.DataFrame(data=np.array(partial_hist)[:, np.newaxis], index=val_steps).to_csv(result_dirname+"partial_val.csv")
                    model.train()

                if optimizer_step % args.savestep == 0:
                    torch.save(model.state_dict(), result_dirname+f"models/{optimizer_step}.pth")
                if (train_data_iterator.epoch < args.minnepoch) or (optimizer_step < args.minnstep) or perfect_hist[-1] < args.minvalperfect:
                    abort_study = False
                else:
                    abort_study = True
                if optimizer_step >= args.maxnstep:
                    abort_study = True
                if abort_study:
                    break
                val_end = time.time()
                start += val_end - val_start

            if abort_study or train_data_iterator.epoch >= args.maxnepoch:
                torch.save(model.state_dict(), result_dirname+f"models/{optimizer_step}.pth")
                pd.DataFrame(data={"loss":loss_hist}, index=loss_steps).to_csv(result_dirname+"loss.csv", index_label="batch_step")
                break
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Saving model...")
        torch.save(model.state_dict(), result_dirname+f"models/{optimizer_step}.pth")
        pd.DataFrame(data={"loss":loss_hist}, index=loss_steps).to_csv(result_dirname+"loss.csv", index_label="batch_step")
        raise KeyboardInterrupt

if __name__ == '__main__':
    main()


