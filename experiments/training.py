import sys, os
import argparse
import pickle
import datetime
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, cohen_kappa_score
PROJECT_DIR = os.environ["PROJECT_DIR"] if "PROJECT_DIR" in os.environ \
    else ".."
RESULT_DIR = os.path.join(PROJECT_DIR, "training_results")
sys.path += [PROJECT_DIR]

from src.models.model_whole import prepare_model, add_model_args
from src.models.components import NoamOpt, LabelSmoothing, FFGenerator, \
    StartPooler, MeanPooler, KappaLoss
from src.data.buckets import Buckets

def load_data(file):
    print(f"Loading {file} ...", end='', flush=True)
    ext = file.split(".")[-1]
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

def load_tokenizer(tokenizer_file):
    with open(tokenizer_file, mode='rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

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
        if args.lr is None:
            args.lr = 0.01
        optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    elif args.optimizer.lower() == "adamw":
        if args.lr is None:
            args.lr = 0.01
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

AUX_METRICS = {
    'weighted_kappa': lambda y_true, y_pred: cohen_kappa_score(y_true, y_pred.argmax(axis=1), weights='quadratic'),
    'independent_kappa': lambda y_true, y_pred: cohen_kappa_score(y_true, y_pred.argmax(axis=1), weights=None),
    'ROC_AUC_OVO': lambda y_true, y_pred: roc_auc_score(y_true, y_pred, multi_class='ovo')
}
def aux_val_model(args, model, generator, pooler, tokenizer, val_data, DEVICE):
    preds = []
    feats = []
    with torch.no_grad():
        for batch_src, batch_feat in tqdm(val_data.iterbatches(iterfeatures=args.aux_iterfeatures)):
            batch_src = torch.tensor(batch_src, dtype=torch.long, device=DEVICE)
            memory, src_padding_mask = model.encode(batch_src, pad_token=tokenizer.pad_token)
            pred = F.softmax(generator(pooler(memory, batch_src)), dim=-1)
            feats.append(batch_feat)
            preds.append(pred.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        feats = np.concatenate(feats, axis=0)
        scores = {name:metric(feats, preds) for name, metric in AUX_METRICS.items()}
    return scores

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"DEVICE: {DEVICE}", flush=True)

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--studyname", default="default_study")
    parser.add_argument("--pretrained_model_file")
    parser.add_argument("--tokenizer_file", default='../preprocess/result/tokenizer.pkl')
    parser.add_argument("--train_file", nargs='+', default=["../preprocess/result/train"])
    parser.add_argument("--val_file", nargs='*', default=["../preprocess/result/val"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", type=bool, default=True)
    add_model_args(parser)
    parser.add_argument("--optimizer", default="warmup_adam")
    parser.add_argument("--lr", default=None)
    parser.add_argument("--optimizer_start_step", default=0)
    parser.add_argument("--warmup", default=None)
    parser.add_argument("--minnstep", type=int, default=0)
    parser.add_argument("--maxnstep", type=int, default=200000)
    parser.add_argument("--minnepoch", type=int, default=0)
    parser.add_argument("--maxnepoch", type=int, default=np.inf)
    parser.add_argument("--minvalperfect",default=1, type=float, help="Abort learing when perfect accuracy of validation set exceeded the specified value.")
    parser.add_argument("--val_file_judge_abort", type=str)
    parser.add_argument("--valstep", type=int, default=2000)
    parser.add_argument("--savestep", type=int, default=2000)
    parser.add_argument("--logstep", type=int, default=1000)
    parser.add_argument("--keepseed", action="store_true")
    parser.add_argument("--saveallopt", action="store_true")
    parser.add_argument("--optstep", type=int, default=2)
    parser.add_argument("--duplicate", default='ask')
    
    parser.add_argument("--aux", action="store_true")
    parser.add_argument("--aux_train_file")
    parser.add_argument("--aux_val_file")
    parser.add_argument("--aux_iterfeatures", nargs=2, default=[0,1], type=int)
    parser.add_argument("--aux_activation", default='relu')
    parser.add_argument("--aux_dropout", default=0.1, type=float)
    parser.add_argument("--aux_pooler", default="start")
    parser.add_argument("--aux_hidden_dims", nargs='*', type=int)
    parser.add_argument("--aux_n_out", default=2, type=int)
    parser.add_argument("--aux_criterion", default='ce')
    parser.add_argument("--aux_loss_ratio", default=1, type=float)
    parser.add_argument("--aux_batch_step", default=2, type=int)

    args = parser.parse_args()


    # make directory
    dt_now = datetime.datetime.now()
    result_dirname_base = RESULT_DIR+f"/{dt_now.year%100:02}{dt_now.month:02}{dt_now.day:02}_{args.studyname}"
    result_dirname = result_dirname_base
    if os.path.exists(result_dirname):
        if args.duplicate == 'ask':
            answer = None
            while answer not in ['y', 'n']:
                answer = input(f"Study '{args.studyname}' already exists. Will you overwrite this study? (y/n)")
            if answer == 'n':
                return
        elif args.duplicate == 'numbering':
            n_exist = 0
            while os.path.exists(result_dirname):
                print(f"{result_dirname} already exists. Study name was changed to ",
                    end='', file=sys.stderr)
                n_exist += 1
                result_dirname = result_dirname_base+str(n_exist)
                print(result_dirname, file=sys.stderr)
        elif args.duplicate == 'overwrite':
            pass
        else:
            raise ValueError(f"Unsupported args.duplicate: {args.duplicate}")
    result_dirname += '/'
    os.makedirs(result_dirname, exist_ok=True)
    os.makedirs(result_dirname+"models/", exist_ok=True)

    # prepare tokenizer, model, optimizer, data
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = not args.deterministic
    tokenizer = load_tokenizer(args.tokenizer_file)
    args.n_tok = tokenizer.n_tok()
    train_data_iterator = data_iterator(args.train_file)
    if type(args.val_file) == str:
        args.val_file = [args.val_file]
    val_datas = [load_data(val_file) for val_file in args.val_file]
    n_val_datas = len(val_datas)
    if args.val_file_judge_abort is None and n_val_datas > 0:
        args.val_file_judge_abort = args.val_file[0]
        assert args.val_file_judge_abort in args.val_file
    model = load_model(args, DEVICE)
    criterion = LabelSmoothing(size=args.n_tok, padding_idx=tokenizer.pad_token, smoothing=0.1)
    optimizer = prepare_optimizer(args, model.parameters())
    with open(result_dirname+"args.pkl", mode='wb') as f:
        pickle.dump(args, f)

    # auxiliary task
    if args.aux:
        aux_train_data = load_data(args.aux_train_file)
        aux_train_iter = aux_train_data.iterbatches(iterfeatures=args.aux_iterfeatures, 
            n_shuffle_epoch=np.inf)
        aux_val_data = load_data(args.aux_val_file)
        
        if args.aux_activation == 'relu':
            aux_activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported type of args.aux_activaion: {args.aux_activation}")
        aux_generator = FFGenerator(in_dim=args.d_model, hidden_dims=args.aux_hidden_dims,
            n_task=args.aux_n_out, activation=aux_activation, dropout=args.aux_dropout)
        aux_generator.to(DEVICE)
        if args.aux_pooler == 'mean':
            aux_pooler = MeanPooler(tokenizer.end_token, tokenizer.pad_token)
        elif args.aux_pooler == 'start':
            aux_pooler = StartPooler(tokenizer.end_token, tokenizer.pad_token)
        else:
            raise ValueError(f"Unsupported type of args.aux_pooler: {args.aux_pooler}")
        aux_pooler.to(DEVICE)
        if args.aux_criterion == 'ce':
            aux_criterion = nn.CrossEntropyLoss(reduction='mean')
        elif args.aux_criterion == 'kappa':
            aux_criterion = KappaLoss(num_classes=args.aux_n_out)
        else:
            raise ValueError(f"Unsupported type of args.aux_criterion: {args.aux_criterion}")
        aux_criterion.to(DEVICE)
    else:
        if args.aux_train_file is not None:
            print("args.aux_train_file was appointed but ignored.")
        if args.aux_val_file is not None:
            print("args.aux_train_file was appointed but ignored.")
        if args.aux_pooler is not None:
            print("args.aux_pooler was appointed but ignored.")

    parameters = model.parameters()
    if args.aux:
        parameters = list(parameters)+list(aux_pooler.parameters())+list(aux_generator.parameters())
    optimizer = prepare_optimizer(args, parameters)

    loss_steps = []
    loss_hist = []
    val_steps = []
    perfect_hist =  {val_file: [] for val_file in args.val_file}
    partial_hist = {val_file: [] for val_file in args.val_file}
    if args.aux:
        aux_loss_hist = []
        aux_loss_steps = []
        aux_scores = pd.DataFrame(columns=AUX_METRICS.keys())
    log_total_tokens = 0
    log_total_loss = 0
    batch_step = 0
    optimizer_step = 0

    # validation at initial weight
    model.eval()
    for val_file, val_data in zip(args.val_file, val_datas):
        if n_val_datas == 1:
            print("Validating...", flush=True)
        else:
            print(f"Validating {val_file}...", flush=True)
        perfect, partial = val_model(model, tokenizer, val_data, DEVICE)
        print(f"Step {optimizer_step:5} {'Validation' if n_val_datas == 1 else val_file} Perfect: {perfect:.3f}, Partial: {partial:.3f}", flush=True)
        perfect_hist[val_file].append(perfect)
        partial_hist[val_file].append(partial)
    val_steps.append(optimizer_step)
    model.train()
    torch.save(model.state_dict(), result_dirname+f"models/0.pth")

    # training
    optimizer.zero_grad()
    print("training started.", flush=True)
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
                if args.aux and batch_step % args.aux_batch_step == 0:
                    aux_batch_src, aux_batch_feat = aux_train_iter.__next__()
                    aux_batch_src = torch.tensor(aux_batch_src, dtype=torch.long, device=DEVICE)
                    aux_batch_feat = torch.tensor(aux_batch_feat, dtype=torch.long, device=DEVICE)
                    memory, src_padding_mask = model.encode(aux_batch_src, pad_token=tokenizer.pad_token)
                    aux_pred = aux_generator(aux_pooler(memory, aux_batch_src))
                    loss_aux = aux_criterion(aux_pred, aux_batch_feat)*args.aux_loss_ratio
                    aux_loss_hist.append(loss_aux.detach().cpu().numpy())
                    aux_loss_steps.append(batch_step)
                    loss += loss_aux

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
                    if n_val_datas == 0:
                        pass
                    else:
                        model.eval()
                        for val_file, val_data in zip(args.val_file, val_datas):
                            if n_val_datas == 1:
                                print("Validating...", flush=True)
                            else:
                                print(f"Validating {val_file}...", flush=True)
                            perfect, partial = val_model(model, tokenizer, val_data, DEVICE)
                            perfect_hist[val_file].append(perfect)
                            partial_hist[val_file].append(partial)
                            print(f"Step {optimizer_step:5} {'Validation' if n_val_datas == 1 else val_file} Perfect: {perfect:.3f}, Partial: {partial:.3f}", flush=True)
                        pd.DataFrame(data=perfect_hist, index=val_steps).to_csv(result_dirname+"perfect_val.csv")
                        pd.DataFrame(data=partial_hist, index=val_steps).to_csv(result_dirname+"partial_val.csv")
                        if args.aux:
                            aux_scores.loc[optimizer_step] = aux_val_model(args, model, 
                                aux_generator, aux_pooler, tokenizer, aux_val_data, DEVICE)
                            for name in aux_scores.columns:
                                print(f"{name}: {aux_scores[name][optimizer_step]}")
                            aux_scores.to_csv(result_dirname+"aux.csv")
                        model.train()

                if optimizer_step % args.savestep == 0:
                    torch.save(model.state_dict(), result_dirname+f"models/{optimizer_step}.pth")
                if (train_data_iterator.epoch < args.minnepoch) or (optimizer_step < args.minnstep) or perfect_hist[args.val_file_judge_abort][-1] < args.minvalperfect:
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
                if args.aux:
                    pd.DataFrame(data={"loss":aux_loss_hist}, index=aux_loss_steps).to_csv(result_dirname+"aux_loss.csv", index_label="batch_step")
                break
    except KeyboardInterrupt:
        torch.save(model.state_dict(), result_dirname+f"models/{optimizer_step}.pth")
        pd.DataFrame(data={"loss":loss_hist}, index=loss_steps).to_csv(result_dirname+"loss.csv", index_label="batch_step")
        if args.aux:
            pd.DataFrame(data={"loss":aux_loss_hist}, index=aux_loss_steps).to_csv(result_dirname+"aux_loss.csv", index_label="batch_step")
        raise KeyboardInterrupt

if __name__ == '__main__':
    main()


