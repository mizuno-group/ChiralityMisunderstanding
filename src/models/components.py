import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
from torch.autograd import Variable


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class FactorizedEmbeddings(nn.Module):
    def __init__(self, d_model, vocab, emb_dim):
        super().__init__()
        self.lut = nn.Embedding(vocab, emb_dim)
        self.lut1 = nn.Linear(emb_dim, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.lut1(self.lut(x)) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, d_model, warmup, optimizer, factor=1):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.d_model = d_model
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.d_model ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def save_params(self, file_name):
        pd.Series({"_step":self._step, "warmup":self.warmup, "factor":self.factor, 
            "d_model":self.d_model, "_rate":self._rate}, name='value', dtype=float).to_csv(file_name)
    
    def load_params(self, file_name):
        params = pd.read_csv(file_name, index_col=0)['value']
        self._step = int(params['_step'])
        self.warmup = int(params['warmup'])
        self.factor = params['factor']
        self.d_model = int(params['d_model'])
        self._rate = params['_rate']

    def zero_grad(self):
        self.optimizer.zero_grad()

        
class ArbitraryOpt:
    "Optim wrapper that implements rate."
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class KappaLoss(nn.Module):
    def __init__(self, num_classes, weightage="quadratic", epsilon=1e-6, add_softmax=True):
        super().__init__()

        if weightage not in ("linear", "quadratic"):
            raise ValueError("Unknown kappa weighting type.")

        self.weightage = weightage
        self.num_classes = num_classes
        self.epsilon = epsilon

        
        self.label_vec = nn.parameter.Parameter(torch.arange(num_classes, dtype=torch.float),
            requires_grad=False)


        row_mat = self.label_vec.expand(num_classes, num_classes)
        col_mat = row_mat.T
        if weightage == "linear":
            self.weight_mat = nn.parameter.Parameter(torch.abs(col_mat - row_mat), 
            requires_grad=False)
        else:
            self.weight_mat = nn.parameter.Parameter((col_mat - row_mat) ** 2,
            requires_grad=False)
        if add_softmax:
            self.softmax = nn.Softmax(dim=-1)
        else:
            self.softmax = nn.Identity()


    def forward(self, y_pred, y_true):
        """
        Parameters
        ----------
        y_pred: torch.tensor of torch.float [*batch_size, num_classes]
            predict proba or decision function if self.softmax is True
        y_true: torch.tensor of torch.long [*batch_size]
            target labels.
        
        Returns
        -------
        loss: torch.tensor of torch.float
            Loss computed.
        
        """
        batch_size = y_true.shape
        y_true = F.one_hot(y_true, num_classes=self.num_classes).to(torch.float)
        y_pred = self.softmax(y_pred)
        cat_labels = torch.matmul(y_true, self.label_vec)
        cat_label_mat = cat_labels.unsqueeze(-1).expand(-1, self.num_classes)
        row_label_mat = self.label_vec.expand(*batch_size, -1)
        if self.weightage == "linear":
            weight = torch.abs(cat_label_mat - row_label_mat)
        else:
            weight = (cat_label_mat - row_label_mat) ** 2

        numerator = torch.sum(weight * y_pred)
        label_dist = torch.sum(y_true, axis=0)
        pred_dist = torch.sum(y_pred, axis=0)
        w_pred_dist = torch.matmul(self.weight_mat, pred_dist)
        denominator = torch.sum(torch.dot(w_pred_dist, label_dist))

        denominator /= batch_size.numel()
        loss = numerator / denominator
        return torch.log(loss+self.epsilon)

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data * norm

class WeightSharingEncoder(nn.Module):
    r"""A variant of TransformerEncoder whose layers share the common weight.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layer = encoder_layer
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask, src_key_padding_mask = None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for i in range(self.num_layers):
            output = self.layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class WeightSharingDecoder(nn.Module):
    r"""A variant of TransformerEncoder whose layers share the common weight.

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layer = decoder_layer
        self.num_layers = num_layers
        self.norm = norm


    def forward(self, tgt, memory, tgt_mask = None,
                memory_mask = None, tgt_key_padding_mask = None,
                memory_key_padding_mask = None):
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for i in range(self.num_layers):
            output = self.layer(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class StartMeanEndPooler(nn.Module):
    def __init__(self, end_token, pad_token):
        super().__init__()
        self.end_token = end_token
        self.pad_token = pad_token

    def forward(self, x, src):
        """
        Parameters
        ----------
        x: torch.tensor of torch.float [max_len, batch_size, d_model]
            Output of base model.
        src: torch.tensor of torch.log [batch_size, max_len]
    
        Returns
        -------
        out: torch.tensor of torch.float [batch_size, d_model*3]
            Concatenated First x, Last (with end_token) x, mean x.
        """
        begin = x[0]
        end = (x*(src.transpose(0, 1) == self.end_token).unsqueeze(2)).sum(dim=0)
        pad_mask = (src.transpose(0, 1) != self.pad_token).to(torch.int)
        mean = (x*pad_mask.unsqueeze(2)).sum(dim=0) / pad_mask.sum(axis=0).unsqueeze(1)
        return torch.cat([begin, mean, end], dim=1)

class StartMeanEndMaxPooler(nn.Module):
    def __init__(self, end_token, pad_token):
        super().__init__()
        self.end_token = end_token
        self.pad_token = pad_token

    def forward(self, x, src):
        """
        Parameters
        ----------
        x: torch.tensor of torch.float [max_len, batch_size, d_model]
            Output of base model.
        src: torch.tensor of torch.log [batch_size, max_len]
    
        Returns
        -------
        out: torch.tensor of torch.float [batch_size, d_model*5]
            Concatenated First x, Last (with end_token) x, mean x, max x.
        """
        begin = x[0]
        src = src.transpose(0, 1)
        end = (x*(src == self.end_token).unsqueeze(2)).sum(dim=0)
        pad_mask = (src != self.pad_token).to(torch.int)
        mean = (x*pad_mask.unsqueeze(2)).sum(dim=0) / pad_mask.sum(dim=0).unsqueeze(1)
        nonpad_mask_bool = src == self.pad_token
        x[nonpad_mask_bool] = float('-inf')
        max_ = torch.max(x, dim=0)[0]
        return torch.cat([begin, mean, end, max_], dim=1)

class StartMeanEndMaxMinStdPooler(nn.Module):
    def __init__(self, end_token, pad_token):
        super().__init__()
        self.end_token = end_token
        self.pad_token = pad_token

    def forward(self, x, src):
        """
        Parameters
        ----------
        x: torch.tensor of torch.float [max_len, batch_size, d_model]
            Output of base model.
        src: torch.tensor of torch.log [batch_size, max_len]
    
        Returns
        -------
        out: torch.tensor of torch.float [batch_size, d_model*5]
            Concatenated First x, Last (with end_token) x, mean x, max x, min x, std x.
        """
        begin = x[0]
        src = src.transpose(0, 1)
        end = (x*(src == self.end_token).unsqueeze(2)).sum(dim=0)
        pad_mask = (src != self.pad_token).to(torch.int)
        mean = (x*pad_mask.unsqueeze(2)).sum(dim=0) / pad_mask.sum(dim=0).unsqueeze(1)
        nonpad_mask_bool = src == self.pad_token
        x[nonpad_mask_bool] = float('-inf')
        max_ = torch.max(x, dim=0)[0]
        x[nonpad_mask_bool] = float('inf')
        min_ = torch.min(x, dim=0)[0]
        std = torch.std(x, dim=0, unbiased=False)
        return torch.cat([begin, mean, end, max_, min_, std], dim=1)

class StartPooler(nn.Module):
    def __init__(self, end_token, pad_token):
        super().__init__()

    def forward(self, x, src):
        """
        Parameters
        ----------
        x: torch.tensor of torch.float [max_len, batch_size, d_model]
            Output of base model.
        src: torch.tensor of torch.log [batch_size, max_len]
    
        Returns
        -------
        out: torch.tensor of torch.float [batch_size, d_model]
            First x.
        """
        return x[0]

class MeanPooler(nn.Module):
    def __init__(self, end_token, pad_token):
        super().__init__()
        self.pad_token = pad_token

    def forward(self, x, src):
        """
        Parameters
        ----------
        x: torch.tensor of torch.float [max_len, batch_size, d_model]
            Output of base model.
        src: torch.tensor of torch.log [batch_size, max_len]
    
        Returns
        -------
        mean: torch.tensor of torch.float [batch_size, d_model*3]
            Mean x.
        """
        src = src.transpose(0, 1)
        pad_mask = (src != self.pad_token).to(torch.int)
        return (x*pad_mask.unsqueeze(2)).sum(dim=0) / pad_mask.sum(dim=0).unsqueeze(1)

class FFGenerator(nn.Module):
    def __init__(self, in_dim, hidden_dims, n_task, activation, dropout):
        super().__init__()
        hidden_dims =[in_dim]+list(hidden_dims)
        self.linears_hidden = nn.ModuleList([
            nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i+1])
         for i in range(len(hidden_dims)-1)])
        self.linear_out = nn.Linear(in_features=hidden_dims[-1], out_features=n_task)
        self.n_task = n_task
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for linear in self.linears_hidden:
            x = self.dropout(self.activation(linear(x)))
        x = self.linear_out(x)
        return x