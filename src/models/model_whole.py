import argparse
import torch
import torch.nn as nn
from .components import Generator, Embeddings, FactorizedEmbeddings, PositionalEncoding, \
    WeightSharingEncoder, WeightSharingDecoder

class Whole(nn.Module):
    def __init__(self, transformer, emb, generator, d_model, device):
        super().__init__()
        """
        Parameters
        ----------
        transformer: nn.Transformer
            transformer model.
        emb: nn.Module (Similar to nn.Embedding)
            Embedding layer. Positional encoding is not needed to be included.
        d_model: int
            Dimention of transformer.
        generator: nn.Module 
            Generates output of transformer to probability distribution of each tokens.
            Similar to nn.Linear(in_features=d_model, out_features=n_vocab)
        device: str, 'cuda', 'cpu' etc.
            Name of device in which parameters are stored
        """
        self.transformer = transformer
        self.emb = emb
        self.posenc = PositionalEncoding(d_model, dropout=0.1)
        self.generator = generator
        self.device=device

    def encode(self, src, pad_token):
        """
        Parameters
        ----------
        src: torch.tensor of torch.long (batch_size, max_len)
            One-hot-encoded tensor of source
        pad_token: int
            Token representing padding.

        Returns
        -------
        memory: torch.tensor of torch.float (max_len, batch_size, d_model)
            Memory from encoder.
        src_padding_mask: torch.tensor of torch.bool (batch_size, max_len)
            src_padding_mask used for encoder of transformer.
        """
        src_padding_mask = src == pad_token
        src = self.posenc(self.emb(src))
        return self.transformer.encoder(src.transpose(0, 1), mask=None, src_key_padding_mask=src_padding_mask), src_padding_mask

    def decode(self, memory, src_padding_mask, tgt_in, pad_token):
        """
        Parameters
        ----------
        memory: torch.tensor of torch.float (max_len, batch_size, d_model)
            Memory from encoder.
        src_padding_mask: torch.tensor of torch.bool (batch_size, memory_max_len)
            Padding mask for memory. memory where src_padding_mask is True is not attended by decoder.
        tgt_in: torch.tensor of torch.long (batch_size, max_len)
            Input of decoder.
        pad_token: int
            Token representing padding.

        Returns
        -------
        tgt_out: torch.tensor of torch.float (batch_size, max_len-1, d_model)
            Output of decoder.
        """
        tgt_padding_mask = tgt_in == pad_token
        tgt_in = self.posenc(self.emb(tgt_in))
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_in.size(1)).to(self.device)
        return self.transformer.decoder(tgt_in.transpose(0,1), memory, tgt_mask=tgt_mask,
            memory_mask=None, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask).transpose(0,1)

    def generate(self, tgt_out):
        """
        Parameters
        ----------
        tgt_out: torch.tensor of torch.float (batch_size, max_len, d_model)
            Output of decoder.
        
        Returns
        -------
        pred: torch.tensor of torch.long (batch_size, max_len-1)
            Predicted tokens.
        """
        pred = self.generator(tgt_out)
        return torch.argmax(pred, dim=-1)

    def forward(self, src, tgt_in, pad_token):
        """

        Parameters
        ----------
        src: torch.tensor of torch.float (batch_size, src_len)
            Input of encoder.
        tgt_in: torch.tensor of torch.float (batch_size, tgt_len)
           Input of decoder.
        pad_token: int

        Returns
        -------
        tgt_out: torch.tensor of torch.float (batch_size, src_len, d_model)
            generator is not applied to output.
        """
        return self.decode(*self.encode(src, pad_token), tgt_in, pad_token)

    def greedy_decode(self, memory, src_padding_mask, tgt_out_len, start_token, pad_token):
        """
        Parameters
        ----------
        memory: torch.tensor of torch.float (max_len, batch_size, d_model)
            Memory from encoder.
        src_padding_mask: torch.tensor of torch.bool
            Padding mask for memory. memory where src_padding_mask is True is not attended by decoder.
        tgt_out_len: int
            Length of predicted tokens.
        start_token: int
        pad_token: int

        Returns
        -------
        tgt_out: torch.tensor of torch.long (batch_size, max_len)
            Predicted tokens.
        """
        batch_size = memory.size(1)

        pred = torch.full(size=(batch_size, 1), fill_value=start_token, dtype=torch.long, device=self.device)
        for i_len in range(tgt_out_len-1):
            pred = self.generate(self.decode(memory, src_padding_mask, pred, pad_token))
            pred = torch.cat([torch.full(size=(batch_size, 1), fill_value=start_token, dtype=torch.long,
                device=self.device), pred], dim=1)
        return pred

class Whole2(nn.Module):
    def __init__(self, transformer, emb, generator, d_model, device, emb_src=None, emb_tgt=None):
        super().__init__()
        """
        Embedding layer for source/target is splitted compared to Whole.

        Parameters
        ----------
        transformer: nn.Transformer
            transformer model.
        emb: nn.Module (Similar to nn.Embedding)
            Embedding layer. Positional encoding is not needed to be included.
        d_model: int
            Dimention of transformer.
        generator: nn.Module 
            Generates output of transformer to probability distribution of each tokens.
            Similar to nn.Linear(in_features=d_model, out_features=n_vocab)
        device: str, 'cuda', 'cpu' etc.
            Name of device in which parameters are stored
        emb_src: nn.Module (Similar to nn.Embedding)
            If specified, embedding for src is replaced from emb. Positional encoding is not needed to be included.
        emb_tgt: nn.Module (Similar to nn.Embedding)
            If specified, embedding for tgt is replaced from emb. Positional encoding is not needed to be included.
        """
        self.transformer = transformer
        self.emb_src = emb
        self.emb_tgt = emb
        self.posenc = PositionalEncoding(d_model, dropout=0.1)
        self.generator = generator
        self.device=device
        if emb_src is not None:
            self.emb_src = emb_src
        if emb_tgt is not None:
            self.emb_tgt = emb_tgt

    def encode(self, src, pad_token):
        """
        Parameters
        ----------
        src: torch.tensor of torch.long (batch_size, max_len)
            One-hot-encoded tensor of source
        pad_token: int
            Token representing padding.

        Returns
        -------
        memory: torch.tensor of torch.float (max_len, batch_size, d_model)
            Memory from encoder.
        src_padding_mask: torch.tensor of torch.bool (batch_size, max_len)
            src_padding_mask used for encoder of transformer.
        """
        src_padding_mask = src == pad_token
        src = self.posenc(self.emb_src(src))
        return self.transformer.encoder(src.transpose(0, 1), mask=None, src_key_padding_mask=src_padding_mask), src_padding_mask

    def decode(self, memory, src_padding_mask, tgt_in, pad_token):
        """
        Parameters
        ----------
        memory: torch.tensor of torch.float (max_len, batch_size, d_model)
            Memory from encoder.
        src_padding_mask: torch.tensor of torch.bool (batch_size, memory_max_len)
            Padding mask for memory. memory where src_padding_mask is True is not attended by decoder.
        tgt_in: torch.tensor of torch.long (batch_size, max_len)
            Input of decoder.
        pad_token: int
            Token representing padding.

        Returns
        -------
        tgt_out: torch.tensor of torch.float (batch_size, max_len-1, d_model)
            Output of decoder.
        """
        tgt_padding_mask = tgt_in == pad_token
        tgt_in = self.posenc(self.emb_tgt(tgt_in))
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_in.size(1)).to(self.device)
        return self.transformer.decoder(tgt_in.transpose(0,1), memory, tgt_mask=tgt_mask,
            memory_mask=None, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask).transpose(0,1)

    def generate(self, tgt_out):
        """
        Parameters
        ----------
        tgt_out: torch.tensor of torch.float (batch_size, max_len, d_model)
            Output of decoder.
        
        Returns
        -------
        pred: torch.tensor of torch.long (batch_size, max_len-1)
            Predicted tokens.
        """
        pred = self.generator(tgt_out)
        return torch.argmax(pred, dim=-1)

    def forward(self, src, tgt_in, pad_token):
        """

        Parameters
        ----------
        src: torch.tensor of torch.float (batch_size, src_len)
            Input of encoder.
        tgt_in: torch.tensor of torch.float (batch_size, tgt_len)
           Input of decoder.
        pad_token: int

        Returns
        -------
        tgt_out: torch.tensor of torch.float (batch_size, src_len, d_model)
            generator is not applied to output.
        """
        return self.decode(*self.encode(src, pad_token), tgt_in, pad_token)

    def greedy_decode(self, memory, src_padding_mask, tgt_out_len, start_token, pad_token):
        """
        Parameters
        ----------
        memory: torch.tensor of torch.float (max_len, batch_size, d_model)
            Memory from encoder.
        src_padding_mask: torch.tensor of torch.bool
            Padding mask for memory. memory where src_padding_mask is True is not attended by decoder.
        tgt_out_len: int
            Length of predicted tokens.
        start_token: int
        pad_token: int

        Returns
        -------
        tgt_out: torch.tensor of torch.long (batch_size, max_len)
            Predicted tokens.
        """
        batch_size = memory.size(1)

        pred = torch.full(size=(batch_size, 1), fill_value=start_token, dtype=torch.long, device=self.device)
        for i_len in range(tgt_out_len-1):
            pred = self.generate(self.decode(memory, src_padding_mask, pred, pad_token))
            pred = torch.cat([torch.full(size=(batch_size, 1), fill_value=start_token, dtype=torch.long,
                device=self.device), pred], dim=1)
        return pred

def prepare_model(args, DEVICE):
    model_optional_args = {}
    if args.preLN:
        model_optional_args['norm_first'] = True
    if args.weightshare:
        encoder_layer = nn.modules.TransformerEncoderLayer(args.d_model, 
            args.n_head, args.d_feedforward, args.dropout, activation=args.activation,
            **model_optional_args)
        encoder_norm = nn.LayerNorm(args.d_model, eps=args.layer_norm_eps)
        encoder = WeightSharingEncoder(encoder_layer, num_layers=6, norm=encoder_norm)
        decoder_layer = nn.TransformerDecoderLayer(args.d_model, args.n_head,
            args.d_feedforward, args.dropout, activation=args.activation, 
            **model_optional_args)
        decoder_norm = nn.LayerNorm(args.d_model, eps=args.layer_norm_eps)
        decoder = WeightSharingDecoder(decoder_layer, num_layers=6, norm=decoder_norm)
        transformer = nn.Transformer(args.d_model, args.n_head,
            custom_encoder=encoder, custom_decoder=decoder, **model_optional_args)
    else:
        transformer = nn.Transformer(d_model=args.d_model, dim_feedforward=args.d_feedforward,
            nhead=args.n_head, **model_optional_args)
    generator = Generator(args.d_model, args.n_tok)
    if args.splitemb:
        if args.femb:
            emb_src = FactorizedEmbeddings(args.d_model, args.n_tok, emb_dim=args.emb_dim)
            emb_tgt = FactorizedEmbeddings(args.d_model, args.n_tok, emb_dim=args.emb_dim)
        else:
            emb_src = Embeddings(args.d_model, args.n_tok)
            emb_tgt = Embeddings(args.d_model, args.n_tok)
        model = Whole2(transformer, None, generator, args.d_model, DEVICE, emb_src, emb_tgt)
    else:
        if args.femb:
            emb = FactorizedEmbeddings(args.d_model, args.n_tok, emb_dim=args.emb_dim)
        else:
            emb = Embeddings(args.d_model, args.n_tok)
        model = Whole(transformer, emb, generator, args.d_model, device=DEVICE)
    for p in model.parameters():
        if p.dim() > 1:
            if args.weightinit == "glorot_uniform":
                nn.init.xavier_uniform_(p)
            elif args.weightinit == "glorot_normal":
                nn.init.xavier_normal_(p)
            elif args.weightinit == "he_uniform":
                nn.init.kaiming_uniform_(p)
            elif args.weightinit == "he_normal":
                nn.init.kaiming_normal_(p)
    model.to(DEVICE)
    return model

def add_model_args(parser):
    parser.add_argument("--femb",  action="store_true", help="If specified, embedding matrix is factorized (specification in ALBERT).")
    parser.add_argument("--preLN", action="store_true")
    parser.add_argument("--activation", type=str, default='relu', help="Specify either 'relu' or 'gelu'")
    parser.add_argument("--weightshare", action="store_true", help="If specified, weight sharing is applied to Transformer (specification in ALBERT).")
    parser.add_argument("--weightinit", type=str, default='glorot_uniform', help="Specify either 'xavier_uniform', 'xavier_normal', 'he_uniform', 'he_normal'")
    parser.add_argument("--splitemb", action="store_true", help="If specified, embedding layer is not shared for source and target.")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--d_feedforward", type=int, default=2048)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--layer_norm_eps", type=float, default=1e-5)