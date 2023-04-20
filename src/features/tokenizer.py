class Tokenizer:
    def __init__(self, tok2voc, pad_token, start_token, end_token, unk_token, max_token_len):
        self.tok2voc = tok2voc
        self.voc2tok = {}
        for tok, voc in enumerate(tok2voc):
            self.voc2tok[voc] = tok
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        self.unk_token = unk_token
        self.max_token_len = max_token_len
    def tokenize(self, sentence):
        """
        Parameters
        ----------
        sentence: str

        Returns
        -------
        toks: list of int
        """
        sentence_left = sentence
        toks = [self.start_token]
        while len(sentence_left)>0:
            for token_len in range(self.max_token_len, 0, -1):
                if sentence_left[:token_len] in self.voc2tok:
                    toks.append(self.voc2tok[sentence_left[:token_len]])
                    sentence_left = sentence_left[token_len:]
                    break
                if token_len == 1:
                    raise KeyError(sentence_left)
        toks.append(self.end_token)
        return toks

    def detokenize(self, toks):
        """
        Parameters
        ----------
        toks: array_like of int

        Returns
        -------
        sentence: str
            detokenized sentence.
        """
        sentence = ""
        for tok in toks:
            if tok == self.end_token:
                break
            elif tok != self.start_token:
                sentence += self.tok2voc[tok]
        return sentence

    def n_tok(self):
        return len(self.tok2voc)