
def load_tokenizer(vocabulary_file):
    with open(vocabulary_file, 'r') as f:
        vocabulary = f.read().splitlines()
    tokenizer = Tokenizer(vocabulary)
    return tokenizer

class Tokenizer:
    def __init__(self, vocabulary):
        """
        Parameters
        ----------
        vocabulary: array_like of str
            List of words to recognize.
        
        
        """
        self.tok2voc = ['<pad>', '<start>', '<pad>', '<unk>'] + list(vocabulary)
        self.voc2tok = {}
        for tok, voc in enumerate(vocabulary, 4):
            self.voc2tok[voc] = tok
        self.pad_token = 0
        self.start_token = 1
        self.end_token = 2
        self.unk_token = 3
        self.max_token_len = max([len(voc) for voc in vocabulary])
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
        if toks[0] != self.start_token:
            raise ValueError(f"Invalid token sequence: First token is not <start>.")
        for tok in toks[1:]:
            if tok == self.end_token:
                break
            elif tok != self.start_token:
                sentence += self.tok2voc[tok]
            else:
                raise ValueError(f"Start token appeared inside the sequence.")
        return sentence

    def n_tok(self):
        return len(self.tok2voc)