class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """

        self.id_to_word = {
            0: "<PAD>",
            1: "<UNK>",
            2: "<BOS>",
            3: "<EOS>",
        }

        self.word_to_id = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }

        m_set = set()

        for snt in texts:
            rs = snt.lower().split()
            m_set.update(rs)

        for wrd in sorted(m_set):
            idx = len(self.word_to_id)
            self.id_to_word[idx] = wrd
            self.word_to_id[wrd] = idx

        self.vocab_size = len(self.word_to_id)
        

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """

        s = text.lower().split()
        ans = list()
        for wrd in s:
            if wrd in self.word_to_id:
                ans.append(self.word_to_id[wrd])
            else:
                ans.append(self.word_to_id[self.unk_token])

        return ans

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """

        s = list()
        for id in ids:
            if id in self.id_to_word:
                s.append(self.id_to_word[id])
            else:
                s.append(self.unk_token)

        ans = " ".join(s)
        return ans