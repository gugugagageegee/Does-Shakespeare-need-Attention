class CharTokenizer:

    def __init__(self, characters=None):
        self.chars = []

        self.stoi = {}
        self.itos = {}

        if characters is not None:
            self.chars = list(characters)
            self._build_vocab()

    def fit(self, text):
        self.chars = sorted(set(text))

        self._build_vocab()

        return self

    def _build_vocab(self):
        self.stoi = {
            char: index
            for index, char in enumerate(self.chars)
        }

        self.itos = {
            index: char
            for index, char in enumerate(self.chars)
        }

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        if not self.stoi:
            raise ValueError("Tokenizer has not been fitted.")

        unknown = set(text) - set(self.stoi)

        if unknown:
            raise ValueError(
                f"Unknown characters: {unknown}"
            )

        return [
            self.stoi[char]
            for char in text
        ]

    def decode(self, token_ids):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        return "".join(
            self.itos[int(index)]
            for index in token_ids
        )

    def to_dict(self):
        return {
            "chars": self.chars
        }

    @classmethod
    def from_dict(cls, state):
        return cls(state["chars"])