from transformers.generation import LogitsProcessor

from ..eval.language_id import SWEDISH_WORDS

_SWEDISH_CHARS = "åäöÅÄÖ"


def swedish_token_ids(tokenizer):
    ids = set()
    for tok_id, tok in tokenizer.get_vocab().items():
        s = tok.lstrip("Ġ▁").lower()
        if any(c in tok for c in _SWEDISH_CHARS) or s in SWEDISH_WORDS:
            ids.add(int(tok_id))
    return sorted(ids)


class TokenBiasProcessor(LogitsProcessor):
    def __init__(self, token_ids, bias):
        if not isinstance(token_ids, (list, tuple, set)):
            token_ids = [token_ids]
        self.token_ids = list(token_ids)
        self.bias = float(bias)

    def __call__(self, input_ids, scores):
        for t in self.token_ids:
            scores[:, t] += self.bias
        return scores