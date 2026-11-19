import re

SWEDISH_WORDS = {
    "och", "att", "det", "som", "en", "på", "är", "av", "för", "med", "till",
    "den", "har", "de", "inte", "om", "ett", "men", "jag", "han", "hon", "vi",
    "ni", "sig", "så", "man", "kan", "ska", "skall", "från", "över", "under",
    "efter", "vid", "då", "nu", "där", "här", "när", "hur", "vad", "vem",
    "vilken", "denna", "detta", "dessa", "någon", "något", "inga", "alla",
    "mycket", "också", "redan", "alltid", "aldrig", "snart", "vara", "är",
    "var", "blev", "blir", "skulle", "måste", "kunna", "människa", "svar",
    "fråga", "säger", "idag", "imorgon", "igår", "svensk", "sverige",
}

ENGLISH_WORDS = {
    "the", "and", "to", "of", "a", "in", "that", "is", "for", "it", "with",
    "as", "on", "are", "be", "this", "have", "from", "or", "by", "not", "but",
    "they", "you", "he", "she", "we", "at", "an", "there", "their", "what",
    "which", "when", "where", "who", "how", "all", "any", "do", "does", "did",
    "will", "would", "can", "could", "should", "so", "if", "about", "into",
    "than", "then", "them", "these", "those", "some", "no", "yes", "my", "your",
    "his", "her", "our", "i", "answer", "question", "was", "were", "been",
}

_TOKEN_RE = re.compile(r"[^\wåäöÅÄÖ]+")


def _tokens(text):
    return [t.lower() for t in _TOKEN_RE.split(text) if t]


def detect(text):
    toks = _tokens(text)
    sv = sum(1 for t in toks if t in SWEDISH_WORDS)
    en = sum(1 for t in toks if t in ENGLISH_WORDS)
    sv_chars = sum(text.count(c) for c in "åäöÅÄÖ")
    sv_score = sv + sv_chars
    if len(toks) == 0:
        label = "empty"
        ratio = 0.0
    elif sv_score > en:
        label = "sv"
        ratio = sv_score / max(en + sv_score, 1)
    elif en > sv_score:
        label = "en"
        ratio = en / max(sv_score + en, 1)
    else:
        label = "mixed"
        ratio = 0.5
    return {"label": label, "sv_tokens": sv, "en_tokens": en, "sv_chars": sv_chars}


def swedish_ratio(text):
    d = detect(text)
    if d["label"] == "sv":
        return 1.0
    if d["label"] == "en":
        return 0.0
    return 0.5


def detect_batch(texts):
    return [detect(t) for t in texts]