import json
from pathlib import Path

V2_PROBES = [
    {"id": "v2-01", "prompt": "I går ", "options": ["läste jag en bok.", "jag läste en bok.", "en bok jag läste."], "correct": 0},
    {"id": "v2-02", "prompt": "Idag ", "options": ["är vädret fint.", "vädret är fint.", "fint är vädret."], "correct": 0},
    {"id": "v2-03", "prompt": "Nu ", "options": ["ska vi äta.", "vi ska äta.", "äta ska vi."], "correct": 0},
    {"id": "v2-04", "prompt": "I morgon ", "options": ["kommer han hit.", "han kommer hit.", "kommer hit han."], "correct": 0},
    {"id": "v2-05", "prompt": "Ibland ", "options": ["går vi ut.", "vi går ut.", "ut går vi."], "correct": 0},
    {"id": "v2-06", "prompt": "Därför ", "options": ["stannar vi hemma.", "vi stannar hemma.", "hemma stannar vi."], "correct": 0},
    {"id": "v2-07", "prompt": "På måndag ", "options": ["börjar skolan.", "skolan börjar.", "börjar skolan vi."], "correct": 0},
    {"id": "v2-08", "prompt": "Oftast ", "options": ["dricker hon kaffe.", "hon dricker kaffe.", "kaffe dricker hon."], "correct": 0},
    {"id": "v2-09", "prompt": "Senare ", "options": ["kommer vi tillbaka.", "vi kommer tillbaka.", "tillbaka kommer vi."], "correct": 0},
    {"id": "v2-10", "prompt": "Aldrig ", "options": ["har jag sett det.", "jag har sett det.", "det har jag sett."], "correct": 0},
    {"id": "v2-11", "prompt": "Förra veckan ", "options": ["reste de till Lund.", "de reste till Lund.", "till Lund reste de."], "correct": 0},
    {"id": "v2-12", "prompt": "Äntligen ", "options": ["är det klart.", "det är klart.", "klart är det."], "correct": 0},
]

AGREEMENT_PROBES = [
    {"id": "agr-01", "prompt": "Jag ser en röd ", "options": ["bil.", "hus."], "correct": 0},
    {"id": "agr-02", "prompt": "Jag ser ett rött ", "options": ["hus.", "bil."], "correct": 0},
    {"id": "agr-03", "prompt": "Hon har en stor ", "options": ["hund.", "bord."], "correct": 0},
    {"id": "agr-04", "prompt": "Hon har ett stort ", "options": ["bord.", "hund."], "correct": 0},
    {"id": "agr-05", "prompt": "Vi köpte en ny ", "options": ["soffa.", "bord."], "correct": 0},
    {"id": "agr-06", "prompt": "Vi köpte ett nytt ", "options": ["bord.", "soffa."], "correct": 0},
    {"id": "agr-07", "prompt": "Det är en gammal ", "options": ["bok.", "hus."], "correct": 0},
    {"id": "agr-08", "prompt": "Det är ett gammalt ", "options": ["hus.", "bok."], "correct": 0},
    {"id": "agr-09", "prompt": "Jag ser den röda ", "options": ["bilen.", "bil.", "bilar."], "correct": 0},
    {"id": "agr-10", "prompt": "Jag ser det röda ", "options": ["huset.", "hus.", "husen."], "correct": 0},
    {"id": "agr-11", "prompt": "Vi gillar den nya ", "options": ["filmen.", "film.", "filmer."], "correct": 0},
    {"id": "agr-12", "prompt": "Han bor i ", "options": ["huset.", "husen."], "correct": 0},
]

OPENENDED_PROMPTS = [
    "Vad är huvudstaden i Sverige? Svara kort på svenska.",
    "Förklara på svenska vad en cykel är.",
    "Nämn tre svenska traditioner.",
    "Vad betyder ordet 'fika'? Svara på svenska.",
    "Beskriv vädret i Sverige på vintern på svenska.",
    "Vem skrev Pippi Långstrump? Svara kort.",
    "Hur säger man 'thank you' på svenska?",
    "Berätta kort på svenska vad Lucia är.",
]


def all_probes():
    items = []
    for p in V2_PROBES:
        items.append({**p, "category": "v2", "kind": "mc"})
    for p in AGREEMENT_PROBES:
        items.append({**p, "category": "agreement", "kind": "mc"})
    return items


def split_probes(seed=0):
    items = all_probes()
    v2 = [p for p in items if p["category"] == "v2"]
    agr = [p for p in items if p["category"] == "agreement"]
    dev = v2[::3] + agr[::3]
    evl = [p for p in items if p not in dev]
    return {"dev": dev, "eval": evl}


def save_probes(path="configs/probes.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "probes": all_probes(),
                "openended": OPENENDED_PROMPTS,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return path


def load_probes(path="configs/probes.json"):
    with open(path) as f:
        return json.load(f)