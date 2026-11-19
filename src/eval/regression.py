ENGLISH_MCQ = [
    {"id": "en-01", "prompt": "The capital of France is ", "options": ["Paris.", "London.", "Berlin."], "correct": 0},
    {"id": "en-02", "prompt": "Water freezes at ", "options": ["0 degrees Celsius.", "100 degrees Celsius.", "50 degrees Celsius."], "correct": 0},
    {"id": "en-03", "prompt": "The largest planet in the solar system is ", "options": ["Jupiter.", "Mars.", "Earth."], "correct": 0},
    {"id": "en-04", "prompt": "The chemical symbol for gold is ", "options": ["Au.", "Ag.", "Fe."], "correct": 0},
    {"id": "en-05", "prompt": "The author of 'Hamlet' is ", "options": ["William Shakespeare.", "Charles Dickens.", "Mark Twain."], "correct": 0},
    {"id": "en-06", "prompt": "The square root of 144 is ", "options": ["12.", "14.", "16."], "correct": 0},
    {"id": "en-07", "prompt": "The longest river in the world is generally considered to be the ", "options": ["Nile.", "Amazon.", "Yangtze."], "correct": 0},
    {"id": "en-08", "prompt": "The speed of light is approximately ", "options": ["300,000 km per second.", "300 km per second.", "30,000 km per second."], "correct": 0},
    {"id": "en-09", "prompt": "The largest ocean on Earth is the ", "options": ["Pacific.", "Atlantic.", "Indian."], "correct": 0},
    {"id": "en-10", "prompt": "The number of continents is typically counted as ", "options": ["seven.", "five.", "nine."], "correct": 0},
]


def load_items():
    return [{**p, "category": "english", "kind": "mc"} for p in ENGLISH_MCQ]


def run_regression(model, tokenizer, device):
    from .scoring import evaluate_mc

    return evaluate_mc(model, tokenizer, load_items(), device)