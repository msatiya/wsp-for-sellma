SWEDISH_EXAMPLES = [
    ("Vad heter Sveriges huvudstad?", "Stockholm."),
    ("Vilken färg har himlen?", "Blå."),
    ("Hur många ben har en hund?", "Fyra."),
    ("Vad dricker man oftast till fika?", "Kaffe."),
    ("Vilket år firas varje år den 24 december?", "Jul."),
    ("Vad kallar man vatten i fast form?", "Is."),
    ("Vilket är Sveriges största djur?", "Älgen."),
    ("Vad betyder tack?", "Tack."),
    ("Vilken veckodag kommer efter måndag?", "Tisdag."),
    ("Vad är motsatsen till stor?", "Liten."),
    ("Vilken årstid kommer efter vinter?", "Vår."),
    ("Vad säger man när man hälsar?", "Hej."),
    ("Vilket land ligger Sverige i?", "Europa."),
    ("Vad använder man för att klippa papper?", "Sax."),
    ("Vilken är den tredje månaden?", "Mars."),
    ("Vad äter en ko?", "Gräs."),
    ("Vad kallas en ung hund?", "Valp."),
    ("Vilket väderstreck ligger norrut?", "Norr."),
    ("Vad gör man med en nyckel?", "Låser."),
    ("Vilken färg får man om man blandar rött och gult?", "Orange."),
]


def as_prompt_completion_pairs():
    return [dict(prompt=q, completion=a) for q, a in SWEDISH_EXAMPLES]


def as_text():
    return [f"Fråga: {q}\nSvar: {a}" for q, a in SWEDISH_EXAMPLES]