import re
from typing import Optional


_TARGET_PATTERNS = [
    r"where is the (?P<target>.+)",
    r"where is (?P<target>.+)",
    r"where are the (?P<target>.+)",
    r"where can i find (?P<target>.+)",
    r"where can i get (?P<target>.+)",
    r"find the (?P<target>.+)",
    r"find (?P<target>.+)",
    r"please find (?P<target>.+)",
    r"help me find (?P<target>.+)",
    r"help me get (?P<target>.+)",
    r"can you find (?P<target>.+?)(?: for me)?",
    r"can you show me (?P<target>.+)",
    r"i am looking for (?P<target>.+)",
    r"i want to find (?P<target>.+)",
    r"i need (?P<target>.+)",
    r"take me to the (?P<target>.+)",
    r"take me to (?P<target>.+)",
    r"show me (?P<target>.+)",
    r"is this the (?P<target>.+) section",
    r"is this (?P<target>.+)",
    r"do you see (?P<target>.+) here",
    r"do you see (?P<target>.+)",
    r"are there (?P<target>.+) here",
    r"d[oó]nde est[aá]n (?:el |la |los |las )?(?P<target>.+)",
    r"d[oó]nde est[aá] (?:el |la |los |las )?(?P<target>.+)",
    r"d[oó]nde puedo encontrar (?:el |la |los |las )?(?P<target>.+)",
    r"d[oó]nde puedo coger (?:el |la |los |las )?(?P<target>.+)",
    r"ay[uú]dame a encontrar (?:el |la |los |las )?(?P<target>.+)",
    r"estoy buscando (?:el |la |los |las )?(?P<target>.+)",
    r"quiero encontrar (?:el |la |los |las )?(?P<target>.+)",
    r"necesito (?:el |la |los |las )?(?P<target>.+)",
    r"ll[eé]vame (?:al |a la |a los |a las )?(?P<target>.+)",
    r"ens[eé][ñn]ame (?:el |la |los |las )?(?P<target>.+)",
    r"busca (?:el |la |los |las )?(?P<target>.+)",
    r"puedes encontrar (?:el |la |los |las )?(?P<target>.+?) para m[ií]",
    r"puedes encontrar (?:el |la |los |las )?(?P<target>.+)",
    r"me puedes ense[ñn]ar (?:el |la |los |las )?(?P<target>.+)",
    r"esta es la zona de (?:el |la |los |las )?(?P<target>.+)",
    r"esta es la secci[oó]n de (?:el |la |los |las )?(?P<target>.+)",
    r"ves (?:el |la |los |las )?(?P<target>.+) por aqu[ií]",
    r"hay (?:el |la |los |las )?(?P<target>.+) aqu[ií]",
]


def normalize_text(text: str) -> str:
    return text.lower().strip().strip(".,?!¿¡")


def normalize_target(target: str) -> str:
    target = target.lower().strip().strip(".,?!¿¡")

    prefixes = [
        "the ",
        "a ",
        "an ",
        "some ",
        "el ",
        "la ",
        "los ",
        "las ",
        "un ",
        "una ",
        "unos ",
        "unas ",
    ]

    for prefix in prefixes:
        if target.startswith(prefix):
            target = target[len(prefix):]

    suffixes = [
        ", please",
        " please",
        ", por favor",
        " por favor",
    ]

    for suffix in suffixes:
        if target.endswith(suffix):
            target = target[: -len(suffix)]

    return target.strip()


def extract_target(text: str) -> Optional[str]:
    normalized = normalize_text(text)

    for pattern in _TARGET_PATTERNS:
        match = re.fullmatch(pattern, normalized)
        if match:
            return normalize_target(match.group("target"))

    return None
