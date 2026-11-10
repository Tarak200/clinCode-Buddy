"""Pull the assessment / diagnosis text out of a clinical note (local, no API).

The diagnosis is usually one short statement inside a long note: an `A:`, `Dx:` or
`Imp:` entry in typed notes, or the doctor's "this is ..." turn in a transcript.
Retrieving with that text instead of the whole note removes history, vitals and
plan noise. Extraction is a heuristic; callers must keep the full note as a
fallback query so a miss here cannot lose the answer.
"""

import re

# `A:` is case-sensitive so ordinary lowercase "a:" in prose is not treated as a section.
_MARKER = re.compile(
    r"(?<![\w/])(?:(?:(?i:dx|diagnosis|impression|imp|assessment)|A)\s*:|(?i:dx)\s*[-\u2013\u2014]\s*|(?i:dx)\s+(?=[A-Za-z?]))"
)
# Start of the next section: the diagnosis text ends here.
_SECTION = re.compile(
    r"(?<![\w/])(?:(?i:plan|rx|mx|tx|management|advice|ix|o/e|exam|labs|follow-?up|f/u)|[SOPA])\s*:"
)
# U+FFFD shows up where the source encoding lost a dash, so treat it as one.
_DASH = re.compile(r"\s+[-\u2013\u2014\ufffd]+\s+")
# A full stop that ends a sentence, but not an initial such as "E. coli".
_SENTENCE_END = re.compile(r"(?<!\b[A-Za-z])[.!?](?=\s+[A-Z0-9])")
_LIST_ITEM = re.compile(r"^\s*(?:\d+[.)]|[-*\u2022])\s+")
_LEADING_ENUM = re.compile(r"^\s*(?:\d+[.)]|[-*\u2022])\s*")

_NEGATION = re.compile(
    r"\b(?:exclu(?:ded|ding)|ruled?\s*out|rule\s*out|r/o|negative|neg|no\s+evidence"
    r"|not\s+(?:consistent|keeping|likely)|unlikely|denies|denied|absent)\b"
    r"|^\s*(?:no|not|nil|without)\b",
    re.IGNORECASE,
)
_CLAUSE_SPLIT = re.compile(r"[;,]|\bbut\b", re.IGNORECASE)
_PARENTHETICAL = re.compile(r"\([^()]*\)")
_NEGATED_WORD = re.compile(r"\b(?:no|not|nil|without)\b", re.IGNORECASE)

_SPEAKER = re.compile(r"^\s*([^\W\d_][\w .'-]{0,30}?)\s*:\s*(.*)$")
_DOCTOR = re.compile(
    r"^(?:doctor|doctora|dr\.?|doc|clinician|physician|provider|nurse|clinical officer"
    r"|m[eé]dico|m[eé]decin|docteur|daktari|arzt|[ée]rzt?in)$",
    re.IGNORECASE,
)
_STRONG_CUE = re.compile(
    r"\b(?:this is|this looks like|looks like|sounds like|appears to be|seems to be"
    r"|consistent with|suggestive of|likely|probably|diagnosis(?: is)?|diagnosed with"
    r"|impression(?: is)?|you have|you've got|you have got|could be)\b[:,]?\s*",
    re.IGNORECASE,
)
# "It's ..." is also used for plans and advice, so it is only a fallback cue.
_WEAK_CUE = re.compile(r"\b(?:it[\u2019']s|it is)\b[:,]?\s*", re.IGNORECASE)
# A cue inside a conditional ("if it's not better ...") is advice, not a diagnosis.
_CONDITIONAL = re.compile(r"\b(?:if|when|unless|whether|once)\b", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" \t.;:,-")


def drop_negated(text: str) -> str:
    """Remove clauses that rule a condition out ("malaria excluded", "no convulsions").

    Commas inside parentheses do not split clauses, and a parenthetical that
    contains a negation ("(no alarm features)") is dropped whole.
    """
    text = _PARENTHETICAL.sub(
        lambda match: ""
        if _NEGATION.search(match.group(0)) or _NEGATED_WORD.search(match.group(0))
        else match.group(0).replace(",", "\u0000"),
        text,
    )
    kept = [
        clause.replace("\u0000", ",")
        for clause in _CLAUSE_SPLIT.split(text)
        if clause.strip() and not _NEGATION.search(clause)
    ]
    return _clean(", ".join(kept))


def _marker_text(text: str, start: int) -> str:
    """Return the diagnosis text that follows a marker ending at `start`."""
    line_end = text.find("\n", start)
    if line_end == -1:
        line_end = len(text)
    chunk = text[start:line_end]
    following = text[line_end + 1 :].split("\n") if line_end < len(text) else []

    parts: list[str] = []
    if chunk.strip():
        parts.append(chunk)
        # Only an enumerated first item ("A: 1. malaria") continues onto the next lines.
        if _LIST_ITEM.match(chunk.strip() + " "):
            for line in following:
                if not _LIST_ITEM.match(line):
                    break
                parts.append(line)
    else:
        # Marker alone on its line: the diagnosis is on the next non-blank lines.
        for line in following:
            if not line.strip() or (_SECTION.match(line.lstrip()) and parts):
                break
            parts.append(line)
            if not _LIST_ITEM.match(line):
                break
    chunk = ", ".join(
        item
        for item in (_LEADING_ENUM.sub("", part).strip(" .;") for part in parts)
        if item
    )

    for terminator in (_SECTION, _DASH, _SENTENCE_END):
        match = terminator.search(chunk)
        if match:
            chunk = chunk[: match.start()]
    return _clean(chunk)


def _transcript_turns(text: str) -> list[tuple[str, str]]:
    """Split a transcript into (speaker, utterance) turns; [] if it is not one."""
    turns: list[list[str]] = []
    for line in text.splitlines():
        match = _SPEAKER.match(line)
        if match and len(match.group(1).split()) <= 3:
            turns.append([match.group(1).strip(), match.group(2)])
        elif turns and line.strip():
            turns[-1][1] += " " + line.strip()
    if len({speaker.casefold() for speaker, _ in turns}) < 2:
        return []
    return [(speaker, utterance) for speaker, utterance in turns]  # lists -> tuples


def _transcript_diagnosis(turns: list[tuple[str, str]], max_sentences: int = 2) -> list[str]:
    """Collect the doctor's diagnostic statements, scanning from the last turn back."""
    for cue_pattern in (_STRONG_CUE, _WEAK_CUE):
        found: list[str] = []
        for speaker, utterance in reversed(turns):
            if not _DOCTOR.match(speaker):
                continue
            for sentence in reversed(_SENTENCE_SPLIT.split(utterance)):
                cue = cue_pattern.search(sentence)
                if (
                    not cue
                    or sentence.rstrip().endswith("?")
                    or _CONDITIONAL.search(sentence[: cue.start()])
                ):
                    continue
                tail = drop_negated(_clean(sentence[cue.end() :]))
                if tail:
                    found.append(tail)
                if len(found) >= max_sentences:
                    return found[::-1]
        if found:
            return found[::-1]
    return []


def extract_assessment(text: str) -> str | None:
    """Return the note's assessment text with negated clauses removed, or None.

    None means "no recognisable assessment"; it also covers an assessment that
    consists only of ruled-out conditions.
    """
    pieces: list[str] = []
    for match in _MARKER.finditer(text):
        piece = _marker_text(text, match.end())
        if piece:
            pieces.append(piece)

    if not pieces:
        turns = _transcript_turns(text)
        if turns:
            pieces = _transcript_diagnosis(turns)

    unique: list[str] = []
    for piece in pieces:
        cleaned = drop_negated(piece)
        if cleaned and cleaned.casefold() not in {u.casefold() for u in unique}:
            unique.append(cleaned)
    return "; ".join(unique) or None
