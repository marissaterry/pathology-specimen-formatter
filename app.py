import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI

# --- LOAD API KEY ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ALLOWED_SPECIMEN_TYPES = {
    "Bone",
    "Larynx",
    "Lip",
    "Lymph node",
    "Lymph nodes",
    "Nasal cavity",
    "Nasal cavity and paranasal sinuses",
    "Nasopharynx",
    "Nose",
    "Oral cavity",
    "Oral cavity and bone",
    "Oral cavity/oropharynx",
    "Oropharynx",
    "Paranasal sinus",
    "Parathyroid",
    "Submandibular gland",
    "Thyroid",
    "Thyroid gland",
    "Unknown",
}

ALLOWED_PROCEDURES = {
    "biopsy",
    "biopsy (fs)",
    "dissection",
    "excision",
    "excision (fs)",
    "extraction",
    "neck dissection",
    "partial glossectomy (fs)",
    "radical tonsillectomy (fs)",
    "resection",
    "thyroidectomy",
    "thyroidectomy (including fs)",
    "total laryngectomy",
    "total thyroidectomy",
}

ALLOWED_LATERALITY = {"", "left", "right", "bilateral"}
ARABIC_TO_ROMAN = {
    "1": "I",
    "2": "II",
    "3": "III",
    "4": "IV",
    "5": "V",
    "6": "VI",
}
COMMON_SPELLING_FIXES = {
    "rigth": "right",
    "rigt": "right",
    "rihgt": "right",
    "leftt": "left",
    "bilaterl": "bilateral",
    "bilatreal": "bilateral",
    "tonuge": "tongue",
    "tounge": "tongue",
    "toungue": "tongue",
    "tonsilectomy": "tonsillectomy",
    "tonsillectmy": "tonsillectomy",
    "glosso tonsillar": "glossotonsillar",
    "glosso-tonsillar": "glossotonsillar",
    "cribiform": "cribriform",
    "nasopharngeal": "nasopharyngeal",
    "oropharnyx": "oropharynx",
    "lymphnode": "lymph node",
    "lymphnodes": "lymph nodes",
}


def normalize_lookup_key(text):
    text = preprocess_text(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

# --- LOAD EXAMPLES ---
def load_examples():
    with open("examples.txt", "r") as f:
        text = f.read()

    cases = text.split("INPUT:")
    examples = []

    for case in cases:
        case = case.strip()
        if not case or "OUTPUT:" not in case:
            continue

        input_part, output_part = case.split("OUTPUT:")

        inputs = [line.strip() for line in input_part.strip().split("\n") if re.match(r"^[A-Z]\.\s*", line.strip())]
        outputs = [line.strip() for line in output_part.strip().split("\n") if re.match(r"^[A-Z]\.\s*", line.strip())]

        examples.append((inputs, outputs))

    return examples


def build_example_records(examples):
    records = []
    for inputs, outputs in examples:
        for inp, out in zip(inputs, outputs):
            input_match = re.match(r"^[A-Z]\.\s*(.*)", inp)
            output_match = re.match(r"^[A-Z]\.\s*(.*)", out)
            if not input_match or not output_match:
                continue
            raw_input = input_match.group(1).strip()
            raw_output = output_match.group(1).strip()
            records.append(
                {
                    "input": raw_input,
                    "output": raw_output,
                    "key": normalize_lookup_key(raw_input),
                }
            )
    return records

# --- NORMALIZATION ---
def extract_levels(text):
    t = text.lower()
    compact = re.sub(r"\s+", "", t)

    normalized_anterior_match = re.search(
        r"levels?\s+([ivx]+)\s*,\s*([ivx]+)\s*,\s*([ivx]+)\s*,?\s*and\s*anterior\s*([ivx]+)",
        t,
        flags=re.IGNORECASE,
    )
    if normalized_anterior_match:
        vals = [normalized_anterior_match.group(i).upper() for i in range(1, 5)]
        return f"levels {vals[0]}, {vals[1]}, {vals[2]}, and anterior {vals[3]}"

    anterior_match = re.search(
        r"levels?\s+([1-6ivx]+)\s*,\s*([1-6ivx]+)\s*,\s*([1-6ivx]+)\s*and\s*anterior\s*([1-6ivx]+)",
        t,
        flags=re.IGNORECASE,
    )
    if anterior_match:
        vals = []
        for i in range(1, 5):
            token = anterior_match.group(i).upper()
            token = ARABIC_TO_ROMAN.get(token, token)
            vals.append(token)
        return f"levels {vals[0]}, {vals[1]}, {vals[2]}, and anterior {vals[3]}"

    digit_range_match = re.search(r"levels?([1-6])-([1-6])", compact)
    if digit_range_match:
        start = ARABIC_TO_ROMAN[digit_range_match.group(1)]
        end = ARABIC_TO_ROMAN[digit_range_match.group(2)]
        return f"levels {start}-{end}"

    if "2,3and4" in compact or "2,3,4" in compact:
        return "levels II-IV"
    if "2,3" in compact:
        return "levels II-III"
    if "3,4" in compact:
        return "levels III-IV"

    single_digit_match = re.search(r"\blevel\s*([1-6])\b", t)
    if single_digit_match:
        return f"level {ARABIC_TO_ROMAN[single_digit_match.group(1)]}"

    range_match = re.search(r"levels?([ivx]+)-([ivx]+)", compact)
    if range_match:
        start = range_match.group(1).upper()
        end = range_match.group(2).upper()
        return f"levels {start}-{end}"

    if "ii,iiiandiv" in compact or "ii,iii,iv" in compact:
        return "levels II-IV"
    if "ii,iii" in compact:
        return "levels II-III"
    if "iii,iv" in compact:
        return "levels III-IV"

    single_match = re.search(r"\blevel\s*([ivx]+)\b", t)
    if single_match:
        return f"level {single_match.group(1).upper()}"

    return ""

def normalize_levels(text):
    text = re.sub(
        r"\blevels?\s+2\s*,\s*3\s*,\s*4\s*,?\s*and\s*anterior\s*5\b",
        "levels II, III, IV, and anterior V",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\blevels?\s+2\s*,\s*3\s*,\s*4\s*,?\s*and\s*anterior5\b",
        "levels II, III, IV, and anterior V",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\banterior\s*([1-6])\b",
        lambda m: f"anterior {ARABIC_TO_ROMAN[m.group(1)]}",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\blevels?\s+2\s*,\s*3\s*and\s*4\b(?!\s*,?\s*and\s*anterior)",
        "levels II-IV",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\blevels?\s+2\s*,\s*3\s*,\s*4\b(?!\s*,?\s*and\s*anterior)",
        "levels II-IV",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\blevels?\s+2\s*,\s*3\b",
        "levels II-III",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\blevels?\s+3\s*,\s*4\b",
        "levels III-IV",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\blevels?\s+([1-6])\s*-\s*([1-6])\b",
        lambda m: f"levels {ARABIC_TO_ROMAN[m.group(1)]}-{ARABIC_TO_ROMAN[m.group(2)]}",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\blevel\s+([1-6])\b",
        lambda m: f"level {ARABIC_TO_ROMAN[m.group(1)]}",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\blevel\s+1\s*a\b",
        "level IA",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\blevel\s+1\s*b\b",
        "level IB",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"levels?\s+([IVX]+)\s*-\s*([IVX]+)", r"levels \1-\2", text, flags=re.IGNORECASE)
    text = re.sub(r"level\s+([IVX]+)\s*-\s*([IVX]+)", r"levels \1-\2", text, flags=re.IGNORECASE)
    text = re.sub(r"II,\s*III\s*and\s*IV", "II-IV", text, flags=re.IGNORECASE)
    text = re.sub(r"II,\s*III,\s*IV", "II-IV", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\blevels\s+II-IV,\s*and\s*anterior\s*V\b",
        "levels II, III, IV, and anterior V",
        text,
        flags=re.IGNORECASE,
    )
    return text

def normalize_terms(text):
    t = preprocess_text(text).lower()

    replacements = {
        "ln": "lymph nodes",
        "bot": "base of tongue",
    }

    for k, v in replacements.items():
        t = t.replace(k, v)

    return t


def normalize_parentheses(text):
    text = text.replace("((", "(").replace("))", ")")
    while text.count("(") > text.count(")"):
        text += ")"
    while text.count(")") > text.count("("):
        text = text.replace(")", "", 1)
    return text


def normalize_punctuation(text):
    text = normalize_parentheses(text)
    text = re.sub(r"\s*[,;:]\s*", lambda m: m.group(0).strip() + " ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ,;")


def normalize_spelling(text):
    fixed = text
    for wrong, right in COMMON_SPELLING_FIXES.items():
        fixed = re.sub(rf"\b{re.escape(wrong)}\b", right, fixed, flags=re.IGNORECASE)
    return fixed


def preprocess_text(text):
    text = normalize_spelling(text)
    text = normalize_punctuation(text)
    text = normalize_levels(text)
    return text.strip()

def detect_laterality(text):
    t = text.lower()

    if "right and left" in t or "bilateral" in t:
        return "bilateral"
    if "right" in t:
        return "right"
    if "left" in t:
        return "left"

    return ""

def detect_specimen(text):
    t = text.lower()

    if "neck level i a lymph node" in t or "neck level ia lymph node" in t:
        laterality = detect_laterality(text)
        return "Lymph nodes", f"{laterality} level IA".strip(), "dissection"

    if "neck level i b lymph node" in t or "neck level ib lymph node" in t:
        laterality = detect_laterality(text)
        return "Lymph nodes", f"{laterality} level IB".strip(), "dissection"

    if "facial lymph node" in t:
        laterality = detect_laterality(text)
        return "Lymph node", f"{laterality} facial".strip(), "dissection"

    if "submandibular gland" in t:
        laterality = detect_laterality(text)
        return "Submandibular gland", laterality, "neck dissection"

    if "parathyroid" in t and "biopsy" in t:
        laterality = detect_laterality(text)
        detail = re.sub(r"\b(left|right)\b", "", text, flags=re.IGNORECASE)
        detail = re.sub(r"\bparathyroid\b", "", detail, flags=re.IGNORECASE)
        detail = re.sub(r"\bbiopsy\b", "", detail, flags=re.IGNORECASE)
        detail = re.sub(r"\s+", " ", detail).strip(" ,")
        return "Parathyroid", f"{laterality} {detail}".strip(), "biopsy (fs)"

    if "lower lip" in t and "border of skin" in t:
        laterality = detect_laterality(text)
        detail = re.sub(r"\b(left|right)\b", "", text, flags=re.IGNORECASE)
        detail = re.sub(r"\blower lip\b", "lower,", detail, flags=re.IGNORECASE)
        detail = re.sub(r"\s+", " ", detail).strip(" ,")
        return "Lip", f"{laterality} {detail}".strip(), "excision"

    if "lower lip" in t or re.search(r"\blip\b", t):
        laterality = detect_laterality(text)
        site = "lower" if "lower lip" in t else text
        procedure = "resection" if "lower lip" in t and "border of skin" not in t else "excision"
        return "Lip", f"{laterality} {site}".strip(), procedure

    if "thyroid and isthmus" in t:
        laterality = detect_laterality(text)
        return "Thyroid gland", f"{laterality} lobe and isthmus".strip(), "thyroidectomy"

    if "total thyroid" in t:
        return "Thyroid gland", "", "total thyroidectomy"

    if "thyroid tissue" in t:
        return "Thyroid", text, "excision"

    if "central neck" in t and "nerve insertion" in t:
        laterality = detect_laterality(text)
        site = re.sub(r"\blymph nodes?\b", "", text, flags=re.IGNORECASE)
        site = re.sub(r"\s+", " ", site).strip(" ,")
        return "Lymph nodes", site, "dissection"

    if "central neck" in t:
        return "Lymph nodes", "central neck", "dissection"

    if "neck" in t:
        return "Lymph nodes", "neck", "dissection"

    if "rhinectomy" in t:
        return "Nose", text, ""

    if "nasal bone" in t or "anterior nasal spine" in t:
        return "Bone", text, "excision"

    if "sinonasal contents" in t:
        if detect_laterality(text):
            return "Nasal cavity and paranasal sinuses", detect_laterality(text), "extraction"
        return "Nasal cavity and paranasal sinuses", text, "extraction"

    if "nasopharyngeal" in t or "nasopharynx" in t:
        site = text
        site = re.sub(r"\bnasopharyngeal\b", "", site, flags=re.IGNORECASE)
        site = re.sub(r"\bnasopharynx\b", "", site, flags=re.IGNORECASE)
        site = re.sub(r"\s+", " ", site).strip(" ,")
        if "tumor" in t and "margin" not in t and site == "tumor":
            site = '"tumor"'
        if "cyst" in t:
            site = site.replace("cyst", '"cyst"').strip()
        return "Nasopharynx", site, "excision (fs)" if "margin" in t else "excision"

    if "middle meatus mass" in t:
        site = re.sub(r"\bmass\b", "", text, flags=re.IGNORECASE).strip(" ,")
        return "Nasal cavity", site, "excision (fs)"

    if any(x in t for x in ["maxillary", "sinonasal", "septal", "turbinate", "cribiform", "cribriform", "nasal tumor", "middle meatus"]):
        if "sinus contents" in t or "sinonasal contents" in t or "debris" in t:
            return "Paranasal sinus" if "maxillary" in t else "Nasal cavity and paranasal sinuses", text, "extraction"
        return "Paranasal sinus" if "maxillary" in t else "Nasal cavity", text, "excision"

    if any(x in t for x in ["floor of mouth", "tongue tumor", "maxilla", "soft palate"]):
        if "tongue tumor" in t:
            cleaned = text.replace("tumor", "").strip()
            return "Oral cavity", cleaned, "partial glossectomy (fs)"
        if "maxilla and soft palate" in t:
            return "Oral cavity/oropharynx", text, "resection"
        if "anterior maxillary wall soft and hard" in t:
            return "Oral cavity and bone", text, "resection"
        return "Oral cavity", text, "excision"

    if any(x in t for x in ["tongue", "tonsil", "oropharynx"]):
        return "Oropharynx", text, "excision"

    if "larynx" in t and t.strip() == "larynx":
        return "Larynx", "", "total laryngectomy"

    if any(x in t for x in ["larynx", "vocal fold", "paraglottic", "ventricle"]):
        return "Larynx", text, "excision"

    return "Unknown", text, "excision"

def detect_margin(text):
    return "margin" in text.lower()

def confidence_flag(text):
    if detect_margin(text):
        return ""

    specimen, _, _ = detect_specimen(text)

    if specimen == "Unknown":
        return "⚠️ CHECK SPECIMEN DESCRIPTION"

    return ""

# --- STRUCTURED BUILD ---
def clean_site_text(text):
    t = text.lower()

    # Remove margin wording from site
    if margin_needs_context(text):
        t = t.replace("margin", "").strip()
    t = re.sub(r"\blevels?\s+[ivx]+(?:\s*-\s*[ivx]+)?\b", "", t)
    t = re.sub(r"\blevels?\s+[ivx]+(?:\s*,\s*[ivx]+)*(?:\s*and\s*[ivx]+)?\b", "", t)

    # Normalize abbreviations
    t = t.replace("bot", "base of tongue")
    t = t.replace("ln", "lymph nodes")

    # Remove duplicate words
    t = t.replace("right right", "right")
    t = t.replace("left left", "left")

    return t.strip()


def margin_needs_context(text):
    lowered = normalize_terms(text).lower()
    anchored_tokens = [
        "tongue", "tonsil", "oropharynx", "nasopharyngeal", "nasopharynx",
        "septal", "turbinate", "cribriform", "cribiform", "maxillary",
        "floor of mouth", "lip", "vocal fold", "larynx", "meatus",
    ]
    return not any(token in lowered for token in anchored_tokens)

def build_line(label, text):
    return build_line_from_fields(label, extract_rule_fields(text))

def build_memory(examples):
    memory = {}

    for inputs, outputs in examples:
        for inp, out in zip(inputs, outputs):
            input_match = re.match(r"^[A-Z]\.\s*(.*)", inp)
            output_match = re.match(r"^[A-Z]\.\s*(.*)", out)
            if not input_match or not output_match:
                continue
            key = normalize_lookup_key(input_match.group(1))
            memory[key] = output_match.group(1)

    return memory


def tokenize_for_validation(text):
    return set(re.findall(r"[a-z0-9]+", normalize_terms(text).lower()))


def normalize_variant_key(text):
    normalized = normalize_lookup_key(text)
    normalized = re.sub(r"\b(left|right|bilateral)\b", "{LAT}", normalized)
    normalized = re.sub(
        r"\blevels?\s+[a-z0-9,\s-]+(?:and\s+[a-z0-9\s-]+)?",
        "{LEVEL}",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def find_template_examples(text, limit=3):
    variant_key = normalize_variant_key(text)
    input_tokens = tokenize_for_validation(text)
    boosted_tokens = {
        tok for tok in input_tokens
        if tok in {
            "parathyroid", "thyroid", "delphian", "tonsil", "tongue", "lip",
            "larynx", "meatus", "turbinate", "septal", "nasopharyngeal",
            "sinonasal", "maxillary", "floor", "mouth", "neck",
        }
    }
    scored = []

    for record in example_records:
        score = 0
        if record["key"] == normalize_lookup_key(text):
            score += 100
        if normalize_variant_key(record["input"]) == variant_key:
            score += 50

        overlap = input_tokens & tokenize_for_validation(record["input"])
        score += len(overlap)
        score += 5 * len(boosted_tokens & tokenize_for_validation(record["input"]))

        if score > 0:
            scored.append((score, record))

    scored.sort(key=lambda item: (-item[0], item[1]["input"]))
    return [record for _, record in scored[:limit]]


def has_variant_markers(text):
    lowered = normalize_lookup_key(text)
    if re.search(r"\b(left|right|bilateral)\b", lowered):
        return True
    if re.search(r"\blevels?\b", lowered):
        return True
    return False


def should_try_llm_first(text, memory_key):
    if memory_key in memory:
        return False
    if detect_specimen(normalize_terms(text))[0] == "Unknown":
        return True
    template_examples = find_template_examples(text, limit=2)
    if has_variant_markers(text) and template_examples:
        return True
    return False


def should_prefer_template_adaptation(text, memory_key, template_examples):
    if memory_key in memory or not template_examples:
        return False
    if not has_variant_markers(text):
        return False
    return normalize_variant_key(template_examples[0]["input"]) == normalize_variant_key(text)


def adapt_template_output(text, template_example):
    if not template_example:
        return None

    template_input = template_example["input"]
    template_output = template_example["output"]
    if normalize_variant_key(template_input) != normalize_variant_key(text):
        return None

    adapted = template_output

    source_laterality = detect_laterality(template_input)
    target_laterality = detect_laterality(text)
    if source_laterality and target_laterality and source_laterality != target_laterality:
        adapted = re.sub(
            rf"\b{re.escape(source_laterality)}\b",
            target_laterality,
            adapted,
            flags=re.IGNORECASE,
        )

    source_levels = extract_levels(template_input)
    target_levels = extract_levels(text)
    if source_levels and target_levels and source_levels != target_levels:
        adapted = re.sub(
            re.escape(source_levels),
            target_levels,
            adapted,
            flags=re.IGNORECASE,
        )

    return adapted


def is_rule_confident(text):
    specimen_type, _, _ = detect_specimen(normalize_terms(text))
    return specimen_type != "Unknown"


def extract_rule_fields(text):
    normalized_text = normalize_levels(normalize_terms(text))
    laterality = detect_laterality(normalized_text)
    specimen_type, site, procedure = detect_specimen(normalized_text)
    site_clean = clean_site_text(site)
    site_clean = site_clean.replace("right", "").replace("left", "").strip(" ,")

    if detect_margin(normalized_text) and not procedure:
        procedure = "excision (fs)"
    elif detect_margin(normalized_text) and procedure == "excision":
        procedure = "excision (fs)"
    elif not procedure:
        procedure = "excision"

    return {
        "specimen_type": specimen_type,
        "site": site_clean,
        "procedure": procedure,
        "laterality": laterality,
        "levels": extract_levels(normalized_text),
    }


def build_line_from_fields(label, fields):
    specimen_type = fields.get("specimen_type", "").strip()
    site = fields.get("site", "").strip()
    procedure = fields.get("procedure", "").strip()
    laterality = fields.get("laterality", "").strip()
    levels = fields.get("levels", "").strip()

    parts = [specimen_type]

    if laterality and site and site != laterality:
        parts.append(f"{laterality} {site}".strip())
    elif laterality and not site:
        parts.append(laterality)
    elif site:
        parts.append(site)

    if levels:
        parts.append(levels)

    if procedure:
        parts.append(procedure)

    return f"{label}. {', '.join([p for p in parts if p])}:\n-"


def build_error_line(label, text, reason):
    clean_reason = sanitize_output(reason or "Unable to classify specimen safely.")
    return (
        f'{label}. ERROR - REVIEW REQUIRED:\n'
        f'- Unable to classify specimen safely for "{text.strip()}".\n'
        f"- Reason: {clean_reason}"
    )


def validate_llm_fields(original_text, data):
    if not isinstance(data, dict):
        return False, "LLM response was not a JSON object."

    action = str(data.get("action", "")).strip().lower()
    if action not in {"classify", "error"}:
        return False, "LLM action must be 'classify' or 'error'."

    if action == "error":
        message = str(data.get("error_message", "")).strip()
        if not message:
            return False, "LLM error response omitted an explanation."
        return True, ""

    specimen_type = str(data.get("specimen_type", "")).strip()
    procedure = str(data.get("procedure", "")).strip()
    laterality = str(data.get("laterality", "")).strip().lower()
    levels = str(data.get("levels", "")).strip()
    site = str(data.get("site", "")).strip()
    expected_laterality = detect_laterality(original_text)
    expected_levels = extract_levels(original_text)

    if specimen_type not in ALLOWED_SPECIMEN_TYPES or specimen_type == "Unknown":
        return False, "LLM returned an unsupported or unknown specimen type."
    if procedure not in ALLOWED_PROCEDURES:
        return False, "LLM returned an unsupported procedure."
    if laterality not in ALLOWED_LATERALITY:
        return False, "LLM returned unsupported laterality."
    if expected_laterality and laterality != expected_laterality:
        return False, "LLM laterality did not match the input."
    if levels and not re.fullmatch(r"level [IVX]+|levels [IVX]+-[IVX]+", levels):
        return False, "LLM returned levels in an invalid format."
    if expected_levels and levels != expected_levels:
        return False, "LLM levels did not match the input."
    if not site and specimen_type not in {"Thyroid gland", "Nose", "Larynx"}:
        return False, "LLM omitted the site for a specimen that requires one."

    original_tokens = tokenize_for_validation(original_text)
    site_tokens = set(re.findall(r"[a-z0-9]+", normalize_terms(site).lower()))
    allowed_extra = {
        "and", "of", "the", "level", "levels", "ia", "ib", "ii", "iii", "iv", "v",
        "anterior", "posterior", "superior", "inferior", "medial", "lateral", "deep",
        "base", "tongue", "mouth", "floor", "soft", "hard", "maxillary", "sinus",
        "sinuses", "cavity", "gland", "node", "nodes", "contents", "tumor", "margin",
        "tonsil", "tonsillar", "glossotonsillar", "vocal", "fold", "space",
    }
    unexpected_tokens = {tok for tok in site_tokens if tok not in original_tokens and tok not in allowed_extra}
    if unexpected_tokens:
        return False, "LLM introduced site wording not grounded in the input text."

    return True, ""


def extract_with_llm(text, context, template_examples=None, rule_guess=None):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "OPENAI_API_KEY is not set."

    template_examples = template_examples or []
    formatted_examples = "\n".join(
        [
            f'Example input: {example["input"]}\nExample output: {example["output"]}'
            for example in template_examples
        ]
    ) or "No close examples available."

    rule_guess_block = (
        "Rule-based guess:\n"
        f"specimen_type={rule_guess.get('specimen_type', '')}\n"
        f"site={rule_guess.get('site', '')}\n"
        f"procedure={rule_guess.get('procedure', '')}\n"
        f"laterality={rule_guess.get('laterality', '')}\n"
        f"levels={rule_guess.get('levels', '')}\n"
    ) if rule_guess else "Rule-based guess:\nNone"

    prompt = f"""
You are extracting structured data from a single pathology specimen line.

Be conservative. Do not guess.
Use only information present in the specimen line and the context fields.
If you are not confident, return action="error".
When the input appears to be a variant of one of the examples, adapt only the changed laterality, level, or directly corresponding wording. Do not invent a new anatomy or procedure.

Allowed specimen_type values:
{sorted(ALLOWED_SPECIMEN_TYPES)}

Allowed procedure values:
{sorted(ALLOWED_PROCEDURES)}

Allowed laterality values:
["", "left", "right", "bilateral"]

Rules:
- Return valid JSON only.
- Prefer copying the site wording directly from the input line.
- Normalize levels only as "level II" or "levels II-IV" style.
- Never invent anatomy or procedure.
- If an example differs only by laterality or level, keep the same specimen_type and procedure unless the input clearly requires otherwise.
- A substitution such as left->right, right->left, or level VI->level IV is sufficient context when the rest of the specimen phrase matches a nearby example.
- If context is needed but still insufficient, return action="error".

Context:
last_site={context.get("last_site", "")}
last_laterality={context.get("last_laterality", "")}
last_specimen_type={context.get("last_specimen_type", "")}

Nearby examples:
{formatted_examples}

{rule_guess_block}

Input line:
{text}

Return exactly one JSON object with this shape:
{{
  "action": "classify" or "error",
  "specimen_type": "",
  "site": "",
  "procedure": "",
  "laterality": "",
  "levels": "",
  "error_message": ""
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
    except Exception as exc:
        return None, f"LLM extraction failed: {exc}"

    is_valid, reason = validate_llm_fields(text, data)
    if not is_valid:
        return None, reason

    if str(data.get("action", "")).strip().lower() == "error":
        return None, str(data.get("error_message", "")).strip()

    return {
        "specimen_type": str(data.get("specimen_type", "")).strip(),
        "site": clean_site_text(str(data.get("site", "")).strip()),
        "procedure": str(data.get("procedure", "")).strip(),
        "laterality": str(data.get("laterality", "")).strip().lower(),
        "levels": str(data.get("levels", "")).strip(),
    }, ""


def sanitize_output(text):
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\blevel Ib\b", "level IB", text)
    text = re.sub(r"\blevel ia\b", "level IA", text, flags=re.IGNORECASE)
    text = re.sub(r"\blevel ib\b", "level IB", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+:", ":", text)
    text = re.sub(r":{2,}", ":", text)
    return text.strip()


def ensure_placeholder_bullet(text):
    if ":\n-" in text:
        return text
    if text.rstrip().endswith(":"):
        return text.rstrip() + "\n-"
    return text

# --- OPTIONAL LLM CLEANUP ---
def refine_output(text):
    prompt = f"""
You are a formatting tool.

ONLY return the cleaned version of the text below.
DO NOT add examples.
DO NOT add explanations.
DO NOT repeat or continue anything.
DO NOT add extra lines.

Text:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

examples = load_examples()
memory = build_memory(examples)
example_records = build_example_records(examples)

def convert_specimens(input_text):
    outputs = []
    lines = input_text.split("\n")
    context = {
        "last_site": "",
        "last_laterality": "",
        "last_specimen_type": ""
    }

    for line in lines:
        match = re.match(r"([A-Z])\.\s*(.*)", line)

        if not match:
            continue

        label = match.group(1)
        text = preprocess_text(match.group(2))
        memory_key = normalize_lookup_key(text)

        # --- UPDATE CONTEXT FIRST ---
        if ("tongue" in text.lower() or "tonsil" in text.lower()) and not detect_margin(text):
            context["last_site"] = text

        if "right" in text.lower():
            context["last_laterality"] = "right"
        elif "left" in text.lower():
            context["last_laterality"] = "left"

        # --- BUILD NORMAL LINE ---
        used_memory = False
        used_llm = False
        used_template = False
        rule_guess = extract_rule_fields(text)
        template_examples = find_template_examples(text)
        template_output = adapt_template_output(text, template_examples[0]) if template_examples else None
        if memory_key in memory:
            structured = f"{label}. {memory[memory_key]}"
            used_memory = True
        elif should_prefer_template_adaptation(text, memory_key, template_examples) and template_output:
            structured = f"{label}. {template_output}"
            used_template = True
        else:
            llm_fields = None
            llm_error = ""
            if should_try_llm_first(text, memory_key):
                llm_fields, llm_error = extract_with_llm(
                    text,
                    context,
                    template_examples=template_examples,
                    rule_guess=rule_guess if rule_guess.get("specimen_type") != "Unknown" else None,
                )
                if llm_fields:
                    structured = build_line_from_fields(label, llm_fields)
                    used_llm = True
                elif template_output:
                    structured = f"{label}. {template_output}"
                    used_template = True
                elif is_rule_confident(text):
                    structured = build_line_from_fields(label, rule_guess)
                else:
                    structured = build_error_line(label, text, llm_error)
            elif is_rule_confident(text):
                structured = build_line_from_fields(label, rule_guess)
            else:
                llm_fields, llm_error = extract_with_llm(
                    text,
                    context,
                    template_examples=template_examples,
                    rule_guess=None,
                )
                if llm_fields:
                    structured = build_line_from_fields(label, llm_fields)
                    used_llm = True
                elif template_output:
                    structured = f"{label}. {template_output}"
                    used_template = True
                else:
                    structured = build_error_line(label, text, llm_error)

        specimen_type, _, _ = detect_specimen(normalize_terms(text))
        if specimen_type != "Unknown" and not detect_margin(text):
            context["last_specimen_type"] = specimen_type
        elif not used_memory and "ERROR - REVIEW REQUIRED" not in structured and not detect_margin(text):
            llm_specimen = re.match(rf"^{label}\.\s*([^,:]+)", structured)
            if llm_specimen:
                context["last_specimen_type"] = llm_specimen.group(1).strip()

        # --- FIX MARGIN USING CONTEXT ---
        if "margin" in text.lower() and not used_memory and margin_needs_context(text):
            margin_detail = normalize_terms(text).lower()
            margin_detail = margin_detail.replace("right", "").replace("left", "").strip(" ,")
            clean_site = clean_site_text(text)
            detected_specimen_type = detect_specimen(normalize_terms(text))[0]
            specimen_type = detected_specimen_type if detected_specimen_type != "Unknown" else context["last_specimen_type"]
            use_margin_detail_only = False
            if context["last_site"]:
                prior_site = clean_site_text(context["last_site"])
                if margin_detail and margin_detail != prior_site:
                    clean_site = f"{prior_site}, {margin_detail}"
                else:
                    clean_site = prior_site
            elif clean_site and clean_site in margin_detail:
                clean_site = margin_detail
                use_margin_detail_only = True
            clean_site = clean_site.replace("right", "").replace("left", "").strip()
            if specimen_type == "Unknown":
                structured = build_error_line(label, text, "Margin specimen referenced prior context, but specimen type is still unknown.")
            elif use_margin_detail_only:
                structured = f"{label}. {specimen_type}, {clean_site}, excision (fs):\n-"
            else:
                site_with_side = f"{context['last_laterality']} {clean_site}".strip()
                structured = f"{label}. {specimen_type}, {site_with_side}, excision (fs):\n-"

        # --- CONFIDENCE FLAG ---
        flag = "" if used_memory or used_llm or used_template else confidence_flag(text)
        if flag and "ERROR - REVIEW REQUIRED" not in structured:
            structured = structured + f"\n{flag}"

        structured = ensure_placeholder_bullet(structured)
        structured = sanitize_output(structured)
        outputs.append(structured)

    return "FINAL DIAGNOSIS\n" + "\n\n".join(outputs)

# --- RUN ---
if __name__ == "__main__":
    print("Paste specimen list (press ENTER twice when done):\n")

    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    input_text = "\n".join(lines)

    output = convert_specimens(input_text)

    print("\n--- OUTPUT ---\n")
    print(output)
