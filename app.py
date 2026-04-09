import os
import re
from dotenv import load_dotenv
from openai import OpenAI

# --- LOAD API KEY ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def normalize_lookup_key(text):
    text = text.strip().lower()
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

# --- NORMALIZATION ---
def extract_levels(text):
    t = text.lower()
    compact = re.sub(r"\s+", "", t)

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
    text = re.sub(r"levels?\s+([IVX]+)\s*-\s*([IVX]+)", r"levels \1-\2", text, flags=re.IGNORECASE)
    text = re.sub(r"level\s+([IVX]+)\s*-\s*([IVX]+)", r"levels \1-\2", text, flags=re.IGNORECASE)
    text = re.sub(r"II,\s*III\s*and\s*IV", "II-IV", text, flags=re.IGNORECASE)
    text = re.sub(r"II,\s*III,\s*IV", "II-IV", text, flags=re.IGNORECASE)
    return text

def normalize_terms(text):
    t = text.lower()

    replacements = {
        "ln": "lymph nodes",
        "bot": "base of tongue",
    }

    for k, v in replacements.items():
        t = t.replace(k, v)

    return t

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

    if "thyroid and isthmus" in t:
        laterality = detect_laterality(text)
        return "Thyroid gland", f"{laterality} lobe and isthmus".strip(), "thyroidectomy"

    if "total thyroid" in t:
        return "Thyroid gland", "", "total thyroidectomy"

    if "thyroid tissue" in t:
        return "Thyroid", text, "excision"

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

    if any(x in t for x in ["maxillary", "sinonasal", "septal", "turbinate", "cribiform", "cribriform", "nasal tumor"]):
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

def build_line(label, text):
    text = normalize_levels(text)
    text = normalize_terms(text)

    laterality = detect_laterality(text)
    specimen_type, site, procedure = detect_specimen(text)

    # Clean site text
    site_clean = clean_site_text(site)
    site_clean = site_clean.replace("right", "").replace("left", "").strip()

    if detect_margin(text) and not procedure:
        procedure = "excision (fs)"

    if detect_margin(text) and procedure == "excision":
        procedure = "excision (fs)"

    if not procedure:
        procedure = "excision"

    parts = [specimen_type]

    if laterality and site_clean and site_clean != laterality:
        parts.append(f"{laterality} {site_clean}".strip())
    elif site_clean:
        parts.append(site_clean)

    # ✅ FIXED LEVELS LOGIC
    levels = extract_levels(text)
    if levels:
        parts.append(levels)

    parts.append(procedure)

    return f"{label}. {', '.join(parts)}:\n\n-"

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
    if ":\n\n-" in text:
        return text
    if text.rstrip().endswith(":"):
        return text.rstrip() + "\n\n-"
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
        text = match.group(2)
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
        if memory_key in memory:
            structured = f"{label}. {memory[memory_key]}"
            used_memory = True
        else:
            structured = build_line(label, text)

        specimen_type, _, _ = detect_specimen(normalize_terms(text))
        if specimen_type != "Unknown" and not detect_margin(text):
            context["last_specimen_type"] = specimen_type

        # --- FIX MARGIN USING CONTEXT ---
        if "margin" in text.lower() and not used_memory:
            margin_detail = normalize_terms(text).lower()
            margin_detail = margin_detail.replace("right", "").replace("left", "").strip(" ,")
            clean_site = clean_site_text(text)
            specimen_type = context["last_specimen_type"] or detect_specimen(normalize_terms(text))[0]
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
            if use_margin_detail_only:
                structured = f"{label}. {specimen_type}, {clean_site}, excision (fs):\n\n-"
            else:
                site_with_side = f"{context['last_laterality']} {clean_site}".strip()
                structured = f"{label}. {specimen_type}, {site_with_side}, excision (fs):\n\n-"

        # --- CONFIDENCE FLAG ---
        flag = "" if used_memory else confidence_flag(text)
        if flag:
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
