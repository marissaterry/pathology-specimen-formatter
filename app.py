import os
import re
from dotenv import load_dotenv
from openai import OpenAI

# --- LOAD API KEY ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- LOAD EXAMPLES ---
def load_examples():
    try:
        with open("examples.txt", "r") as f:
            return f.read()
    except:
        return ""

# --- NORMALIZATION ---
def extract_levels(text):
    t = text.lower()

    if "ii, iii and iv" in t or "ii, iii, iv" in t:
        return "levels II-IV"
    if "ii, iii" in t:
        return "levels II-III"
    if "iii, iv" in t:
        return "levels III-IV"
    if "ii" in t:
        return "level II"

    return ""

def normalize_levels(text):
    text = text.replace("II, III and IV", "II-IV")
    text = text.replace("II, III, IV", "II-IV")
    return text

def normalize_terms(text):
    t = text.lower()

    replacements = {
        "ln": "lymph nodes",
        "bot": "base of tongue",
        "tonsillectomy": "tonsil",
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

    if "neck" in t:
        return "Lymph nodes", "neck", "dissection"

    if any(x in t for x in ["tongue", "tonsil", "oropharynx"]):
        return "Oropharynx", text, "excision"

    if "larynx" in t or "vocal fold" in t:
        return "Larynx", text, "excision"

    return "Unknown", text, "excision"

def detect_margin(text):
    return "margin" in text.lower()

def confidence_flag(text):
    t = text.lower()
            
    known_terms = ["neck", "tongue", "tonsil", "larynx", "margin"]
            
    if not any(term in t for term in known_terms):
        return "⚠️ CHECK SPECIMEN DESCRIPTION"
        
    return ""

# --- STRUCTURED BUILD ---
def clean_site_text(text):
    t = text.lower()

    # Remove margin wording from site
    t = t.replace("margin", "").strip()

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

    if detect_margin(text):
        procedure = "excision (fs)"
        if "oropharynx" not in specimen_type.lower():
            specimen_type = "Oropharynx"

    parts = [specimen_type]

    if laterality:
        parts.append(f"{laterality} {site_clean}".strip())
    else:
        parts.append(site_clean)

    # ✅ FIXED LEVELS LOGIC
    levels = extract_levels(text)
    if levels:
        parts.append(levels)

    parts.append(procedure)

    return f"{label}. {', '.join(parts)}:\n\n-"

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

# --- MAIN ---
context = {
    "last_site": "",
    "last_laterality": ""
}

def convert_specimens(input_text):
    outputs = []
    lines = input_text.split("\n")

    for line in lines:
        match = re.match(r"([A-Z])\.\s*(.*)", line)

        if not match:
            continue

        label = match.group(1)
        text = match.group(2)

        # --- UPDATE CONTEXT FIRST ---
        if "tongue" in text.lower() or "tonsil" in text.lower():
            context["last_site"] = text

        if "right" in text.lower():
            context["last_laterality"] = "right"
        elif "left" in text.lower():
            context["last_laterality"] = "left"

        # --- BUILD NORMAL LINE ---
        structured = build_line(label, text)

        # --- FIX MARGIN USING CONTEXT ---
        if "margin" in text.lower():
            if context["last_site"]:
                clean_site = clean_site_text(context["last_site"])
            structured = f"{label}. Oropharynx, {context['last_laterality']} {clean_site}, excision (fs):\n\n-"

        # --- CONFIDENCE FLAG ---
        flag = confidence_flag(text)
        if flag:
            structured = structured + f"\n{flag}"

        outputs.append(structured)

    return "\n".join(outputs)

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
