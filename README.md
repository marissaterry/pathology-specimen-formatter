# Pathology Specimen Formatter

AI-assisted tool for converting pathology specimen accession labels into standardized diagnostic lines.

## Overview

This tool automates a common workflow in surgical pathology: transforming specimen labels into structured diagnostic lines.

It uses a **hybrid approach**:
- Rule-based parsing (deterministic, reliable)
- Context-aware logic (handles margins and laterality)
- Optional LLM refinement (for formatting)

## Features

- Deterministic formatting (clinical consistency)
- Context-aware margin handling
- Laterality detection (right, left, bilateral)
- Anatomical normalization (e.g., BOT → base of tongue)
- Level normalization (II, III → II–III)
- Confidence flagging for ambiguous inputs
- Simple web interface (Streamlit)

## Example

**Input:**
A. Right base of tongue
B. Revised lateral margin

**Output:**
A. Oropharynx, right base of tongue, excision:
B. Oropharynx, right base of tongue, excision (fs):

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/pathology-specimen-formatter.git
cd pathology-specimen-formatter
pip3 install -r requirements.txt
