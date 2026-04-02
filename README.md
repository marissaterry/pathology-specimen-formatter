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

## Setup

Create a .env file:
OPENAI_API_KEY=your_api_key_here

## Run the App

streamlit run ui.py
Then open the local URL shown in your terminal.

## Methodology

This system uses:
Rule-based NLP for structure
Context memory for sequential specimens
Controlled LLM usage for optional refinement
This hybrid approach improves reliability over LLM-only systems.

## Status

Accuracy: ~95–99% (internal testing)
Designed for: Head & neck pathology workflows

## Disclaimer

This tool is for research and workflow assistance only.
Not intended for independent clinical decision-making.

## Future Work

Multi-organ system support
Larger validation dataset
LIS integration
Fine-tuned models

## Author

Marissa Terry
Mayo Clinic Alix School of Medicine
terry.marissa@mayo.edu

