import csv
import json
import random
import ollama

OLLAMA_MODEL = "gemma:2b"
INPUT_CSV = "your_own_dataset.csv"
OUTPUT_JSON = "siamese_pairs.json"

PROMPT_TEMPLATE = """
Generate 2 to 3 user-style search phrases for this food description.
These should describe how someone might search for it.

Example:
Description: "Spinach masala is a savory Indian dish featuring spinach leaves and spices."
Tags: hearty but not heavy, nutritious Indian curry, vegetarian dinner

Now do the same for:
"{description}"

Only output a comma-separated list of short tags.
"""

def call_ollama(prompt):
    response = ollama.chat(model=OLLAMA_MODEL, messages=[
        {"role": "user", "content": prompt}
    ])
    return response['message']['content']

print(f"Reading input from '{INPUT_CSV}'...")
with open(INPUT_CSV, newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    descriptions = [row['description'] for row in reader]

print(f"Found {len(descriptions)} descriptions.")
pairs = []

for i, desc in enumerate(descriptions, 1):
    print(f"\n[{i}/{len(descriptions)}] Processing description:")
    print(f"  → {desc[:80]}...")

    prompt = PROMPT_TEMPLATE.format(description=desc)
    tag_string = call_ollama(prompt)
    tags = [t.strip() for t in tag_string.split(",") if t.strip()]

    if not tags:
        print("  No tags generated, skipping this item.")
        continue

    print(f" Tags: {tags}")

    for tag in tags:
        pairs.append({
            "sentence1": desc,
            "sentence2": tag,
            "label": 1.0
        })
        print(f"  [+] Pos pair: '{desc[:30]}...' <-> '{tag}'")

    negatives = random.sample([d for d in descriptions if d != desc], min(2, len(descriptions) - 1))
    for neg in negatives:
        pairs.append({
            "sentence1": desc,
            "sentence2": neg,
            "label": 0.0
        })
        print(f"  [-] Neg pair: '{desc[:30]}...' <-> '{neg[:30]}...'")


print(f"\nWriting {len(pairs)} sentence pairs to '{OUTPUT_JSON}'...")
with open(OUTPUT_JSON, "w", encoding='utf-8') as outfile:
    json.dump(pairs, outfile, indent=2)

print(" Done!")
