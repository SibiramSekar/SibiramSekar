import ollama
import csv

ollama_model = "mistral:7b-instruct"
input_csv = "your_own_dataset.csv"
output_file = "BIO_outputfile.txt"

response = ollama.chat(model="mistral:7b-instruct", messages=[
    {"role": "user", "content": "Say only: hello world"}
])
print(response['message']['content'])

valid_labels = {"B-ING", "I-ING", "O"}

prompt_template = """
You are an expert at tagging INGREDIENTS in food descriptions using BIO format.

Instructions:
- Tag each token (word) in the sentence.
- Use these labels ONLY:
  - B-ING (Beginning of an ingredient)
  - I-ING (Inside an ingredient)
  - O (Other, not part of any ingredient)

Format:
- One word per line.
- Each line = token<TAB>label (e.g., rice<TAB>B-ING)
- No colons, quotes, section headers, or punctuation-only lines.
- Do NOT invent ingredients not in the sentence.

Examples:

Spicy\tO  
lentil\tB-ING  
soup\tI-ING  
with\tO  
garlic\tB-ING  
and\tO  
onions\tB-ING

crispy\tO  
fried\tO  
rice\tB-ING  
with\tO  
soy\tB-ING  
sauce\tI-ING  
and\tO  
carrots\tB-ING

sweet\tO  
corn\tB-ING  
soup\tI-ING  
with\tO  
chopped\tO  
peas\tB-ING  
and\tO  
leeks\tB-ING

Now label this:
{sentence}
"""

def call_ollama(prompt):
    response = ollama.chat(model=ollama_model, messages=[
        {"role": "user", "content": prompt}
    ])
    return response['message']['content']

with open(input_csv, newline='', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    for row in reader:
        sentence = row['description'].strip()
        prompt = prompt_template.format(sentence=sentence)
        raw_output = call_ollama(prompt)

        cleaned_lines = []
        for line in raw_output.strip().splitlines():
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                print(f"[WARN] Skipping malformed line: {line}")
                continue

            *token_parts, label = parts
            token = " ".join(token_parts)

            if label in valid_labels:
                cleaned_lines.append(f"{token}\t{label}")
            else:
                print(f"[WARN] Skipping invalid label: {label}")

        if cleaned_lines:
            outfile.write("\n".join(cleaned_lines))
            outfile.write("\n\n")
            print(f" Labeled: {sentence[:60]}...")
        else:
            print(f"[ERROR] No valid lines returned for: {sentence[:60]}...")
