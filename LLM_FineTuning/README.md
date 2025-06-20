# LLM_finetuning

This repo contains two scripts designed for preparing datasets to fine-tune LLMs in the food domain. 
All labeling is done using the **BIO format**, based on specific project requirements.

---

## 🥣 1. BIO Ingredient Tagging (`bio_formatter.py`)

Uses an LLM (e.g., `mistral:7b-instruct` via Ollama) to tag ingredients in food descriptions using the BIO scheme.

### BIO Tags:
- `B-ING`: Beginning of ingredient  
- `I-ING`: Inside ingredient  
- `O`: Not an ingredient  

### Input:
CSV file with a `description` column.

### Output:
`BIO_outputfile.txt` – each word per line with its BIO label:
```
carrots    B-ING  
and        O  
peas       B-ING
```

---

## 🔁 2. Siamese Sentence Pair Generator (`siamese_pair_generator.py`)

Generates sentence pairs for training similarity models (like Siamese networks).

### How it works:
- Uses an LLM (e.g., `gemma:2b`) to generate user-style tags for food descriptions.
- Pairs those with the original descriptions as positive pairs.
- Samples other descriptions as negatives.

### Output:
`siamese_pairs.json` with pairs like:
```json
{
  "sentence1": "Creamy tomato soup",
  "sentence2": "vegetarian dinner",
  "label": 1.0
}
```

---

## 🛠️ Requirements
- Python 3.8+
- Ollama installed and running locally
- Models: `mistral:7b-instruct`, `gemma:2b`

Install dependencies:
```bash
pip install ollama
```

---

## 📌 Note

All labeling uses the BIO format to meet project-specific requirements related to ingredient-level tagging.

---

## 🗂 Structure
```
LLM_FineTuning/
├── DataFormatting/
│   ├── bio_formatter.py               # BIO tagging using LLM
│   ├── siamese_pair_generator.py      # Generate similarity sentence pairs
│   ├── your_own_dataset.csv           # Input file with descriptions
│   ├── BIO_outputfile.txt             # BIO-labeled token output
│   └── siamese_pairs.json             # Sentence pairs with similarity labels
│
├── TrainingModel/
│   └── ...                            # Scripts & notebooks for fine-tuning models
```

---
This project explores LLM-based data preparation for food/NLP tasks.
