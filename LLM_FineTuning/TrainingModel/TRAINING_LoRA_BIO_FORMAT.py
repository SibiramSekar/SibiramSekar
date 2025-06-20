from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from peft import get_peft_model, LoraConfig, TaskType
import torch
import os

MODEL_NAME = "distilbert-base-uncased"
BIO_FILE = "BIO_outputfile.txt"
SAVE_DIR = "ingredient-gemma-finetuned"

cleaned_lines = []
with open("BIO_outputfile.txt", "r", encoding="utf-8") as infile:
    for line in infile:
        parts = line.strip().split()
        if len(parts) == 2:
            token, label = parts
            if label in ['B-ING', 'I-ING', 'O']:
                cleaned_lines.append(f"{token}\t{label}")
            else:
                print(f"[WARN] Skipping invalid label: {label}")
        else:
            print(f"[WARN] Skipping malformed line: {line.strip()}")

with open("BIO_outputfile_cleaned.txt", "w", encoding="utf-8") as outfile:
    outfile.write("\n".join(cleaned_lines))


BIO_FILE = "BIO_outputfile_cleaned.txt"

def load_bio_data(file_path):
    sentences, labels = [], []
    with open(file_path, encoding='utf-8') as f:
        tokens, tags = [], []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
                continue
            if '\t' in line:
                token, tag = line.split('\t')
                tokens.append(token)
                tags.append(tag)
            else:
                print(f"[WARN] Skipping malformed line: {line}")
    return Dataset.from_dict({"tokens": sentences, "ner_tags": labels})

unique_tags = ['O', 'B-ING', 'I-ING']
label2id = {label: idx for idx, label in enumerate(unique_tags)}
id2label = {idx: label for label, idx in label2id.items()}

print("[1/5] Loading BIO-tagged dataset...")
dataset = load_bio_data(BIO_FILE)
print(f"Loaded {len(dataset)} samples.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def encode_batch(example):
    tokens = example['tokens']
    labels = example['ner_tags']
    encodings = tokenizer(tokens, is_split_into_words=True, truncation=True, padding='max_length')
    word_ids = encodings.word_ids()
    enc_labels = [-100 if word_id is None else label2id[labels[word_id]] for word_id in word_ids]
    encodings['labels'] = enc_labels
    return encodings

print("[2/5] Tokenizing and encoding data...")
encoded_dataset = dataset.map(encode_batch, batched=True)
print("Tokenization complete.")

print("[3/5] Loading model and applying LoRA...")
base_model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=len(unique_tags), 
    id2label=id2label, 
    label2id=label2id
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.TOKEN_CLS
)
model = get_peft_model(base_model, peft_config)
print("LoRA applied to model.")

data_collator = DataCollatorForTokenClassification(tokenizer)

args = TrainingArguments(
    output_dir=SAVE_DIR,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=4,
    save_strategy="epoch",
    evaluation_strategy="no",
    fp16=torch.cuda.is_available()
)

print("[4/5] Starting training...")
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
print(" Training complete.")


print("[5/5] Saving model to:", SAVE_DIR)
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(" Model and tokenizer saved.")