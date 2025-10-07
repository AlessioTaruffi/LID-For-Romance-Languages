from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import os

#configurazione parametri
MODEL_NAME = "xlm-roberta-base"
NUM_EPOCHS = 3 # Numero di epoche per il fine-tuning
BATCH_SIZE = 8 # Dimensione del batch
MAX_LENGTH = 128 # Lunghezza massima dei token
OUTPUT_DIR = "./xlmr_lid_output_breve"

#caricamento dataset
print("caricamento CSV")
#dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"}) 
dataset = load_dataset("csv", data_files={"train": "trainBreve.csv", "test": "testBreve.csv"}) 

#encoding delle label
print("encoding labels")
label_encoder = LabelEncoder() 
all_labels = list(dataset["train"]["label"]) + list(dataset["test"]["label"]) #list necessario in quanto converte da DatasetColumn
label_encoder.fit(all_labels)

def encode_labels(example):
    example["label"] = label_encoder.transform([example["label"]])[0]
    return example

dataset = dataset.map(encode_labels)

#tokenizzazione dei testi
print("tokenizzazione testi")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

dataset = dataset.map(tokenize)

#modellazione del modello
print("caricamento modello")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_encoder.classes_)
)

#computazione di metriche per una prima evaluation
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(pred.label_ids, preds),
        "f1": f1_score(pred.label_ids, preds, average="macro")
    }

#argomenti di training
print("configurazione training arguments")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch", # Valutazione ad ogni epoca
    save_strategy="epoch", # Salvataggio del modello ad ogni epoca
    per_device_train_batch_size=BATCH_SIZE, # Dimensione del batch per il training
    per_device_eval_batch_size=BATCH_SIZE, # Dimensione del batch per l'evaluation
    num_train_epochs=NUM_EPOCHS, # Numero di epoche
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=10, # Log ogni 10 passi
    load_best_model_at_end=True, # Carica il miglior modello alla fine del training
    metric_for_best_model="f1", # Usa F1 come metrica per il miglior modello
    save_total_limit=1 # Limita il numero di modelli salvati
)

# definizione del trainer
print("configurazione trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

#training effettivo
print("inizio del training")
trainer.train()

#salvataggio delle labels
print("salvataggio labels")
with open(os.path.join(OUTPUT_DIR, "label_mapping.txt"), "w") as f:
    for i, label in enumerate(label_encoder.classes_):
        f.write(f"{i}: {label}\n")

print("Fine-tuning completato.")
