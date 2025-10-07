from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import torch


# configurazione parametri

MODEL_DIR = "./xlmr_lid_output/checkpoint-626"
#MODEL_DIR = "./xlmr_lid_output_breve/checkpoint-939"
#TEST_CSV = "test.csv"
TEST_CSV = "testBreve.csv"
LABEL_MAP_FILE = "./xlmr_lid_output/label_mapping.txt"
#LABEL_MAP_FILE = "./xlmr_lid_output_breve/label_mapping.txt"
MAX_LEN = 128 #lunghezza massima dei token 

#caricamento modello e tokenizer
print("Caricamento modello finetuned")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR) 
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval() # imposta il modello in modalità valutazione

#caricamento label mapping
print("caricamento label mapping")
label_map = {}
with open(LABEL_MAP_FILE, "r", encoding="utf-8") as f:
    for line in f:
        idx, lang = line.strip().split(":")
        label_map[int(idx)] = lang.strip()

#inverso per decodifica
inv_label_map = {v: k for k, v in label_map.items()}

# caricamento del test set
print("caricamento test set")
df = pd.read_csv(TEST_CSV)
texts = df["text"].tolist()
true_labels = [inv_label_map[label] for label in df["label"].tolist()] 

#tokenizzazione dei testi con il tokenizer del modello
print("Tokenizzazione dei testi")
encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")

#calcolo predizioni
print("Calcolo delle predizioni")
with torch.no_grad():
    outputs = model(**encodings) # ottiene le uscite del modello
    probs = outputs.logits # calcola le probabilità
    preds = torch.argmax(probs, dim=1).numpy() # converte le predizioni in numpy array

#decodifica predizioni e calcolo metriche scelte
print("\nRISULTATI\n")
accuracy = accuracy_score(true_labels, preds)
f1 = f1_score(true_labels, preds, average="macro")

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 (macro): {f1:.4f}\n")

print("Classification Report:")
print(classification_report(true_labels, preds, target_names=[label_map[i] for i in range(len(label_map))]))

print("Confusion Matrix:")
print(confusion_matrix(true_labels, preds))
