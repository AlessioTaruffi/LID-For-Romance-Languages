import os
import pandas as pd
import random

DATASET_FOLDER = "./datasetBreve"
LANGUAGE_DATA = {}

for filename in os.listdir(DATASET_FOLDER):
    filepath = os.path.join(DATASET_FOLDER, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            _, lang, sentence = parts
            if lang not in LANGUAGE_DATA:
                LANGUAGE_DATA[lang] = []
            LANGUAGE_DATA[lang].append(sentence)

SELECTED_DATA = {}
for lang, sentences in LANGUAGE_DATA.items():
    if len(sentences) < 1000:
        raise ValueError(f"La lingua {lang} ha solo {len(sentences)} frasi, ne servono almeno 1000.")
    SELECTED_DATA[lang] = random.sample(sentences, 1000)

train_rows = []
test_rows = []

for lang, sentences in SELECTED_DATA.items():
    random.shuffle(sentences)
    train = sentences[:500]
    test = sentences[500:1000]
    train_rows.extend([(s, lang) for s in train])
    test_rows.extend([(s, lang) for s in test])

train_df = pd.DataFrame(train_rows, columns=["text", "label"])
test_df = pd.DataFrame(test_rows, columns=["text", "label"])

train_df.to_csv("trainBreve.csv", index=False)
test_df.to_csv("testBreve.csv", index=False)
print("Dataset cleaned and split into train and test sets.")
