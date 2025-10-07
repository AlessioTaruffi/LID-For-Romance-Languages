import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib

#caricamento dati di train e test
#modificare il percorso se necessario
print("caricamento dati di train e test")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("testBreve.csv")

print("encoding delle label")
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df["label"])
y_test = label_encoder.transform(test_df["label"])

print("creazione pipeline TF-IDF + SVM...")
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer='char', ngram_range=(3, 5))),
    ("clf", LinearSVC())
])

#Training vero e proprio
print("addestramento")
pipeline.fit(train_df["text"], y_train)

#evaluation
print("valutaziojne")
y_pred = pipeline.predict(test_df["text"])

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"F1 Macro: {f1:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#salvataggio del modello e dell'encoder
joblib.dump(pipeline, "svm_baseline_model.pkl")
joblib.dump(label_encoder, "svm_label_encoder.pkl")
