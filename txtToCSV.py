import pandas as pd

def convert_to_csv(text_file, label_file, output_csv):
    with open(text_file, "r", encoding="utf-8") as f_text, \
         open(label_file, "r", encoding="utf-8") as f_label:
        texts = [line.strip() for line in f_text.readlines()]
        labels = [line.strip() for line in f_label.readlines()]

    assert len(texts) == len(labels), "Mismatch between text and labels"

    df = pd.DataFrame({
        "text": texts,
        "label": labels
    })

    df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv} with {len(df)} samples.")

if __name__ == "__main__":
    convert_to_csv("./dataset/x_train_romance.txt", "./dataset/y_train_romance.txt", "train.csv")
    convert_to_csv("./dataset/x_test_romance.txt", "./dataset/y_test_romance.txt", "test.csv")