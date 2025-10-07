ROMANCE_LANGUAGES = {"ita", "fra", "spa", "por", "ron"}

def filter_wili(input_x, input_y, output_x, output_y):
    with open(input_x, "r", encoding="utf-8") as f_x, \
         open(input_y, "r", encoding="utf-8") as f_y:
        texts = f_x.readlines()
        labels = f_y.readlines()

    assert len(texts) == len(labels), "Mismatch between text and label lines"

    filtered_texts = []
    filtered_labels = []

    for text, label in zip(texts, labels):
        label = label.strip()
        if label in ROMANCE_LANGUAGES:
            filtered_texts.append(text.strip())
            filtered_labels.append(label)

    print(f"Kept {len(filtered_texts)} samples from {input_x}")

    with open(output_x, "w", encoding="utf-8") as f_x_out, \
         open(output_y, "w", encoding="utf-8") as f_y_out:
        for text, label in zip(filtered_texts, filtered_labels):
            f_x_out.write(text + "\n")
            f_y_out.write(label + "\n")

if __name__ == "__main__":
    #training set
    filter_wili(
        input_x="C:/Users/aless/OneDrive/Desktop/HWp/dataset/x_train.txt",
        input_y="C:/Users/aless/OneDrive/Desktop/HWp/dataset/y_train.txt",
        output_x="x_train_romance.txt",
        output_y="y_train_romance.txt"
    )

    #test set
    filter_wili(
        input_x="C:/Users/aless/OneDrive/Desktop/HWp/dataset/x_test.txt",
        input_y="C:/Users/aless/OneDrive/Desktop/HWp/dataset/y_test.txt",
        output_x="x_test_romance.txt",
        output_y="y_test_romance.txt"
    )
