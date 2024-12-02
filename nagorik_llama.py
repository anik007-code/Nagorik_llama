"""this code is responsible fine-tuning the Ollama llama2 model for Imdb movie review dataset. It's a basic implementation of llama model.
. it will depend on CPU architecture."""

import pandas as pd
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline)
from sklearn.model_selection import train_test_split


# Load the CSV Dataset
def load_csv_dataset(file_path):
    print("Loading CSV dataset...")
    df = pd.read_csv(file_path)
    print(f"Dataset Sample:\n{df.head()}")
    return df


# Split the Data
def split_dataset(df, test_size=0.2):
    print("Splitting the dataset into train and test sets...")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"], df["label"], test_size=test_size, random_state=42
    )
    return train_texts.tolist(), test_texts.tolist(), train_labels.tolist(), test_labels.tolist()


# Tokenize the Dataset
def tokenize_data(texts, labels, tokenizer, max_length=512):
    print("Tokenizing the data...")
    tokenized = tokenizer(
        texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
    )
    tokenized["labels"] = labels
    return tokenized


# Load the Model
def load_pretrained_model(model_name, num_labels=2):
    print(f"Loading model: {model_name}")
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


# Define Training Settings
def get_training_arguments(output_dir, epochs=3, batch_size=16):
    print("Defining training arguments...")
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=2,
        load_best_model_at_end=True
    )


# Train the Model
def train_model(model, args, train_data, test_data, tokenizer):
    print("Starting model training...")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
    )
    trainer.train()
    print("Model training completed!")
    return trainer


# Make Predictions
def make_prediction(text, model_path, tokenizer_name):
    print("Making prediction...")
    nlp = pipeline("text-classification", model=model_path, tokenizer=tokenizer_name)
    prediction = nlp(text)
    return prediction[0]


# Main Function to start the training
def main():
    csv_file = "imdb.csv"
    model_name = "meta-llama/Llama-2"
    output_dir = "./fine_tuned_model"
    test_text = "This movie was absolutely fantastic!"


    df = load_csv_dataset(csv_file)
    train_texts, test_texts, train_labels, test_labels = split_dataset(df)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_data = tokenize_data(train_texts, train_labels, tokenizer)
    test_data = tokenize_data(test_texts, test_labels, tokenizer)

    model = load_pretrained_model(model_name)
    training_args = get_training_arguments(output_dir)
    trainer = train_model(model, training_args, train_data, test_data, tokenizer)

    # Predict
    prediction = make_prediction(test_text, output_dir, model_name)
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
