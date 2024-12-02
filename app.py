import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Model path
MODEL_PATH = "./fine_tuned_model"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

st.title('Fine Tuned model for Question Answering')
st.write("Interact with the model..")

tokenizer, model = load_model()
nlp_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

input_text = st.text_input("Enter your text:")

if input_text:
    try:
        result = nlp_pipeline(input_text)
        st.write(f"Prediction: {result[0]['label']} with score {result[0]['score']:.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
