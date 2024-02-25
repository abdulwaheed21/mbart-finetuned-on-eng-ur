import streamlit as st
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Function to load model
@st.cache(allow_output_mutation=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("abdulwaheed1/english-to-urdu-translation-mbart")
    model = AutoModelForSeq2SeqLM.from_pretrained("abdulwaheed1/english-to-urdu-translation-mbart")
    return tokenizer, model

# Custom CSS to style the GUI
st.markdown(
    """
    <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #4b6cb7;
        color: #fff;
        margin: 0;
        padding: 0;
    }
    .container {
        max-width: 800px;
        margin: 50px auto;
        background-color: #fff;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    h1 {
        color: #007bff;
        text-align: center;
    }
    label, .text-white {
        color: #333;
        font-weight: bold;
    }
    textarea {
        width: 100%;
        resize: none;
        border: 1px solid #ced4da;
        padding: 10px;
        margin-top: 10px;
    }
    button {
        width: 100%;
        cursor: pointer;
        transition: background-color 0.3s ease, box-shadow 0.3s ease;
        padding: 12px;
        font-size: 18px;
        border-radius: 8px;
        border: none;
        background-color: #007bff;
        color: #fff;
    }
    button:hover {
        background-color: #0056b3;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    #translation-result {
        margin-top: 10px;
        border: 1px solid #ced4da;
        padding: 10px;
        width: 100%;
        color: #333;
    }
    footer {
        text-align: center;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    # Remove special characters and extra spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Create or load model
tokenizer, model = load_model()

# Creating a container for the main content
with st.container():
    st.title("English to Urdu Translation")

    # Display input field for English text
    st.subheader("Enter English Text:")
    english_text = st.text_area("", height=200)

    # Translate button
    if st.button("Translate"):
        if english_text:
            # Preprocess text
            english_text = preprocess_text(english_text)

            # Tokenize input text
            inputs = tokenizer(english_text, return_tensors="pt", max_length=1024, truncation=True)

            # Generate translation
            translation_ids = model.generate(**inputs)
            translation = tokenizer.batch_decode(translation_ids, skip_special_tokens=True)[0]

            # Display translated text
            st.subheader("Translation:")
            st.text_area("", value=translation, height=200)
        else:
            st.warning("Please enter some text to translate.")

# Footer
st.markdown(
    """
    <footer>
        <p>© 2024 Translation App. All rights reserved.</p>
    </footer>
    """,
    unsafe_allow_html=True,
)
