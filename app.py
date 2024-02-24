!pip install transformers
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("abdulwaheed1/english-to-urdu-translation-mbart")
model = AutoModelForSeq2SeqLM.from_pretrained("abdulwaheed1/english-to-urdu-translation-mbart")

def main():
    st.title("English to Urdu Translation")

    # Input text area
    text = st.text_area("Enter English Text:", height=200)

    # Translate button
    if st.button("Translate"):
        try:
            # Tokenize input text
            inputs = tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)

            # Generate translation
            translation_ids = model.generate(**inputs)
            translation = tokenizer.batch_decode(translation_ids, skip_special_tokens=True)[0]

            # Display translation result
            st.text("Translation:")
            st.text(translation)

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == '__main__':
    main()
