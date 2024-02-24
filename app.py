from flask import Flask, render_template, request, jsonify

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("abdulwaheed1/english-to-urdu-translation-mbart")
model = AutoModelForSeq2SeqLM.from_pretrained("abdulwaheed1/english-to-urdu-translation-mbart")
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        text = data['text']

        # Tokenize input text
        inputs = tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)

        # Generate translation
        translation_ids = model.generate(**inputs)
        translation = tokenizer.batch_decode(translation_ids, skip_special_tokens=True)[0]

        return jsonify({'translation': translation})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
