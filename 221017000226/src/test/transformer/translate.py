from transformers import MarianMTModel, MarianTokenizer
import argparse

parser = argparse.ArgumentParser(description='Translation')
parser.add_argument('--words', type=str, default='The cat ate the mouse', help='input english text')
parser.add_argument('--debug', action='store_true', default=False, help='enable debug mode')
args = parser.parse_args()

# Load pre-trained English-to-Chinese translation model
model_name = "Helsinki-NLP/opus-mt-en-zh"
model = MarianMTModel.from_pretrained(model_name)

if args.debug:
    print(model)

tokenizer = MarianTokenizer.from_pretrained(model_name)

if args.debug:
    print("tokenizer", tokenizer)

# Define a function for translation
def translate_text(text, model, tokenizer):
    # Tokenize the input text
    input_ids = tokenizer.encode(text, return_tensors="pt")

    if args.debug:
        print("encode", input_ids[0])

    # Generate translation
    translation_ids = model.generate(input_ids)

    if args.debug:
        print("translation", translation_ids[0])

    # Decode the translated text
    translated_text = tokenizer.decode(translation_ids[0], skip_special_tokens=True)
    return translated_text

# Example English text
english_text = "The cat ate the mouse."

# Translate to Chinese
chinese_translation = translate_text(args.words, model, tokenizer)

# Print the results
print(chinese_translation)

