import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras.layers import Embedding, Dense, SimpleRNN
from keras.utils import to_categorical

def prepare_data_and_train(file_name):
    input_text = []

    # Read data from the file
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip().replace('\t', ' ')
            if text:  # Ensure the line has content
                input_text.append(text)
    if not input_text:
        raise ValueError("No valid text found in the input file")

    # Preprocess the text
    def preprocess_text(text):
        return text.lower().replace('.', '').replace('?', '').strip()

    new_input = [preprocess_text(i) for i in input_text]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(new_input)

    # Create input sequences
    input_sequences = []
    for sentence in new_input:
        tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]
        if len(tokenized_sentence) < 2:
            continue  # Ignore sequences with fewer than 2 tokens
        for i in range(1, len(tokenized_sentence)):
            input_sequences.append(tokenized_sentence[:i + 1])

    # Ensure there are input sequences to work with
    if not input_sequences:
        raise ValueError("No valid sequences generated from the text")

    max_len = max([len(seq) for seq in input_sequences])

    # Pad sequences and prepare training data
    padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

    X = padded_input_sequences[:, :-1]
    y = padded_input_sequences[:, -1]

    # One-hot encode the labels
    y = to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

    # Build and compile the model
    model = Sequential([
        Embedding(len(tokenizer.word_index) + 1, 100, input_length=max_len - 1),
        SimpleRNN(150),
        Dense(len(tokenizer.word_index) + 1, activation='softmax'),
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X, y, epochs=10, batch_size=32)

    return model, tokenizer, max_len

# Preprocess the input text
def preprocess_text(text):
    return text.lower().strip()  # Ensure consistent preprocessing
def predict_next_word(model, tokenizer, input_text, max_len):
    # Preprocess the input text
    preprocessed_text = preprocess_text(input_text)
    print("Preprocessed Input Text:", preprocessed_text)

    # Tokenize the preprocessed text
    tokenized_text = tokenizer.texts_to_sequences([preprocessed_text])
    print("Tokenized Text:", tokenized_text)

    # Check if tokenization succeeded
    if not tokenized_text or len(tokenized_text[0]) == 0:
        raise ValueError("Input text could not be tokenized")

    # Pad the tokenized text
    padded_tokenized_text = pad_sequences(tokenized_text, maxlen=max_len - 1, padding='pre')
    print("Padded Tokenized Text:", padded_tokenized_text)

    # Get predictions from the model
    predictions = model.predict(padded_tokenized_text)
    print("Model Predictions:", predictions)

    # Get the index of the word with the highest probability
    pos = np.argmax(predictions[0])
    print("Position of Predicted Word:", pos)

    # Retrieve the corresponding word from the tokenizer's word index
    predicted_word = None
    for word, index in tokenizer.word_index.items():
        if index == pos:
            predicted_word = word
            break

    if predicted_word is None:
        raise ValueError("Predicted word could not be determined from the model's output")

    return predicted_word