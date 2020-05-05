import numpy as np
import tensorflow
# from nltk.stem.lancaster import LancasterStemmer
# import nltk
from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from functions import load_dataset, cleaning, create_tokenizer, max_length, encoding_doc, padding_doc, one_hot, create_model

def train():
    filename = './data.csv'

    intent, unique_intent, sentences = load_dataset(filename)
    cleaned_words = cleaning(sentences)

    word_tokenizer = create_tokenizer(cleaned_words)
    vocab_size = len(word_tokenizer.word_index) + 1
    max_len = max_length(cleaned_words)

    encoded_doc = encoding_doc(word_tokenizer, cleaned_words)
    padded_doc = padding_doc(encoded_doc, max_len)

    output_tokenizer = create_tokenizer(unique_intent, filters = '!"#$%&()*+,-/:;<=>?@[\]^`{|}~')

    encoded_output = encoding_doc(output_tokenizer, intent)
    encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)
    output_one_hot = one_hot(encoded_output)

    train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot, shuffle = True, test_size = 0.2)

    model = create_model(vocab_size, max_len)
    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    filename = 'model.h5'

    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    hist = model.fit(train_X, train_Y, epochs = 100, batch_size = 32, validation_data = (val_X, val_Y), callbacks = [checkpoint])

train()