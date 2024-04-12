import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

# Example datasets (replace with your own)
telugu_sentences = df3['telugu'].tolist()
kannada_sentences = df3['kannada'].tolist()

# max_length = max(len(telugu_sequences), len(kannada_sequences))
# telugu_sequences = pad_sequences(telugu_sequences, maxlen=max_length, padding='post')
# kannada_sequences = pad_sequences(kannada_sequences, maxlen=max_length, padding='post')

# Tokenization
telugu_tokenizer = Tokenizer()
telugu_tokenizer.fit_on_texts(telugu_sentences)
telugu_sequences = telugu_tokenizer.texts_to_sequences(telugu_sentences)
telugu_max_len = max([len(seq) for seq in telugu_sequences])
telugu_sequences = pad_sequences(telugu_sequences, maxlen=telugu_max_len, padding='post')

kannada_tokenizer = Tokenizer()
kannada_tokenizer.fit_on_texts(kannada_sentences)
kannada_sequences = kannada_tokenizer.texts_to_sequences(kannada_sentences)
kannada_max_len = max([len(seq) for seq in kannada_sequences])
kannada_sequences = pad_sequences(kannada_sequences, maxlen=kannada_max_len, padding='post')
kannada_tokenizer.word_index['<start>'] = len(kannada_tokenizer.word_index) + 1
kannada_tokenizer.word_index['<end>'] = len(kannada_tokenizer.word_index) + 1

# Define the model
encoder_input = Input(shape=(None,))
encoder_embedding = Embedding(len(telugu_tokenizer.word_index)+1, 256, mask_zero=True)(encoder_input)
encoder_outputs, state_h, state_c = LSTM(256, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_input = Input(shape=(None,))
decoder_embedding = Embedding(len(kannada_tokenizer.word_index)+1, 256, mask_zero=True)(decoder_input)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(len(kannada_tokenizer.word_index)+1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_input, decoder_input], decoder_outputs)

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit([telugu_sequences, kannada_sequences[:, :-1]], 
          np.expand_dims(kannada_sequences[:, 1:], axis=-1), 
          batch_size=64, epochs=50)

# Inference
encoder_model = Model(encoder_input, encoder_states)

decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_input] + decoder_states_inputs, [decoder_outputs] + decoder_states)

def translate_telugu_to_kannada(input_text):
    input_seq = telugu_tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=telugu_max_len, padding='post')
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = kannada_tokenizer.word_index['<start>']
    stop_condition = False
    translated_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = kannada_tokenizer.index_word.get(sampled_token_index, '<unk>')
        if sampled_char == '<end>' or len(translated_sentence.split()) > kannada_max_len:
            stop_condition = True
        else:
            translated_sentence += sampled_char + ' '
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return translated_sentence.strip()


# Example translation
telugu_input_text = 'మీరు ఎలా ఉన్నారు'
kannada_translation = translate_telugu_to_kannada(telugu_input_text)
print(f'Translation: {kannada_translation}')