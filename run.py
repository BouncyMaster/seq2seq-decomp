import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
import numpy as np
import pickle

# Load the tokenizer
with open('assembly_tokenizer.pkl', 'rb') as f:
    assembly_tokenizer, max_length_assembly = pickle.load(f)

with open('c_tokenizer.pkl', 'rb') as f:
    c_tokenizer, max_length_c = pickle.load(f)

assembly_vocab_size = len(assembly_tokenizer.word_index) + 1
c_vocab_size = len(c_tokenizer.word_index) + 1

# Creating the Seq2Seq model
latent_dim = 128

# Encoder
encoder_inputs = Input(shape=(max_length_assembly,))
encoder_embedding = tf.keras.layers.Embedding(assembly_vocab_size, latent_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_length_c,))
decoder_embedding = tf.keras.layers.Embedding(c_vocab_size, latent_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(c_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Train the model
model.load_weights('saved_model/assembly_to_c_model.ckpt').expect_partial()

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Populate the first character of target sequence with the start token.
    target_seq[0, 0] = c_tokenizer.word_index['<start>']

    # Sampling loop for a batch of sequences
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = c_tokenizer.index_word[sampled_token_index]

        # Exit condition: either hit max length or find stop token.
        if (sampled_char == '<end>' or
           len(decoded_sentence.split()) > max_length_c):
            stop_condition = True
            break

        decoded_sentence += ' ' + sampled_char

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

# Define encoder model
encoder_model = Model(encoder_inputs, encoder_states)

# Define decoder model
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Sample inference
test_input_seq = assembly_tokenizer.texts_to_sequences(['mov eax, 5'])
decoded_sentence = decode_sequence(test_input_seq)
print('Input assembly: mov eax, 5')
print('Decoded C code:', decoded_sentence)
