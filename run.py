import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
import pickle

# Load the tokenizer
with open('assembly_tokenizer.pkl', 'rb') as f:
    assembly_tokenizer, max_length_assembly = pickle.load(f)

with open('c_tokenizer.pkl', 'rb') as f:
    c_tokenizer, max_length_c = pickle.load(f)

assembly_vocab_size = len(assembly_tokenizer.word_index) + 1
c_vocab_size = len(c_tokenizer.word_index) + 1

# Transformer Model
def transformer_model(input_vocab_size, target_vocab_size, max_length_input, max_length_target, d_model, num_heads, dff, num_layers):
    # Encoder
    encoder_inputs = Input(shape=(max_length_input,))
    encoder_padding_mask = tf.keras.layers.Lambda(lambda inputs: tf.cast(tf.math.equal(inputs, 0), dtype=tf.float32))(encoder_inputs)
    encoder_embedding = Embedding(input_vocab_size, d_model)(encoder_inputs)
    encoder_embedding *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    encoder_embedding = Masking(mask_value=0.0)(encoder_embedding)
    
    encoder_outputs = encoder_embedding
    for i in range(num_layers):
        encoder_outputs = encoder_layer(encoder_outputs, num_heads, dff, d_model, dropout_rate=0.1)

    # Decoder
    decoder_inputs = Input(shape=(max_length_target,))
    decoder_padding_mask = tf.keras.layers.Lambda(lambda inputs: tf.cast(tf.math.equal(inputs, 0), dtype=tf.float32))(decoder_inputs)
    decoder_embedding = Embedding(target_vocab_size, d_model)(decoder_inputs)
    decoder_embedding *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    decoder_embedding = Masking(mask_value=0.0)(decoder_embedding)
    
    decoder_outputs = decoder_embedding
    for i in range(num_layers):
        decoder_outputs = decoder_layer(decoder_outputs, encoder_outputs, num_heads, dff, d_model, dropout_rate=0.1)
        
    decoder_outputs = Dense(target_vocab_size)(decoder_outputs)
    
    return Model(encoder_inputs, decoder_outputs)

def encoder_layer(x, num_heads, dff, d_model, dropout_rate=0.1):
    # Multi-Head Attention
    attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
    attn_output = attention(x, x)
    attn_output = tf.keras.layers.Dropout(dropout_rate)(attn_output)
    # Add & Normalize
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
    # Feed Forward
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])
    ffn_output = ffn(out1)
    ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output)
    # Add & Normalize
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    return out2

def decoder_layer(x, enc_output, num_heads, dff, d_model, dropout_rate=0.1):
    # Masked Multi-Head Attention (self-attention)
    attention1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
    attn_output1 = attention1(x, x, attention_mask=create_look_ahead_mask(tf.shape(x)[1]))
    attn_output1 = tf.keras.layers.Dropout(dropout_rate)(attn_output1)
    # Add & Normalize
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attn_output1 + x)
    # Multi-Head Attention (encoder-decoder attention)
    attention2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
    attn_output2 = attention2(out1, enc_output)
    attn_output2 = tf.keras.layers.Dropout(dropout_rate)(attn_output2)
    # Add & Normalize
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attn_output2 + out1)
    # Feed Forward
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])
    ffn_output = ffn(out2)
    ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output)
    # Add & Normalize
    out3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ffn_output + out2)
    return out3

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

# Model Parameters
d_model = 64
num_heads = 4
dff = 64
num_layers = 2

model = transformer_model(assembly_vocab_size, c_vocab_size, max_length_assembly, max_length_c, d_model, num_heads, dff, num_layers)

# Train the model
model.load_weights('saved_model/assembly_to_c_model.ckpt').expect_partial()

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    print(f"Sequence len: {len(input_seq[0])}, max len: {max_length_assembly}")
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=max_length_assembly)

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
with open("training_data/re_modified/1/asm/6000.asm") as f:
    in_test = f.read()

test_input_seq = assembly_tokenizer.texts_to_sequences([in_test])
decoded_sentence = decode_sequence(test_input_seq)
print('Decoded C code:', decoded_sentence)
