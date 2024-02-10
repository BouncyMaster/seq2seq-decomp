import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
import numpy as np
import pickle

input_assembly = []
output_c_code = []

# Load data
for i in range(1,10000):
    with open(f"training_data/re/tst_1/asm_src/rd_{i}.s") as f:
        input_assembly.append(f.read())
    with open(f"training_data/re/tst_1/c_src/rd_{i}.cc") as f:
        output_c_code.append(f.read())

output_c_code = ['<start> ' + seq + ' <end>' for seq in output_c_code]

# Tokenization
assembly_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
c_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

assembly_tokenizer.fit_on_texts(input_assembly)
c_tokenizer.fit_on_texts(output_c_code)

assembly_vocab_size = len(assembly_tokenizer.word_index) + 1
c_vocab_size = len(c_tokenizer.word_index) + 1

max_length_assembly = max([len(seq.split()) for seq in input_assembly])
max_length_c = max([len(seq.split()) for seq in output_c_code])

# Padding sequences
input_sequences = assembly_tokenizer.texts_to_sequences(input_assembly)
input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_length_assembly, padding='post')

output_sequences = c_tokenizer.texts_to_sequences(output_c_code)
output_sequences = tf.keras.preprocessing.sequence.pad_sequences(output_sequences, maxlen=max_length_c, padding='post')

# Creating the Seq2Seq model
latent_dim = 256

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

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
model.fit([input_sequences, output_sequences], output_sequences, batch_size=64, epochs=500, validation_split=0.2)

# Save the model
model.save_weights('saved_model/assembly_to_c_model.ckpt')

# Save the tokenizer
with open('assembly_tokenizer.pkl', 'wb') as f:
    pickle.dump((assembly_tokenizer, max_length_assembly), f)

with open('c_tokenizer.pkl', 'wb') as f:
    pickle.dump((c_tokenizer, max_length_c), f)
