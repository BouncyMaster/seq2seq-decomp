import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
import pickle

input_assembly = []
output_c_code = []

# Load data (1)
for i in range(0,10000):
    with open(f"training_data/re_modified/1/asm/{i}.asm") as f:
        input_assembly.append(f.read())
    with open(f"training_data/re_modified/1/c/{i}.c") as f:
        output_c_code.append(f.read())

# Load data (2)
for i in range(0,10000):
    with open(f"training_data/re_modified/2/asm/{i}.asm") as f:
        input_assembly.append(f.read())
    with open(f"training_data/re_modified/2/c/{i}.c") as f:
        output_c_code.append(f.read())

output_c_code = ['<start> ' + seq + ' <end>' for seq in output_c_code]

# Tokenization
assembly_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
c_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

assembly_tokenizer.fit_on_texts(input_assembly)
c_tokenizer.fit_on_texts(output_c_code)

assembly_vocab_size = len(assembly_tokenizer.word_index) + 1
c_vocab_size = len(c_tokenizer.word_index) + 1

# Padding sequences
input_sequences = assembly_tokenizer.texts_to_sequences(input_assembly)

max_length_assembly = max(map(len, input_sequences))
input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_length_assembly)

output_sequences = c_tokenizer.texts_to_sequences(output_c_code)

max_length_c = max(map(len, output_sequences))
output_sequences = tf.keras.preprocessing.sequence.pad_sequences(output_sequences, maxlen=max_length_c)

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
    
    return Model([encoder_inputs, decoder_inputs], decoder_outputs)

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
dff = 128
num_layers = 2

model = transformer_model(assembly_vocab_size, c_vocab_size, max_length_assembly, max_length_c, d_model, num_heads, dff, num_layers)

# Compile the model
model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True))

# Train the model
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard()

model.fit([input_sequences, output_sequences], output_sequences, batch_size=64, epochs=20, validation_split=0.2, callbacks=[callback, tensorboard_callback])
# TODO: try with slicing
#model.fit([input_sequences, output_sequences[:, :-1]], output_sequences[:, 1:], batch_size=64, epochs=20, validation_split=0.2, callbacks=[callback, tensorboard_callback])

# Save the model
model.save_weights('saved_model/assembly_to_c_model.ckpt')

# Save the tokenizer
with open('assembly_tokenizer.pkl', 'wb') as f:
    pickle.dump((assembly_tokenizer, max_length_assembly), f)

with open('c_tokenizer.pkl', 'wb') as f:
    pickle.dump((c_tokenizer, max_length_c), f)
