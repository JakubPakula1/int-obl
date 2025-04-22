# Small LSTM Network to Generate Text for Alice in Wonderland
import numpy as np
from nltk.tokenize import wordpunct_tokenize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read().lower()
tokenized_text = wordpunct_tokenize(raw_text)
tokens = sorted(list(dict.fromkeys(tokenized_text)))
tok_to_int = dict((c, i) for i, c in enumerate(tokens))

n_tokens = len(tokenized_text)
n_token_vocab = len(tokens)
seq_length = 100
dataX, dataY = [], []
for i in range(0, n_tokens - seq_length, 1):
    seq_in = tokenized_text[i:i + seq_length]
    seq_out = tokenized_text[i + seq_length]
    dataX.append([tok_to_int[tok] for tok in seq_in])
    dataY.append(tok_to_int[seq_out])

n_patterns = len(dataX)
X = np.reshape(dataX, (n_patterns, seq_length, 1)) / float(n_token_vocab)
y = to_categorical(dataY)

# Define the LSTM model
model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile the model with gradient clipping
from tensorflow.keras.optimizers import Adam
model.load_weights("")
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)

model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Define callbacks
filepath = "big-token-model-{epoch:02d}-{loss:.4f}.keras"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
callbacks_list = [checkpoint, early_stopping, reduce_lr]

# Train the model
model.fit(X, y, epochs=200, batch_size=128, callbacks=callbacks_list, validation_split=0.1)