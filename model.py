import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Dense
from keras.layers.recurrent import LSTM
from keras.regularizers import l2

class LSTMModel:
    def __init__(self, keywords, vocab_dim, win_size):
        self.keywords = keywords
        self.keyword_size = len(keywords)
        self.vocab_size = self.keyword_size + win_size + 1
        self.win_size = win_size
        self.vocab_dim = vocab_dim
        self.model = self._build_model(vocab_dim, win_size)

    def _build_model(self, vocab_dim, win_size):
        model = Sequential()
        model.add(Embedding(self.vocab_size, vocab_dim, input_length=win_size))
        model.add(LSTM(1024, input_length=win_size, return_sequences=True))
        model.add(Dense(self.vocab_size, activation='softmax',
                        W_regularizer=l2(0)))
        model.compile(optimizer='adagrad', lr=0.001, metrics=['accuracy'],
                      loss='categorical_crossentropy')
        return model

    def train(self, X, y):
        """
        :type X: numpy array (batch, win_size)
        :type y: numpy array (batch,)
        :rtype: loss (accuracy=True)
        """
        B = y.shape[0]
        Y = np.zeros((B, self.vocab_size),
                     dtype=np.int)
        Y[np.arange(B), y] = 1
        return self.model.train_on_batch(X, Y)

    def predict(self, X):
        return self.model.predict(X)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
