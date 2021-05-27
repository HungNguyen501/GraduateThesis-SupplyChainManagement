# py train.py lstm 50 data/train/data.xlsx
import tensorflow as tf 
import keras
import nltk
import pandas as pd
import numpy as np
import re
import codecs
import pickle
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import sys
from datetime import datetime
import matplotlib.pyplot as plt


now = datetime.now()
time_now = f"""{now:%Y-%m-%d_%H-%M-%S}.{"{:03d}".format(now.microsecond // 1000)}"""
list_models = ['svm', 'lstm', 'bilstm']


try:
    model_name = sys.argv[1]
    EMBEDDING_DIM = int(sys.argv[2])
    train_set_file_path = str(sys.argv[3])   
except Exception as ex:
    print('??? Please type enough parameters: model_name, embedding_dimension and train_set_file_path.')
    exit()

if model_name not in list_models:
    print('??? Your model is not in list.')
    print('??? Please type again.')
    exit()


"""
Load data train
"""
try:
    data = pd.read_excel(train_set_file_path, ignore_index=True)
    data = data.astype(str)
except Exception as ex:
    print('??? ', ex)
    exit()

print('===> Loaded {}'.format(train_set_file_path))


"""
=================Preprocessing-data====================
"""
# Clean data
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer

REPLACE_BY_SPACE_RE = re.compile('[/(){}\\[\\]\\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
        text: a string
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub(' ', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = re.sub('\\d+', ' ', text) # Remove digit number from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() ) # Lemmatization

    return text


"""
tokenizer
"""
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

re_tokenizer = RegexpTokenizer(r'\w+')

# Max number of words in each status_description
MAX_SEQUENCE_LENGTH = 250


def tokenizing(df, EMBEDDING_DIM):
    df["tokens"] = df["status_description"].apply(re_tokenizer.tokenize)

    all_words = [word for tokens in df["tokens"] for word in tokens]
    sentence_lengths = [len(tokens) for tokens in clean_data["tokens"]]
    VOCAB = sorted(list(set(all_words)))

    print('Words total: {}'.format(len(all_words)))
    print('Vocabulary size: {}'.format(len(VOCAB)))
    print('Max sentence length is {}'.format(max(sentence_lengths)) )

    # The maximum number of words to be used. (most frequent)
    VOCAB_SIZE = len(VOCAB) + 1
    # OOV = Out of Vocabulary
    oov_tok = '<OOV>' 
    print('VOCAB_SIZE: {}'.format(VOCAB_SIZE))
    print('MAX_SEQUENCE_LENGTH: {}'.format(MAX_SEQUENCE_LENGTH))
    print('EMBEDDING_DIM: {}\n'.format(EMBEDDING_DIM))

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~', lower=True)
    tokenizer = Tokenizer(num_words = VOCAB_SIZE, oov_token=oov_tok)
    tokenizer.fit_on_texts(df['status_description'].values)

    return df, tokenizer, VOCAB_SIZE


def get_embedding_matrix(word_index, VOCAB_SIZE=0, glove_file_path = 'D:\\MyProjects\\Python_projects\\Global Vectors for Word Representation (GloVe)\\glove.6B.' + str(EMBEDDING_DIM) +'d.txt'):
    f = open(glove_file_path, encoding="utf8")
    embeddings_index = {}

    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            pass
        embeddings_index[word] = coefs
    f.close()
    print('Total %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.random.random((VOCAB_SIZE, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) !=len(embedding_vector):
                print("could not broadcast input array from shape",str(len(embedding_matrix[i])),
                                 "into shape",str(len(embedding_vector))," Please make sure your"
                                 " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                exit(1)

            embedding_matrix[i] = embedding_vector

    return embedding_matrix

print('===> preprocessing data')

# Clean data
data['status_description'] = data['status_description'].apply(clean_text)
clean_data = data

# Tokenizer
clean_data, tokenizer, VOCAB_SIZE = tokenizing(clean_data, EMBEDDING_DIM)
word_index = tokenizer.word_index

# Get embedding_matrix
embedding_matrix = get_embedding_matrix(word_index, VOCAB_SIZE)

# Split data to train set and test set
X = tokenizer.texts_to_sequences(clean_data['status_description'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(clean_data['checkpoint_status']).values
print('Shape of label tensor:', Y.shape)

# Split data to train set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


"""
================================Build models=============================================
"""
def LSTM():
    from keras.models import Sequential
    from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Conv1D, MaxPooling1D

    model = Sequential()
    model.add(Embedding(input_dim=VOCAB_SIZE,
                        output_dim=EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=True))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(EMBEDDING_DIM, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(6, activation='softmax'))

    return model


def BILSTM():
    from keras.models import Sequential
    from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional, Dropout

    model = Sequential()
    model.add(Embedding(input_dim=VOCAB_SIZE,
                        output_dim=EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=True))

    model.add(SpatialDropout1D(0.3))
    model.add(Bidirectional(LSTM(EMBEDDING_DIM, dropout=0.3, recurrent_dropout=0.3)))
    model.add(Dense(EMBEDDING_DIM, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(EMBEDDING_DIM, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(6, activation='softmax'))

    return model


from keras.callbacks import EarlyStopping, ModelCheckpoint

print('===> kick off training '+model_name+' model')

if model_name == 'lstm':
    model = LSTM()
elif model_name == 'bilstm':
    model = BILSTM()    

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

best_model_file_path = 'models/'+model_name+'/'+model_name+'_model.h5'
callbacks = [EarlyStopping(monitor='val_loss', verbose=1, patience=2),
             ModelCheckpoint(filepath=best_model_file_path, monitor='val_loss', save_best_only=True)]

history = model.fit(X_train, Y_train, epochs=100, batch_size=256, validation_split=0.1, callbacks=callbacks)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Load best model
print('===> Loading '+model_name+' best model')
from keras.models import load_model
best_model = load_model('models/'+model_name+'/'+model_name+'_model.h5')


"""
Validate test set
"""
accr = best_model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


"""
Save tokenizer
"""
try:
    tokenizer_file_name = 'models/'+model_name+'/saved_tokenizers/'+model_name+'_tokenizer_'+ time_now +'.pickel'
    pickle.dump(tokenizer, open(tokenizer_file_name, 'wb'))
    pickle.dump(tokenizer, open('models/'+model_name+'/'+model_name+'_tokenizer.pickel', 'wb'))
    print('===> Saved '+model_name+' tokenizer.')
except Exception as ex:
    print(ex)    


"""
Save model
"""
try:
    model_file_name = 'models/'+model_name+'/saved_models/'+model_name+'_model_' + time_now + '.h5'
    best_model.save(model_file_name)
    print('===> Saved '+model_name+' model.')
except Exception as ex:
    print(ex)