from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.sequence import pad_sequences
from transformers import BertForSequenceClassification, BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import tensorflow as tf
import torch
import pickle
import re
import warnings
warnings.filterwarnings("ignore")


REPLACE_BY_SPACE_RE = re.compile("[/()/{/}/[/]|@,;]")
BAD_SYMBOLS_RE = re.compile("[^0-9a-z .]")
STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
re_tokenizer = RegexpTokenizer(r"\w+")


def clean_text(text):
    text = text.lower() # lowercase text
    #text = REPLACE_BY_SPACE_RE.sub(" ", text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub(" ", text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = re.sub("\\d+", " ", text)
    text = re.sub("[.]", " <EOS> ", text)
    text = " ".join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    text = " ".join(lemmatizer.lemmatize(word) for word in text.split() ) # Lemmatization

    return text


def bert_clean_text(text):
    """
        text: a string
        return: modified initial string
    """
    text = text.lower() # lowercase text
    #text = REPLACE_BY_SPACE_RE.sub(" ", text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub(" ", text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = re.sub("\\d+", " ", text)
    # text = re.sub("[.]", " <EOS> ", text)
    # text = " ".join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    # text = " ".join(lemmatizer.lemmatize(word) for word in text.split() ) # Lemmatization

    return text


def text_fit_tokenizer(max_len, text):
    tokenizer = re_tokenizer.tokenize(text)
    len_tokenizer = len(tokenizer)
    temp = tokenizer[(0 if (len_tokenizer - max_len < 0) else len_tokenizer - max_len):len_tokenizer]
    return " ".join(temp)


class CheckWrongOrder:
    def __init__(self):
        self.root_path = "models/CheckLog/"
        self.model_path = self.root_path + "model.h5"
        self.tokenizer_path = self.root_path + "tokenizer.pickel"
        self.MAX_SEQUENCE_LENGTH = 256
        
        try:
            self.tokenizer = pickle.load(open(self.tokenizer_path, "rb"))
        except Exception as e:
            print(e)
            return None
        self.model = load_model(self.model_path)

    def predict(self, text):
        labels = [0, 1]
        li_txt = []

        text = clean_text(text)
        li_txt.append(text_fit_tokenizer(self.MAX_SEQUENCE_LENGTH, text))

        seq = self.tokenizer.texts_to_sequences(li_txt)
        padded = pad_sequences(seq, maxlen=self.MAX_SEQUENCE_LENGTH)
        predictions = self.model.predict(padded)
    
        return np.round(predictions[0].astype(float), decimals=5), labels[np.argmax(predictions)]


class LSTM:
    def __init__(self):
        self.root_path = "models/LSTM/2021-02-07_16-57-40.769/"
        self.model_path = self.root_path + "model.h5"
        self.tokenizer_path = self.root_path + "tokenizer.pickel"
        self.MAX_SEQUENCE_LENGTH = 256
        
        try:
            self.tokenizer = pickle.load(open(self.tokenizer_path, "rb"))
        except Exception as e:
            print(e)
            return None
        self.model = load_model(self.model_path)

    def predict(self, text):
        labels = ["COMPLETED", "DELIVERED_GUARANTEE", "IN_US", "RETURN_TO_SENDER", "TRACKING_AVAILABLE", "TRACKING_ONLINE"]
        li_txt = []

        text = clean_text(text)
        li_txt.append(text_fit_tokenizer(self.MAX_SEQUENCE_LENGTH, text))

        seq = self.tokenizer.texts_to_sequences(li_txt)
        padded = pad_sequences(seq, maxlen=self.MAX_SEQUENCE_LENGTH)
        predictions = self.model.predict(padded)
    
        return np.round(predictions[0].astype(float), decimals=5), labels[np.argmax(predictions)]


class BILSTM:
    def __init__(self):
        self.root_path = "models/BILSTM/2021-01-28_08-18-14.225/"
        self.model_path = self.root_path + "model.h5"
        self.tokenizer_path = self.root_path + "tokenizer.pickel"
        self.MAX_SEQUENCE_LENGTH = 256
        
        try:
            self.tokenizer = pickle.load(open(self.file_path, "rb"))
        except Exception as e:
            print(e)
            return None
        self.model = load_model(self.model_path)

    def predict(self, text):
        labels = ["COMPLETED", "DELIVERED_GUARANTEE", "IN_US", "RETURN_TO_SENDER", "TRACKING_AVAILABLE", "TRACKING_ONLINE"]
        li_txt = []

        text = clean_text(text)
        li_txt.append(text_fit_tokenizer(self.MAX_SEQUENCE_LENGTH, text))

        seq = self.tokenizer.texts_to_sequences(li_txt)
        padded = pad_sequences(seq, maxlen=self.MAX_SEQUENCE_LENGTH)
        predictions = self.model.predict(padded)
    
        return np.round(predictions[0].astype(float), decimals=5), labels[np.argmax(predictions)]


class BERT:
    def __init__(self): 
        # If there"s a GPU available...
        if torch.cuda.is_available():    
            # Tell PyTorch to use the GPU.    
            self.device = torch.device("cuda")
        # If not...
        else:
            self.device = torch.device("cpu")  

        self.root_path = "models/BERT/2021-05-04_16-56-23.587/" #  2021-02-07_10-19-42.703
        self.model_path = self.root_path + "model/"
        self.tokenizer_path = self.root_path + "tokenizer/"
        self.MAX_LEN = 256

        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)


    def get_dataloader(self, text: str):
        li_txt = []
        text = bert_clean_text(text)
        li_txt.append(text_fit_tokenizer(self.MAX_LEN, text))

        input_ids = []
        encoded_sent = self.tokenizer.encode(li_txt[0], # Sentence to encode.
                                        add_special_tokens = True, # Add "[CLS]" and "[SEP]"
                                        ) 
        input_ids.append(encoded_sent)
        input_ids = pad_sequences(sequences=input_ids, maxlen=self.MAX_LEN, dtype="long", truncating="post", padding="post")

        # Create attention masks
        attention_masks = []

        # Create a mask of 1s for each token followed by 0s for padding
        seq_mask = [float(i>0) for i in input_ids[0]]
        attention_masks.append(seq_mask)

        prediction_inputs = torch.tensor(input_ids)
        prediction_masks = torch.tensor(attention_masks)

        # Set the batch size.
        batch_size = 32

        # Create the DataLoader.
        prediction_data = TensorDataset(prediction_inputs, prediction_masks)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

        return prediction_dataloader


    def predict(self, text:str):
        # Get dataloader 
        prediction_dataloader = self.get_dataloader(text)

        # Put model in evaluation mode
        self.model.eval()

        # Tracking variables
        predictions = []

        # Predict
        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask = batch
            b_input_ids = b_input_ids.clone().detach().to(torch.int64)

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # Store predictions and true labels
            predictions.append(logits)

        list_labels = ["COMPLETED", "DELIVERED_GUARANTEE", "IN_US", "RETURN_TO_SENDER", "TRACKING_AVAILABLE", "TRACKING_ONLINE"]
        score = list_labels[np.argmax(predictions[0])]
        
        return np.round(predictions[0][0].astype(float), decimals=5), score