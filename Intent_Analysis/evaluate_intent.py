import bert
from bert import BertModelLayer
from tensorflow.keras.models import load_model
from bert.tokenization.bert_tokenization import FullTokenizer
import numpy as np
import pandas as pd
import os
# from model import create_model

bert_model_name="uncased_L-12_H-768_A-12"
bert_ckpt_dir = os.path.join("model/", bert_model_name)

class Evaluate_Intent:
    def calculateIntent(self, text, tokenizer: FullTokenizer):
        self.tokenizer = tokenizer
        max_seq_len = 38
        train = pd.read_csv('./intent_data/train.csv')
        valid = pd.read_csv('./intent_data/valid.csv')

        train = train.append(valid).reset_index(drop=True)
        classes = train.intent.unique().tolist()

        model = load_model('./intent_model.h5', custom_objects={"BertModelLayer": bert.BertModelLayer})

        pred_tokens = map(tokenizer.tokenize, text)
        pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
        pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

        pred_token_ids = map(lambda tids: tids +[0]*(max_seq_len-len(tids)),pred_token_ids)
        pred_token_ids = np.array(list(pred_token_ids))

        predictions = model.predict(pred_token_ids).argmax(axis=-1)

        for text, label in zip(text, predictions):
            print("text:", text)
            print("intent:", classes[label])

tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
cal = Evaluate_Intent()
cal.calculateIntent('I waited for 20 mins on the call to connect with customer support.', tokenizer)