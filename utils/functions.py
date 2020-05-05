import re
import en_core_web_sm
import tweepy
import os, csv, string
from textblob import TextBlob
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from dotenv import load_dotenv
from twython import Twython
import emoji
load_dotenv()


def give_emoji_free_text(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])

    return clean_text

def getTwitterData(hashTagSubject):
    TWITTER_CONSUMER_KEY = os.getenv('CONSUMER_KEY')
    TWITTER_CONSUMER_SECRET = os.getenv('CONSUMER_SECRET')
    TWITTER_ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')
    TWITTER_ACCESS_TOKEN_SECRET = os.getenv('ACCESS_TOKEN_SECRET')

    auth = Twython(app_key=TWITTER_CONSUMER_KEY, 
                app_secret=TWITTER_CONSUMER_SECRET, 
                oauth_token=TWITTER_ACCESS_TOKEN, 
                oauth_token_secret=TWITTER_ACCESS_TOKEN_SECRET)

    search = auth.search(q='#'+hashTagSubject, tweet_mode='extended', lang='en', count=100)

    tweets = search['statuses']
    cleanedTweetText = []
    tweetText = []
    createdTweet = []
    tweetUserName = []
    tweet_without_url = []

    for i in range(0, len(tweets)):
        if 'retweeted_status' in tweets[i]:
            tweetText.append(tweets[i]['retweeted_status']['full_text'])
            removed_url_tweets = re.sub(r"http\S+", "", tweets[i]['retweeted_status']['full_text'])
            tweet_without_url.append(removed_url_tweets)
            cleaned_text = removed_url_tweets.translate(str.maketrans('', '', string.punctuation))
            cleaned_text = give_emoji_free_text(cleaned_text)
            cleanedTweetText.append(cleaned_text)
            createdTweet.append(tweets[i]['created_at'])
            tweetUserName.append(tweets[i]['user']['name'])
        else:
            tweetText.append(tweets[i]['full_text'])
            removed_url_tweets = re.sub(r"http\S+", "", tweets[i]['full_text'])
            tweet_without_url.append(removed_url_tweets)
            cleaned_text = removed_url_tweets.translate(str.maketrans('', '', string.punctuation))
            cleaned_text = give_emoji_free_text(cleaned_text)
            cleanedTweetText.append(cleaned_text)
            createdTweet.append(tweets[i]['created_at'])
            tweetUserName.append(tweets[i]['user']['name'])

    filename = hashTagSubject + '.csv'
    with open(filename, 'w',newline="") as file_writer:
        fields=["id", "tweet", "tweet_without_url", "cleaned_tweet", "userName", "created_at"]
        writer=csv.DictWriter(file_writer,fieldnames=fields)
        writer.writeheader()
        for i in range(0, len(tweetText)):
            writer.writerow({"id": i, "tweet": tweetText[i], "tweet_without_url": tweet_without_url[i], "cleaned_tweet": cleanedTweetText[i], "userName": tweetUserName[i], "created_at": createdTweet[i]})

    return os.path.dirname(os.path.abspath(filename))

# getTwitterData('india')

# def getTwitterData(hashTagSubject):
#     access_token = os.getenv('ACCESS_TOKEN')
#     access_token_secret = os.getenv('ACCESS_TOKEN_SECRET')
#     consumer_key = os.getenv('CONSUMER_KEY')
#     consumer_secret = os.getenv('CONSUMER_SECRET')
#     auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
#     auth.set_access_token(access_token, access_token_secret)
#     api = tweepy.API(auth)
#     public_tweets = api.search(hashTagSubject, count=1)

#     # public_tweets = tweepy.Cursor(api.search, q = '#'+hashTagSubject, count = 5)

#     # print('tweets', public_tweets)

#     tweetText = []
#     createdTweet = []
#     for tweet in public_tweets:
#         if tweet.lang == "en":
#             tweetText.append(tweet.text)
#             createdTweet.append(tweet.created_at.strftime('%Y-%m-%d %H:%M:%S:%f'))
    
#     filename = hashTagSubject + '.csv'
#     with open(filename, 'w',newline="") as file_writer:
#         fields=["id","tweet", "created_at"]
#         writer=csv.DictWriter(file_writer,fieldnames=fields)
#         writer.writeheader()
#         for i in range(0, len(tweetText)):
#             writer.writerow({"id": i,"tweet": tweetText[i], "created_at": createdTweet[i]})

#     return os.path.dirname(os.path.abspath(filename))


# using vader SentimentAnalyzer
def calculatePolarity(text):
    sent_analyser = SentimentIntensityAnalyzer()
    return sent_analyser.polarity_scores(text)['compound']


# using text blob sentiment analyzer
def getPolarity(text):
    textAnalysis = TextBlob(text)
    polarity = textAnalysis.sentiment.polarity
    # subjectivity = textAnalysis.sentiment.subjectivity
    return polarity

# def remove_urls(text):
#     url_pattern = re.compile(r'https?://\S+|www\.\S+|@[^\s]+')
#     return url_pattern.sub(r'', text)

def sentimentData(data):
    if data['Polarity'] >= 0.05:
        val = "Positive"
    elif data['Polarity'] <= -0.05:
        val = "Negative"
    else:
        val = "Neutral"
    return val

def getAspectTwitter(text):
    nlp = en_core_web_sm.load()
    aspects = []
    doc = nlp(text)
    aspects.extend([token.text for token in doc if token.pos_ == "NOUN"])
    for ent in doc.ents:
        if ent.label_ != 'DATE' and 'TIME' and 'ORDINAL' and 'CARDINAL' and 'PERCENT' and 'PRODUCT' and 'FAC':
            aspects.extend([ent.text])
    return aspects

def load_dataset(filename):
    df = pd.read_csv(filename, encoding = "latin1", names = ["Sentence", "Intent"])
    intent = df["Intent"]
    unique_intent = list(set(intent))
    sentences = list(df["Sentence"])
    return (intent, unique_intent, sentences)

def cleaning(sentences):
    words = []
    for s in sentences:
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
        w = word_tokenize(clean)
        #stemming
        words.append([i.lower() for i in w]) 
    return words

def create_tokenizer(words, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
    token = Tokenizer(filters = filters)
    token.fit_on_texts(words)
    return token

def max_length(words):
    return(len(max(words, key = len)))

def encoding_doc(token, words):
    return(token.texts_to_sequences(words))

def padding_doc(encoded_doc, max_length):
    return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))

def one_hot(encode):
    o = OneHotEncoder(sparse = False)
    return(o.fit_transform(encode))

def create_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length = max_length, trainable = False))
    model.add(Bidirectional(LSTM(128)))
    #model.add(LSTM(128))
    model.add(Dense(32, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(21, activation = "softmax"))
    return model

def predictions(text):
    filename = './data.csv'
    intent, unique_intent, sentences = load_dataset(filename)
    cleaned_words = cleaning(sentences)
    max_len = max_length(cleaned_words)
    word_tokenizer = create_tokenizer(cleaned_words)
    model = load_model("./model.h5")
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
    test_word = word_tokenize(clean)
    test_word = [w.lower() for w in test_word]
    test_ls = word_tokenizer.texts_to_sequences(test_word)
    # print(test_word)
    #Check for unknown words
    if [] in test_ls:
        test_ls = list(filter(None, test_ls))
    test_ls = np.array(test_ls).reshape(1, len(test_ls))
    x = padding_doc(test_ls, max_len)
    pred = model.predict_proba(x)
    return pred

def get_final_output(pred, classes):
    predictions = pred[0]
    classes = np.array(classes)
    ids = np.argsort(-predictions)
    classes = classes[ids]
    predictions = -np.sort(-predictions)
    df = []
    for i in range(pred.shape[1]):
        data = (classes[i], predictions[i])
        df.extend(list(data))
        return str(df)