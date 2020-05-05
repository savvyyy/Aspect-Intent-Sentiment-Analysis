import os.path
from os import path
import pandas as pd
from utils.functions import predictions, get_final_output, load_dataset, getTwitterData

def intentPrediction(hashTagSubject):
    print('hashTagSubject',hashTagSubject)
    filename = './data.csv'
    intent, unique_intent, sentences = load_dataset(filename)
    public_tweets_path = os.getcwd() + '/' + hashTagSubject + '.csv'

    if path.exists(public_tweets_path):
        print('hai')
        public_tweets =  pd.read_csv(os.path.realpath(public_tweets_path))
        prediction = []
        final_data = []
        for text in public_tweets['tweet']:
            predict = predictions(text)
            prediction.append(predict)
        for pred in prediction:
            final_data.append(get_final_output(pred, unique_intent))
        intentData = []
        for i in range(len(public_tweets['tweet'])):
            intentData.append({
                'tweet' : public_tweets['tweet'][i],
                'intent' : final_data[i]
            })
        return intentData

    else:
        print('nai hai')
        public_tweets_file = getTwitterData(hashTagSubject)
        public_tweets_path = public_tweets_file + '/' + hashTagSubject + '.csv'
        public_tweets =  pd.read_csv(os.path.realpath(public_tweets_path))

        prediction = []
        final_data = []
        for text in public_tweets['tweet']:
            predict = predictions(text)
            prediction.append(predict)
        for pred in prediction:
            final_data.append(get_final_output(pred, unique_intent))
        intentData = []
        for i in range(len(public_tweets['tweet'])):
            intentData.append({
                'tweet' : public_tweets['tweet'][i],
                'intent' : final_data[i]
            })
        return intentData


    # public_tweets = getTwitterData(hashTagSubject)
    # tweetText = []
    # prediction = []
    # final_data = []
    # for tweet in public_tweets:
    #     tweetText.append(tweet.text)
    # for text in tweetText:
    #     predict = predictions(text)
    #     prediction.append(predict)
    # for pred in prediction:
    #      final_data.append(get_final_output(pred, unique_intent))
    # # print('final_data', final_data)
    # intentData = []
    # for i in range(len(tweetText)):
    #     intentData.append({
    #         'tweet' : tweetText[i],
    #         'intent' : final_data[i]
    #     })
    # return intentData