from Aspect_Analysis.script.evaluate import calculate_aspect, Model
import argparse, os, json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BASE = os.getcwd() + '/Aspect_Analysis/'

def calculatePolarity(text):
    sent_analyser = SentimentIntensityAnalyzer()
    return sent_analyser.polarity_scores(text)['compound']

def parseData(data):
    processedData = []
    for item in data:
        value = item['word']
        key = item['sentence']
        processedData = insertData(key, value, processedData)
    return processedData

def insertData(sentence, value, processedData):
    for item in processedData:
        if(item['sentence'] == sentence):
            isDataPresent = False
            for x in item['aspect']:
                if x == value:
                    isDataPresent = True
            if not isDataPresent:
                item['aspect'].append(value)
            return processedData
    polarity = calculatePolarity(sentence)
    if polarity < 0:
        sentiment = 'Negative'
    elif polarity == 0:
        sentiment = 'Neutral'
    else:
        sentiment = 'Positive'
    newData = {'sentence' : sentence, 'aspect': [value], 'polarity': polarity, 'sentiment' : sentiment}
    processedData.append(newData)
    
    return processedData

def getAspect(domain, text):
    if domain == 'restaurant':
        if text != '':
            file = open(BASE+'demo/'+domain+'.txt', 'w')
            file.write(text)
            file.close()
        else:
            return 
    elif domain == 'laptop':
        if text != '':
            file = open(BASE+'demo/'+domain+'.txt', 'w')
            file.write(text)
            file.close()
        else:
            return
    else:
        return
    print('here')
    parser = argparse.ArgumentParser()
    if domain == 'restaurant':
        parser.add_argument('--demo_fn', type=str, default='restaurant.txt')
    elif domain == 'laptop':
        parser.add_argument('--demo_fn', type=str, default='laptop.txt')
    parser.add_argument('--emb_dir', type=str, default=BASE + "data/embedding/")
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--demo_dir', type=str, default=BASE+"demo/")
    parser.add_argument('--prep_dir', type=str, default=BASE+"demo/prep/"+domain+'/')
    parser.add_argument('--model_fn', type=str, default=BASE+"model/"+domain+"_model/"+domain)
    parser.add_argument('--gen_emb', type=str, default="gen.vec")
    parser.add_argument('--embeddings', type=str, default=domain+"_emb.vec")
    parser.add_argument('--PoStag', type=bool, default=True)
    parser.add_argument('--crf', type=bool, default=False)
    parser.add_argument('--StanfordPOSTag_dir', type=str, default=BASE+"stanford-postagger-full/")


    args = parser.parse_args()

    return parseData(calculate_aspect(
        args.demo_dir,
        args.demo_fn,
        args.embeddings, 
        args.model_fn,
        args.StanfordPOSTag_dir,
        domain, 
        args.emb_dir+args.gen_emb, 
        args.emb_dir+args.embeddings, 
        args.runs, 
        300, 
        100, 
        args.prep_dir, 
        crf=args.crf, 
        tag=args.PoStag
    ))
    # return data