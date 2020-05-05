# Sentiment Analysis

Sentiment analysis refers to the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information.



## Cloning

```bash
git clone https://github.com/savvyyy/Sentiment-Analysis.git
```

## Installation

```bash
1. sudo chmod +x ./setup.sh
2. Run ./setup.sh
```
## Usage
Change .env.example to .env
```
Run python main.py 
```
main.py will start pointing to 127.0.0.1:5000. Run this url in postman to see the result.

1. Endpoint '/getSentiment' is an Api that provides Sentiment Analysis.
2. Endpoint '/aspect/domain' is an Api that provides Abstract Based Sentiment Analysis. Here domain could be Twitter or Laptop or Restaurant.
3. Endpoint '/intent' is an Api that provides Intent Based Sentiment Analysis(need more work).
4. Endpoint '/graph' is an Api that provides Date wise tweets and their polarity to plot in a graph to study the variations in tweets.

## Description

### 1. Sentiment Analysis 

Sentiment analysis refers to the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information. It returns Polarity,relevance of subjectivity and Sentiment(Very Positive, Positive, Neutral, Negative, Very Negative ).

### 2. Aspect Sentiment

Aspect-based sentiment analysis is a text analysis technique that breaks down text into aspects (attributes or components of a product or service), and then allocates each one a sentiment level (positive, negative or neutral).
Here’s a breakdown of what aspect-based sentiment analysis can extract :- a) Sentiments: positive or negative opinions about a particular aspect.
b) Aspects: the thing or topic that is being talked about.

#### Process to Follow in Aspect Analysis
a. Download general embeddings from (GloVe:[http://nlp.stanford.edu/data/glove.840B.300d.zip](http://nlp.stanford.edu/data/glove.840B.300d.zip)). Rename it gen.vec and save it in folder: ```data/embedding/```

Download restaurant and laptop domain embeddings from [here](https://drive.google.com/open?id=1gkeGnS-4_RufCjHu65FCq1zFORzdYmqy), save them in folder: ```data/embedding/```

b. ```Download Stanford POS Tagger :- ``` Download Stanford Log-linear Part-Of-Speech Tagger from [The Stanford Natural Language Processing Group](https://nlp.stanford.edu/software/tagger.shtml#Download) to folder and rename it as: ```stanford-posttagger-full/```

c. To prepare dataset :- Run ```cd Aspect_Analysis``` and then Run ```python script/prepare_dataset.py --domain "domain"```. Here domain are restaurant or laptop.

d. After runing this script, you should expect to generate the following files in folder: data/prep_data

 - word_idx.json (dictionary for words appeared in the dataset, map a word to a ID number)
- gen.vec.npy, restaurant_emb.vec.npy (prepared embedding for words in word_idx)
- restaurantTrain.npz, restaurantTest.npz (prepared training/text dataset extracted from the .xml files)

e. To Train dataset :- Run ```python script/train_dataset.py --domain "domain"```.

- Training would take around 12-14 hours on CPU(training has been done on CPU(haven't tested on GPU but GPU configuration has been added)).

- Note:- You can skip this ```step e``` because pre-trained models of laptop and restaurant are already added to this repo.

f. To Evaluate the model :- If [UI](https://github.com/savvyyy/Sentiment-Analysis-UI.git) is connected then to see the result run ```python main.py``` from root directory. If UI is not connected then to see the result in the terminal Run ```python script/evaluate_sample.py --domain "domain"```.

g. For Aspect Analysis in case of Twitter data, [N-E-R(Named Entity Recognition has been followed.)](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) which is also called ```IOB tagging```. Ouput can be seen in UI based on a hashtag search.

### 3. Intent Analysis

Intent Analysis acknowledges the intentions from the text. It can be any intentions such as the intention to sell, or intention to complain or intention to purchase etc.

#### Process to Follow in Intent Analysis

This is a [BERT](https://github.com/google-research/bert) based model.

Training and testing of Intent Analysis requires GPU based system.
 
Run directly cell by cell ```intent.ipynb``` [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/savvyyy/Aspect-Intent-Sentiment-Analysis/blob/master/Intent_Analysis/intent_analysis.ipynb)

Or

a.) Open terminal and run following commands:- 

```bash
1. cd Intent_Analysis
2. wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
3. unzip uncased_L-12_H-768_A-12.zip
```

Create a folder named ```model``` in Intent_Analysis folder and place ```uncased_L-12_H-768_A-12``` in model folder.

For Training, Run

```python train_intent_model.py```

For Evaluation, Run
```python evaluate_intent.py```

## UI
Refer to ```https://github.com/savvyyy/Sentiment-Analysis-UI.git``` 