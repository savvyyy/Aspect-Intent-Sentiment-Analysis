import argparse
import numpy as np
import os
import xml.etree.ElementTree as ET
import io, json
from nltk import word_tokenize
import nltk
from nltk.tag import StanfordPOSTagger

"""fastText: https://github.com/facebookresearch/fastText"""
from fasttext import load_model

def build_vocab(data_dir, out_fn, plain = []):
    """plain is a empty str file which will record all text from official dataset"""
    for fn in os.listdir(data_dir):
        if fn.endswith('.xml'):
            with open(data_dir+fn) as f:
                dom=ET.parse(f)
                root=dom.getroot()
                for sent in root.iter("sentence"):
                    text = sent.find('text').text
                    token = word_tokenize(text)
                    plain = plain + token
    vocab = sorted(set(plain))
    word_idx = {}
    for idx, word in enumerate(vocab):
         word_idx[word] = idx+1         
    with io.open(out_fn, 'w') as outfile:
        outfile.write(json.dumps(word_idx))

def gen_np_embedding(fn, word_idx_fn, out_fn, dim=300, emb=False):
    
    # whether use fastText embedding for out-of-vocabulary words. emb=True: yes; emb=False: no
    if emb:
        model = load_model(fn+".bin")
            
    with open(word_idx_fn) as f:
        word_idx=json.load(f)
    embedding=np.zeros((len(word_idx)+2, dim) )
    with open(fn) as f:
        # read the embedding .vec file
        for l in f:
            # for each line, get the word and its vector
            rec=l.rstrip().split(' ')
            if len(rec)==2: #skip the first line.
                continue 
            # if the word in word_idx, fill the embedding
            if rec[0] in word_idx:
                embedding[word_idx[rec[0]]] = np.array([float(r) for r in rec[1:] ])
    for w in word_idx:
        if embedding[word_idx[w] ].sum()==0.:
            if emb:
                embedding[word_idx[w] ] = model.get_word_vector(w)
    np.save(out_fn+".npy", embedding.astype('float32') )


def create_train_data(fn, word_idx_fn, out_dir, POSdir, domain, str_name='Train', sent_len=100, sent_num=3045):
    """
    Output:
        .npz file which includes train_X, trainX_tag, train_y. 
        Each is a size=(2000,83) np.ndarray
        
        .json file which show a list of raw sentences (tokenized) 
        .json file which show corresponding aspect tag 
        [ 
        [[Sentence1-opinion 1: target, catag_main, catag_sub, polarity, word_start_idx, word_end_idx], [Sentence1-opinion 2], ... ], 
        [Sentence2], 
        ... ]
    """
    # map part-of-speech tag to int
    pos_tag_list = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS','NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP','SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB',',','.',':','$','#',"``","''",'(',')']
    tag_to_num = {tag:i+1 for i, tag in enumerate(sorted(pos_tag_list))}

    # initialize
    corpus = []
    corpus_tag = []
    opsList = []
    train_X = np.zeros((sent_num, sent_len), np.int16)
    train_X_tag = np.zeros((sent_num, sent_len), np.int16)
    train_y = np.zeros((sent_num, sent_len), np.int16) 
    
    # read vocab file
    with open(word_idx_fn) as f:
        word_idx=json.load(f) 
    
    # read xml file
    dom=ET.parse(fn)
    root=dom.getroot()
    # iterate the review sentence
    for sx, sent in enumerate(root.iter("sentence") ) : 
        if sx%100==0:
            print('finish sentence: ', str(sx))
        text = sent.find('text').text

        # tokenize the current sentence
        token = word_tokenize(text)
        corpus.append(token)

        # Add the jar and model via their path (instead of setting environment variables):
        jar = POSdir+'stanford-postagger.jar'
        model = POSdir+'models/english-left3words-distsim.tagger'
        pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')
        pos_tag_stf = [tag_to_num[tag] for (_,tag) in pos_tagger.tag(token)]
           
        # write word index and tag in train_X and train_X_tag
        for wx, word in enumerate(token):
            train_X[sx, wx] = word_idx[word]
            train_X_tag[sx, wx] = pos_tag_stf[wx]

        # create a list for opinions in this sentence
        opList = []
        
        # iterate the opinions
        for ox, opin in enumerate(sent.iter('Opinion') ) :
            if args1.domain == 'restaurant':
                # extract attibutes of Opinion
                target, category, polarity, start, end = opin.attrib['target'], opin.attrib['category'], opin.attrib['polarity'], int(opin.attrib['from']), int(opin.attrib['to'])
                catag_main, catag_sub = category.split('#')
                # find word index (instead of str index) if start,end is not (0,0)
                if end != 0:
                    if start != 0:
                        start = len(word_tokenize(text[:start]))
                    end = len(word_tokenize(text[:end]))-1
                    # for training only identify aspect word, but not polarity
                    train_y[sx, start] = 1
                    if end > start:
                        train_y[sx, start+1:end] = 2   
                opList.append([target, catag_main, catag_sub, polarity, start, end])
            elif args1.domain == 'laptop':
                # extract attibutes of Opinion
                target, polarity, start, end = opin.attrib['target'], opin.attrib['polarity'], int(opin.attrib['from']), int(opin.attrib['to'])
                # catag_main, catag_sub = category.split('#')
                # find word index (instead of str index) if start,end is not (0,0)
                if end != 0:
                    if start != 0:
                        start = len(word_tokenize(text[:start]))
                    end = len(word_tokenize(text[:end]))-1
                    # for training only identify aspect word, but not polarity
                    train_y[sx, start] = 1
                    if end > start:
                        train_y[sx, start+1:end] = 2   
                opList.append([target, polarity, start, end])
        # get a list of opinions attributes
        opsList.append(opList)
        
    
    if str_name == 'Train':
        # save .npz file containing train_X, train_X_tag, train_y, train_y_polar
        np.savez(out_dir+domain+str_name+'.npz', train_X=train_X, train_X_tag=train_X_tag, train_y=train_y)
    else:
        np.savez(out_dir+domain+str_name+'.npz', test_X=train_X, test_X_tag=train_X_tag, test_y=train_y)
    
    # save .json files that contains more raw data    
    with io.open(out_dir+domain+str_name+'_text_raw.json', 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(corpus, ensure_ascii=False))
    
    with io.open(out_dir+domain+str_name+'_opinion_raw.json', 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(opsList, ensure_ascii=False))


parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=str, default=None, required=True)
args1 = parser.parse_args()

parser.add_argument('--data_dir', type=str, default="data/official_data/"+args1.domain+'/')
parser.add_argument('--out_dir', type=str, default="data/prep_data/"+args1.domain+'/')
parser.add_argument('--emb_dir', type=str, default="data/embedding/")
parser.add_argument('--gen_emb', type=str, default="gen.vec")
parser.add_argument('--embeddings', type=str, default=args1.domain+"_emb.vec")
parser.add_argument('--word_idx', type=str, default="word_idx.json")
parser.add_argument('--StanfordPOSTag_dir', type=str, default="stanford-postagger-full/")
parser.add_argument('--gen_dim', type=int, default=300)
parser.add_argument('--domain_dim', type=int, default=100)
args = parser.parse_args()

if args1.domain=='restaurant':
    fn_train = "data/official_data/restaurant/ABSA16_Restaurants_Train_SB1_v2.xml"
    sent_len, sent_num = 83, 2000
    fn_test = "data/official_data/restaurant/EN_REST_SB1_TEST_gold.xml"
    sent_len2, sent_num2 = 83, 676
elif args1.domain == 'laptop':
    fn_train = "data/official_data/laptop/ABSA16_Laptops_Train_SB1_v2.xml"
    sent_len, sent_num = 100, 3045
    fn_test = "data/official_data/laptop/EN_LAPT_SB1_TEST_gold.xml"
    sent_len2, sent_num2 = 100, 800
else:
    raise ValueError("Domain is not set! Please pass/set appropriate domain name as --domain 'Domain_name' ")

# generate word index file for words appeared in all dataset ends with .xml
build_vocab(args.data_dir, args.out_dir+args.word_idx)

# build word embedding
gen_np_embedding(args.emb_dir+args.gen_emb, args.out_dir+args.word_idx, args.out_dir+args.gen_emb, args.gen_dim)
gen_np_embedding(args.emb_dir+args.embeddings, args.out_dir+args.word_idx, args.out_dir+args.embeddings, args.domain_dim, True)

# gen_np_embedding(args.emb_dir+args.laptop_emb, args.out_dir+args.word_idx, args.out_dir+args.laptop_emb, args.domain_dim, True)

# create data for train and test
create_train_data(
    fn_train, 
    args.out_dir+args.word_idx, 
    args.out_dir, 
    args.StanfordPOSTag_dir, 
    args1.domain,
    'Train', 
    sent_len, 
    sent_num
)
create_train_data(
    fn_test, 
    args.out_dir+args.word_idx, 
    args.out_dir, 
    args.StanfordPOSTag_dir, 
    args1.domain, 
    'Test', 
    sent_len2, 
    sent_num2
)