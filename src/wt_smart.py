import pandas as pd 
import pickle
import re
import numpy as np
import math
import time
import nltk
from nltk.corpus import stopwords 
from gensim.models import Word2Vec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm 
from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential, load_model, model_from_config
from sklearn.metrics import cohen_kappa_score
from sklearn.decomposition import TruncatedSVD

def tokenize(text):
    word_list = []
    sentences = nltk.sent_tokenize(text)
    stop_words = stopwords.words('english')

    for s in sentences:
        s = re.sub("[^a-zA-Z]", " ", s)
        word_list.extend(nltk.word_tokenize(s))

    #word_list is the list of all words in one essay
    word_list = [w for w in word_list if w not in stop_words] 
    return word_list
 
#Generates the word2Vec model
def w2v_model(word_vector_list):
    #features
    num_features = 300 
    min_word_count = 40
    num_workers = 4
    context = 10
    downsampling = 1e-3

    print("Training Word2Vec Model...")
    model = Word2Vec(word_vector_list, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)

    return model

def generate_vector_for_essay(essay, model, n_dim):
    word_list = tokenize(essay)
    featureVec = np.zeros((n_dim,), dtype="float32") 

    for word in word_list:
        try:
            featureVec = np.add(featureVec, model.wv[word])
        except KeyError:
            continue
    featureVec = np.divide(featureVec, len(word_list)) #get avg vector

    return featureVec 


def conv_feat_training(X_train, X_test, y_train, y_test):
    #training set
    X_train = np.asarray(X_train)
    X_train = np.float64(X_train)

    y_train = np.asarray(y_train)


    #testing set
    X_test = np.asarray(X_test)
    X_test = np.float64(X_test)

    y_test = np.asarray(y_test)

    return X_train, X_test, y_train, y_test

#SVR model
def model_SVR(X_train, X_test, y_train, y_test):
    print("SVR")
    start= time.time()
    clf = svm.SVR(gamma='scale')
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    end= time.time()
    print ("Time taken for SVR:", end-start)
    return predictions, score

    
#LSTM model
def model_LSTM(X_train, X_test, y_train, y_test, input_dim):
    print("LSTM")

    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential()
    model.add(LSTM(input_dim, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, input_dim], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    #model.add(Flatten())
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()
    model.fit(X_train, y_train, batch_size=64, epochs=50)
    model.save('my_model.h5')
    y_pred = model.predict(X_test)
    
    print("Saving LSTM model")
    return y_pred



if __name__ == '__main__': 
    df = pd.read_csv('dataset.csv')
    df = df[['essay_id', 'essay_set', 'essay', 'domain1_score']]

    essays = df['essay']
    #generation of the word2vec model
    #run the below once (by uncommenting) and then comment out (for future executions)
    '''
    word_vector_list = [] #has the list of all word tokenized vectors 

    for essay in essays:
        word_vector_list.append(tokenize(essay))    


    pickle_out = open('wvl.pickle', 'wb')
    pickle.dump(word_vector_list, pickle_out)
    pickle_out.close()
    '''

    pickle_in = open("wvl.pickle","rb")
    word_vector_list = pickle.load(pickle_in)

    '''model = w2v_model(word_vector_list)

    pickle_out = open('word2vec_model.pickle', 'wb')
    pickle.dump(model, pickle_out)
    pickle_out.close()'''

    pickle_in = open("word2vec_model.pickle","rb")
    model = pickle.load(pickle_in)

    #training
    print("training...")
    X = []
    n_dim = 300

    for essay in essays:
        temp = generate_vector_for_essay(essay, model, n_dim)
        X.append(temp)

    
    y = df['domain1_score'] #labels for training
    print(len(X), len(y))
    mean = np.mean(y)
    #removing Nan values
    for i in range(len(y)):
        if np.isnan(y[i]):
            y[i] = mean

    print(np.isnan(y).any())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_test, y_train, y_test = conv_feat_training(X_train, X_test, y_train, y_test)
    
    #Comment out when you want to run LSTM
    #SVR
    '''start = time.time()
    preds_SVR, score_SVR = model_SVR(X_train, X_test, y_train, y_test)
    print(preds_SVR, score_SVR)
    y_pred= preds_SVR
    end = time.time()
    time_svr = end-start'''
    #SVR is till here

    #Comment out when you want to run SVR
    #LSTM
    start = time.time()
    y_pred = model_LSTM(X_train, X_test, y_train, y_test, 300)
    end= time.time()
    time_lstm = end-start
    #LSTM is till here
    
    y_pred = np.around(y_pred)
    result = cohen_kappa_score(y_test, y_pred, weights='quadratic')
    print("Kappa Score: {}".format(result))
    #print("time taken for svr: ",time_svr, "secs")
    print("Time taken for lstm: ", time_lstm, "secs")
