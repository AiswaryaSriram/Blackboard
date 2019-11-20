from project import model_LSTM, model_SVR
import pandas as pd
import numpy as np
import time
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm 
from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential, load_model, model_from_config
from sklearn.metrics import cohen_kappa_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

def conv_feat_training(X_train, X_test, y_train, y_test):
    #training set
    X_train = normalize(X_train)
    y_train = np.asarray(y_train)

    #testing set
    y_test = np.asarray(y_test)
    X_test = normalize(X_test)
    return X_train, X_test, y_train, y_test


def tf_idf(essays):
	print("inside tf_idf")

	vectorizer = TfidfVectorizer()
	X = vectorizer.fit_transform(essays)
	return X


if __name__ == '__main__': 
    df = pd.read_csv('dataset.csv')
    df = df[['essay_id', 'essay_set', 'essay', 'domain1_score']]

    essays = df['essay']
    
    #Run once to generate model and comment out for future runs
    '''X = tf_idf(essays)
    pickle_out = open('tf.pickle', 'wb')
    pickle.dump(X, pickle_out)
    pickle_out.close()'''
    #Till here
    
    pickle_in = open("tf.pickle","rb")
    X = pickle.load(pickle_in)

    y = df['domain1_score']
    mean = np.mean(y)
    for i in range(len(y)):
        if np.isnan(y[i]):
            y[i] = mean
    print(np.isnan(y).any())
    print(X.shape, len(y))
    
    input_dim = X.shape[1]
    
    vectorizer2 = TruncatedSVD(n_components=300, n_iter=150)
    mean = np.mean(y)
    X = vectorizer2.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_test, y_train, y_test = conv_feat_training(X_train, X_test, y_train, y_test)

    #Comment out when you want to run LSTM
    #SVR
    start = time.time()
    y_pred, score = model_SVR(X_train, X_test, y_train, y_test)
    end = time.time()
    time_svr = end-start
    ######

    #Comment out when you want to run SVR
    #LSTM
    '''start = time.time()
    y_pred = model_LSTM(X_train, X_test, y_train, y_test, 300)
    end = time.time()
    time_lstm = end-start'''
    #######
    
    #Cohen's Kappa score
    y_pred = np.around(y_pred)
    result = cohen_kappa_score(y_test, y_pred, weights='quadratic')
    print("Kappa Score: {}".format(result))
    print("time taken for svr: ",time_svr, "secs")
    
    #print("Time taken for lstm: ", time_lstm, "secs")
