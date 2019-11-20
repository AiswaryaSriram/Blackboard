from wt_smart import model_LSTM, model_SVR
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
from keras.models import load_model

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
    df = pd.read_csv('testdata.csv')
    df = df[['essay_id', 'essay_set', 'essay', 'domain1_score']]

    essays = df['essay']
    
    #Run once to generate model and comment out for future runs
    X_test = tf_idf(essays)
    '''pickle_out = open('tf.pickle', 'wb')
    pickle.dump(X, pickle_out)
    pickle_out.close()'''
    #Till here
    
    #pickle_in = open("tf.pickle","rb")
    #X = pickle.load(pickle_in)
    print("X_test before normalizing", X_test)
    y = df['domain1_score']
    mean = np.mean(y)
    for i in range(len(y)):
        if np.isnan(y[i]):
            y[i] = mean
    print(np.isnan(y).any())
    print(X_test.shape, len(y))
    
    input_dim = X_test.shape[1]
    
    vectorizer2 = TruncatedSVD(n_components=300, n_iter=150)
    mean = np.mean(y)
    X_test = vectorizer2.fit_transform(X_test)
    #X_train, X_test, y_train, y_test = train_test_split(X_test, y, test_size=0.5, random_state=0)
    #X_train, X_test, y_train, y_test = conv_feat_training(X_train, X_test, y_train, y_test)
    X_test = normalize(X_test)
    print("X_test", X_test)
    model = load_model('my_model_lstm.h5')
    X_test1 = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    print(X_test1.shape)
    #np.savetxt('/home/aiswarya/Documents/sem6/NLP/Project_Team_L/src/xtest.csv',xtest,delimiter=',')
    #y_pred = model.predict(X_test1)
    #np.savetxt('/home/aiswarya/Documents/sem6/NLP/Project_Team_L/src/pred.csv',y_pred,delimiter=',')
    #y_pred = np.around(y_pred)
    #print(y_pred)
    #y_test= y
    #result = cohen_kappa_score(y_test, y_pred, weights='quadratic')
    #print("Kappa Score: {}".format(result))
