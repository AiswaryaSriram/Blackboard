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
    print("X", X)
    y = df['domain1_score']
    mean = np.mean(y)
    for i in range(len(y)):
        if np.isnan(y[i]):
            y[i] = mean
    print(np.isnan(y).any())
    print(X.shape, len(y))
    #(12978, 38237) 12978

    input_dim = X.shape[1]
    
    #vectorizer2 = TruncatedSVD(n_components=300, n_iter=150)
    #mean = np.mean(y)
    #X = vectorizer2.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_test, y_train, y_test = conv_feat_training(X_train, X_test, y_train, y_test)
    print("X_test",X_test)
    model = load_model('my_model_lstm.h5')
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    print(X_test.shape)
    #(2596, 1, 300)
    #np.savetxt('/home/aiswarya/Documents/sem6/NLP/Project_Team_L/src/xtest.csv',xtest,delimiter=',')
    '''y_pred = model.predict(X_test)
    #np.savetxt('/home/aiswarya/Documents/sem6/NLP/Project_Team_L/src/pred.csv',y_pred,delimiter=',')
    y_pred = np.around(y_pred)
    result = cohen_kappa_score(y_test, y_pred, weights='quadratic')
    print("Kappa Score: {}".format(result))'''

'''y_pred= 
[[15.442488 ]
 [ 3.9721258]
 [ 3.9052274]
 ...
 [ 0.       ]
 [17.299204 ]
 [ 3.9766002]]
AFter pickle load, X is this 
X   (0, 8578)	0.03594060024030617
  (0, 20040)	0.08902580364965984
  (0, 22503)	0.042092098205889814
  (0, 34147)	0.09611406544980296
  (0, 11078)	0.09746658912025312
  (0, 6750)	0.06528776239447541
  (0, 15818)	0.07010618944753001
  (0, 23373)	0.16447213876425498


X_test= [[[ 0.53089703 -0.08918714  0.1521853  ... -0.01526245 -0.01807895
    0.02727299]]

 [[ 0.4097248   0.27621576 -0.25034225 ...  0.00338167 -0.01305134
    0.01992441]]

 [[ 0.59510604 -0.34356726 -0.1342064  ... -0.02228788  0.0208956
    0.00430084]]

 ...

 [[ 0.4231244   0.15133451  0.18352729 ...  0.00428343  0.00219969
   -0.02283874]]

 [[ 0.3433853  -0.11013533  0.06423391 ...  0.05939354  0.00239756
    0.04735716]]

 [[ 0.49523346 -0.25002775 -0.07663963 ...  0.05087421 -0.02151789
   -0.04343667]]]

without truncated SVD
X_test = X_test   (0, 8990)	0.14823901158751962
  (0, 1489)	0.13434753528565288
  (0, 24427)	0.06257075767041172
  (0, 28865)	0.11579442019257108
  (0, 15432)	0.1377305154328552
  (0, 21911)	0.10573851153629384'''
