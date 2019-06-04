from __future__ import print_function
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM
from keras.datasets import imdb
import numpy as np
from sru import SRU
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from keras.optimizers import adam,rmsprop,nadam,adadelta,adamax,SGD
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score

#set parameters
max_features = 10000
maxlen = 16  
batch_size = 128
depth = 1
LR=0.001

#training set, testing set
train_data = pd.read_csv('Input-traindata')
x_train = train_data.iloc[:,0:16]
y_train = np.ravel(train_data.landslide)
x_train = np.array(x_train)

test_data = pd.read_csv('Input-testdata')
x_test = test_data.iloc[:,0:16]
y_test = np.ravel(test_data.landslide)
x_train = np.array(x_train)

#build model
print('Build model...')
ip = Input(shape=(maxlen,))
embed = Embedding(max_features, 128)(ip)

prev_input = embed
hidden_states = []

if depth > 1:
    for i in range(depth - 1):
        h, h_final, c_final = SRU(128, dropout=0.2, recurrent_dropout=0.2,
                                  return_sequences=True, return_state=True,
                                  unroll=True)(prev_input)
        prev_input = h
        hidden_states.append(c_final)

outputs = SRU(128, dropout=0.2, recurrent_dropout=0.2, unroll=True)(prev_input)
outputs = Dense(1, activation='sigmoid')(outputs)

model = Model(ip, outputs)
model.summary()
model.compile(loss='binary_crossentropy', optimizer=adam(LR),metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train, batch_size=batch_size, epochs=280,verbose=0,validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

y_pred_test = model.predict(x_test)
probability = np.ravel(y_pred_test)
pred_class = []
 
for i in probability:
    if i > 0.5:
        pred_class.append(1)
    else:
        pred_class.append(0)

#evaluate test set
test_auc = metrics.roc_auc_score(y_test,probability)
recall = recall_score(y_test,pred_class)
f1_score = f1_score(y_test,pred_class)
matthews_corrcoef= matthews_corrcoef(y_test,pred_class)
print("AUC = " + str(test_auc))
print("recall = " + str(recall))
print("f1_score = " + str(f1_score))
print("matthews_corrcoef = " + str(matthews_corrcoef))
