import numpy as np
np.random.seed(0)
import pandas as pd
from keras.layers import Dense, Dropout, GRU
from keras.models import Sequential
from keras import regularizers
from matplotlib import pyplot
from keras.utils import np_utils
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from keras.optimizers import adam,rmsprop,nadam,adadelta,adamax,SGD
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

#set parameters
dropout_Perc = 0.2
l2_Reg = 0.01
LR=0.001
initial_method='glorot_normal'

#training set, testing set
train_data = pd.read_csv('Input-traindata')
x_train = train_data.iloc[:,0:16]
y_train = np.ravel(train_data.landslide)
x_train = np.array(x_train)
x_train = np.expand_dims(x_train,axis=2)

test_data = pd.read_csv('Input-testdata')
x_test = test_data.iloc[:,0:16]
x_test = np.expand_dims(x_test,axis=2)
y_test = np.ravel(test_data.landslide)
x_test = np.array(x_test)

y_train = np_utils.to_categorical(y_train, num_classes=2)
y_test = np_utils.to_categorical(y_test, num_classes=2)
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

#build model
model = Sequential()
model.add(Dropout(dropout_Perc, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(GRU(100, return_sequences=True))
model.add(Dropout(dropout_Perc))
model.add(GRU(100, return_sequences=True))
model.add(Dropout(dropout_Perc))
model.add(GRU(100))
model.add(Dropout(dropout_Perc))
model.add(Dense(64, activation='softmax',kernel_initializer=initial_method, kernel_regularizer=regularizers.l2(l2_Reg)))
model.add(Dropout(dropout_Perc))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=adam(LR), metrics=['accuracy'])
model.summary()

trained_Model = model.fit(x_train, y_train, epochs=340, batch_size=128,validation_data=(x_test, y_test), verbose=0, shuffle=False)

scores = model.evaluate(x_test, y_test, verbose=0)
print('GRU test score:', scores[0])
print('GRU test accuracy:', scores[1])

y_pred_test = model.predict(x_test)
y_pred_test = [x[1]for x in y_pred_test ]
y_test = [y[1]for y in y_test ]

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
