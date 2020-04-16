import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM,GRU,Dropout,Dense
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy


## LOAD THE DATA HERE


model=Sequential()

model.add(LSTM(50,activation='relu',return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(100,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1000,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer=Adam(lr=0.001,decay=1e-5),loss=binary_crossentropy,metrics=[binary_accuracy])

history=model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=20,batch_size=32,shuffle=True)

# Save the model
