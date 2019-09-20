import tensorflow as tf
from tensorflow.keras import layers,Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model,save_model
import os

def build_model():
    model = Sequential()
    model.add(Dense(16, input_dim=4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    model.compile(loss='mse', optimizer='adam')#, metrics=['mse','mae'])
    return model

def train_nn(model,X_train,y_train,X_test,y_test):
    model.fit(X_train, y_train, epochs=50, batch_size=5,  verbose=1, validation_split=0.2)
    eva_loss = model.evaluate(X_test,y_test)
    return model,eva_loss

def run_nn(X,y,fname):
    # pepare data
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    
    # load old model
    if os.path.isfile(fname):
        model = load_model(fname)
    else:
        model = build_model()
    # calculate evaluation loss and compare
    old_loss = model.evaluate(X_test,y_test)
    model,eva_loss = train_nn(model,X_train,y_train,X_test,y_test)
    
    # if better ,store the new model
    if eva_loss <= old_loss:
        print('Saving the new model')
        save_model(model,fname)
        
def predict(model,X):
    X = X.reshape((1,-1))
    y = model.predict(X)
    return float(y)