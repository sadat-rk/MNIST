#!/usr/bin/env python
# coding: utf-8

# In[29]:


import matplotlib.pyplot as plt 
import numpy as np
import random 
from keras.utils import to_categorical


from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Dropout
# metrics 
from keras.metrics import categorical_crossentropy
# optimization method
from keras.optimizers import SGD


# In[30]:


from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[31]:


print('X_train shape', X_train.shape, 'X_test shape', X_test.shape)


# In[32]:


# visualize some data
plt.figure(figsize = (12,5))
for i in range(8):
    ind = random.randint(0, len(X_train))
    plt.subplot(240+1+i)
    plt.imshow(X_train[ind])


# In[33]:


def preprocess_data(X_train, y_train, X_test, y_test):
    # reshape images to the the required size by Keras
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    # convert from integers to floats
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # normalize to range 0-1
    X_train = X_train/255.0
    X_test_norm = X_test/255.0
    # One-hot encoding label 
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return X_train, y_train, X_test, y_test 


# In[47]:


def train_model(model, X_train, y_train, X_test, y_test, epochs = 50, batch_size = 128):
    # Rescaling all training and testing data
    X_train, y_train, X_test, y_test = preprocess_data(X_train, y_train, X_test, y_test)
    # Fitting the model
    history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, steps_per_epoch = X_train.shape[0]//batch_size, validation_data = (X_test, y_test), validation_steps = X_test.shape[0]//batch_size, verbose = 1)
    # evaluate the model
    _, acc = model.evaluate(X_test, y_test, verbose = 1)
    print('%.3f' % (acc * 100.0))
    summary_history(history)


# In[35]:


def summary_history(history):
    plt.figure(figsize = (10,6))
    plt.plot(history.history['accuracy'], color = 'blue', label = 'train')
    plt.plot(history.history['val_accuracy'], color = 'red', label = 'val')
    plt.legend()
    plt.title('Accuracy')
    plt.show()
    
    plt.plot(history.history['loss'],color = 'blue', label = 'train')
    plt.plot(history.history['val_loss'], color = 'red', label = 'val')
    plt.title('Loss')
    plt.legend()
    plt.show()


# In[39]:


def example(model):
    # all categories
    cates = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    plt.figure(figsize = (12,7))
    for i in np.arange(8):
        ind = random.randint(0,len(X_test))
        img = X_test[ind]
        img = img.reshape(1,28,28,1)
        img = img.astype('float32')
        img = img/255.0
        v_p = (model.predict(img)> 0.5).astype("int32")
        for m in v_p:
            for x in range(len(m)):
                if m[x] == 1:
                    j = x   
        plt.subplot(240+1+i)
        plt.imshow(X_test[ind])
        plt.title(j)


# LeNet5

# ![image.png](attachment:image.png)

# In[36]:


def LeNet():
    model = Sequential()
    model.add(Conv2D(filters = 6, kernel_size = (5,5), padding = 'same', activation = 'relu', input_shape = (28,28,1)))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 16, kernel_size = (5,5), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(120, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    # compile the model with a loss function, a metric and an optimizer function
    opt = SGD(lr = 0.01)
    model.compile(loss = categorical_crossentropy, 
                optimizer = opt, 
                metrics = ['accuracy']) 
    return model


# In[48]:


LeNet_model = LeNet()
LeNet_model.summary()


# In[49]:


train_model(LeNet_model, X_train, y_train, X_test, y_test)


# In[40]:


example(LeNet_model)


# CNN0

# In[50]:


def CNN0():
    model = Sequential()
    model.add(Conv2D(filters = 24, kernel_size = (5,5), padding = 'same', activation = 'relu', input_shape = (28,28,1)))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 48, kernel_size = (5,5), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 64, kernel_size = (5,5), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(265,activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    
    opt = 'adam'
    model.compile(loss = categorical_crossentropy,
                 optimizer = opt,
                 metrics= ['accuracy'])
    return model


# In[51]:


CNN0_model = CNN0()
CNN0_model.summary()


# In[52]:


train_model(CNN0_model, X_train, y_train, X_test, y_test, epochs = 30)


# In[54]:


example(CNN0_model)


# CNN1

# In[55]:


def CNN1():
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size=(3, 3), activation='relu',input_shape= (28,28,1)))
    model.add(Conv2D(filters = 64, kernel_size =(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(250, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))
    opt=SGD(lr=0.01)
    model.compile(loss=categorical_crossentropy,
                  optimizer=opt,metrics=['accuracy'])
    return model


# In[64]:


CNN1_model = CNN1()
CNN1_model.summary()


# In[65]:


train_model(CNN1_model, X_train, y_train, X_test, y_test)


# In[62]:


example(CNN1_model)


# In[ ]:




