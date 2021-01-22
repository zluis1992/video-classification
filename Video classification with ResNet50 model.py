#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm


# In[2]:


#Creacion del DF de entrenamiento 1
# open the .txt file which have names of training videos
'''f = open("./DS/trainlist01.txt", "r")
temp = f.read()
videos = temp.split('\n')

# creating a dataframe having video names
train = pd.DataFrame()
train['video_name'] = videos
train = train[:-1]
train.head()'''


# In[3]:


#Creacion de DF de pruebas 1
# open the .txt file which have names of test videos
'''f = open("./DS/testlist01.txt", "r")
temp = f.read()
videos = temp.split('\n')

# creating a dataframe having video names
test = pd.DataFrame()
test['video_name'] = videos
test = test[:-1]
test.head()'''


# In[4]:


#Creamos la etiqueta de cada video den entrenamiento/prueba para saber a que accion se refieren mas adelante
#todo esto lo tomamos del DF creado anteriormente...
# creating tags for training videos
'''train_video_tag = []
for i in range(train.shape[0]):
    train_video_tag.append(train['video_name'][i].split('/')[0])
    
train['tag'] = train_video_tag

# creating tags for test videos
test_video_tag = []
for i in range(test.shape[0]):
    test_video_tag.append(test['video_name'][i].split('/')[0])
    
test['tag'] = test_video_tag'''


# In[5]:


#Almacenamos los frame de cada video del DF de entrenamiento
# storing the frames from training videos
'''for i in tqdm(range(train.shape[0])):
    count = 0
    folderName = train['tag'][i]
    videoFile = train['video_name'][i]
    cap = cv2.VideoCapture('./DS/'+folderName+'/'+videoFile.split(' ')[0].split('/')[1])   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    x=1
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            # storing the frames in a new folder named train_1
            filename ='./DS/TRAINING_FRAMES/train_1/' + videoFile.split(' ')[0].split('/')[0]+ '__' +videoFile.split(' ')[0].split('/')[1] +"_frame%d.jpg" % count;count+=1
            cv2.imwrite(filename, frame)
    cap.release()'''


# In[6]:


# getting the names of all the images
'''images = glob("./DS/TRAINING_FRAMES/train_1/*.jpg")
print(images[0])
train_image = []
train_class = []
for i in tqdm(range(len(images))):
    #print(images[i].split('__')[1]) #linea para validar la estructura de la ruta de los frames
    # creating the class of image
    train_class.append(images[i].split('__')[0].split('\\')[1])
    # creating the image name
    train_image.append(images[i].split('__')[1])
    
# storing the images and their class in a dataframe
train_data = pd.DataFrame()
train_data['image'] = train_image
train_data['class'] = train_class

# converting the dataframe into csv file 
train_data.to_csv('./DS/train_new.csv',header=True, index=False)'''


# In[7]:


#Reading all the video frames
import keras
from keras.models import Sequential
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# In[77]:


train = pd.read_csv('./DS/train_new.csv')
train.head()


# In[78]:


# leeremos los frames que extrajimos anteriormente y luego los almacenaremos como una matriz NumPy
# creating an empty list
train_image = []
# for loop to read and store frames
for i in tqdm(range(train.shape[0])):
#for i in  tqdm(range(100)):
    # loading the image and keeping the target size as (224,224) por ResNet
    img = image.load_img('./DS/TRAINING_FRAMES/train_1/'+train['class'][i] + '__'+train['image'][i], target_size=(224,224,3))
    # converting it to array
    img = image.img_to_array(img)
    # normalizing the pixel value
    img = img/255
    #normalizamos las dimensiones para el modelo ResNet50
    #img = np.expand_dims(img, axis=0)
    # appending the image to the train_image list
    train_image.append(img)
    
fileName = 'train_fileResNet.npy'
# converting the list to numpy array
X = np.array(train_image)
np.save(fileName, X)


del img
del X


# In[79]:


#cargamos el array
X = np.load("train_fileResNet.npy")


# In[80]:


#mostramos las dimensiones
X.shape


# In[81]:


#Creating a validation set
# separating the target
y = train['class']
#y = train['class'][:100]


# creating the training and validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify = y)


# In[82]:


# creating dummies of target variable for train and validation set
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


# In[83]:


#liberamos RAM
del X


# In[84]:


# creating the base model of pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[85]:


X_train.shape


# In[86]:


# extracting features for training frames
X_train = base_model.predict(X_train)
X_train.shape


# In[87]:


# extracting features for validation frames
X_test = base_model.predict(X_test)
X_test.shape


# In[64]:


# reshaping the training as well as validation frames in single dimension
#X_train = X_train.reshape(80, 7*7*2048)
#X_test = X_test.reshape(20, 7*7*2048)


# In[88]:


# reshaping the training as well as validation frames in single dimension
X_train = X_train.reshape(59075, 7*7*2048)
X_test = X_test.reshape(14769, 7*7*2048)


# In[89]:


# normalizing the pixel values
max = X_train.max()
X_train = X_train/max
X_test = X_test/max


# In[90]:


# shape of images
X_train.shape


# In[91]:


#defining the model architecture
model = Sequential()
model.add(Dense(2048, activation='relu', input_shape=(100352,)))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu', input_shape=(25088,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(101, activation='softmax'))
#model.add(Dense(1, activation='softmax'))


# In[92]:


# defining a function to save the weights of best model
from keras.callbacks import ModelCheckpoint
mcp_save = ModelCheckpoint('weightResNet50.hdf5', save_best_only=True, monitor='val_loss', mode='min')


# In[93]:


# compiling the model
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
#model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


# In[ ]:


# training the model
model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[mcp_save], batch_size=128)


# In[ ]:




