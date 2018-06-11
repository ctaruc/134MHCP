#!/home/anaconda/bin/python3
# coding: utf-8

# In[1]:


import pandas as pd
import os


# In[2]:


import numpy as np


# In[3]:


import time
from tqdm import tqdm


# In[4]:


# ! pip3 install opencv-python 


# In[5]:


import tensorflow as tf
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


# In[6]:


movie = pd.read_csv("MovieGenre.csv", encoding='latin-1')


# In[7]:


movie.head()


# In[8]:


# for i in range(0, len(movie.Poster)):
#     a = movie.Poster[i]
#     b = movie.Title[i]+".jpg"
#     ! wget -O "$b" "$a"


# In[9]:


# ! pip install matplotlib


# In[10]:


import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
# img = cv2.imread("Images/Jumanji (1995).jpg")
# plt.imshow(img)


# In[11]:


# img.shape


# In[12]:


x = []
for name in tqdm((os.listdir("Images/"))):
    sb = "Images/"+name
    img = cv2.imread(sb)
    x.append(img)


# In[13]:


trash = []


# In[14]:


for i in range(len(x)):
    if type(x[i]) != type(x[0]):
        trash.append(i)


# In[15]:


counter = 0
for i in trash:
    del x[i-counter]
    counter+= 1


# In[16]:


shape_trash = []


# In[17]:


for i in range(len(x)):
    if (x[i].shape) != (x[0].shape):
        shape_trash.append(i)


# In[18]:


shape_trash


# In[19]:


trash


# In[20]:


counter = 0
for i in shape_trash:
    del x[i-counter]
    counter+= 1


# In[21]:


# plt.imshow(x[60])


# In[22]:


x = np.asarray(x)


# In[23]:


a = []
for name in (os.listdir("Images/")):
    sb = os.path.splitext(name)[0]
    a.append(sb)
a
titles = pd.DataFrame(a, columns = ['Title'])


# In[24]:


merge = pd.merge(titles,movie, on = ['Title'], left_index = True, right_index = True)
merge


# In[25]:


one_hot = pd.get_dummies(movie['Genre'])
df = merge.drop('Genre', axis=1)
df = merge['Genre'].str.join(sep='').str.get_dummies(sep='|')
df['title'] = merge.Title
df


# In[26]:


df.loc[df['Game-Show'] == 1]


# In[27]:


df = df.drop(df.index[trash])
df


# In[28]:


df = df.drop(df.index[shape_trash])
df


# In[29]:


df.reset_index(drop=True)


# In[30]:


a = ['Drama', 'Comedy', 'Romance', 'Horror', 'Action', 'Crime']
b = [x for x in df.columns if x not in a]
l = df.drop(b, axis = 1)


# In[31]:


y = l.values


# In[32]:


y.shape


# In[33]:


x[0].shape


# In[34]:


for i in range(len(x)):
    if type(x[i]) != type(x[0]):
        print(i)


# In[35]:


x = np.stack(x)


# In[36]:


x.shape


# In[37]:


y.shape


# In[38]:


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# In[39]:


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(268, 182, 3)))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(268, 182, 3)))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(268, 182, 3)))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(.5))
model.add(Dense(6))
model.add(Activation('softmax'))


# In[40]:


model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


# In[41]:


x_train.shape


# In[42]:


y_train.shape


# In[43]:


# x_train = np.array(x_train)
# y_train = np.array(y_train)


# In[44]:


model.summary()


# In[ ]:


model.fit(x_train, y_train,
          batch_size=32, nb_epoch=100, verbose=1)


# In[ ]:


prediction = model.predict(x_test)

