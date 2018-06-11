
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


import tensorflow as tf
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


# In[5]:


movie = pd.read_csv("/Users/michael/Documents/UCSB_2017-2018/PSTAT 134/134MHCP/MovieGenre.csv", encoding='latin-1')


# In[6]:


movie.head()


# In[7]:


len(movie)


# In[8]:


movie2 = movie.dropna()


# In[9]:


movie2.head()


# In[10]:


len(movie2)


# In[11]:


# for i in range(0, len(movie.Poster)):
#     a = movie.Poster[i]
#     b = movie.Title[i]+".jpg"
#     ! wget -O "$b" "$a"


# In[12]:


slash_movie_title = []
for i in range(len(movie.Title)):
    if ('/' in movie.Title[i]):
        slash_movie_title.append(movie.Title[i])
slash_movie_title


# In[13]:


movie3 = movie2[~movie2.Title.isin(slash_movie_title)]


# In[14]:


movie = movie3


# In[15]:


len(movie)


# In[16]:


# ! pip install matplotlib


# In[17]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
img = cv2.imread("/Users/michael/Documents/UCSB_2017-2018/PSTAT 134/134MHCP/Images/Jumanji (1995).jpg")
plt.imshow(img)


# In[18]:


img.shape


# In[19]:


x = []
x_titles = []
for name in tqdm((os.listdir("/Users/michael/Documents/UCSB_2017-2018/PSTAT 134/134MHCP/Images/"))):
    sb = "/Users/michael/Documents/UCSB_2017-2018/PSTAT 134/134MHCP/Images/"+name
    s = name[:-4]
    x_titles.append(s)
    img = cv2.imread(sb)
    x.append(img)


# In[22]:


shape_trash = []


# In[23]:


for i in range(len(x)):
    if (x[i].shape) != (x[0].shape):
        shape_trash.append(i)


# In[24]:


shape_trash


# In[25]:


counter = 0
for i in shape_trash:
    del x[i-counter]
    del x_titles[i-counter]
    counter+= 1


# In[28]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(x[65])


# In[35]:


titles = pd.DataFrame(x_titles, columns = ['Title'])


# In[38]:


df = pd.merge(titles, movie,how = 'inner', on='Title', copy = False)


# In[39]:


len(df)


# In[40]:


df = df.drop_duplicates(subset = 'Title')


# In[41]:


len(df.Title)


# In[42]:


titles2 = df.Title.tolist()


# In[43]:


excluded = []
excluded_titles = []
for i in range(len(y)):
    if(y[i] not in titles2):
        excluded.append(i)
        excluded_titles.append(y[i])


# In[44]:


counter = 0
for i in excluded:
    del x[i-counter]
    del y[i-counter]
    counter+= 1


# In[55]:


count = 0
for i in excluded:
    del(y[i-count])
    count+=1


# In[45]:


len(x)


# In[56]:


len(x_titles)


# In[78]:


plt.imshow(x[46])


# In[76]:


df = df.reset_index(drop = True)


# In[80]:


len(df.Title)


# In[82]:


col_list = ['Title', 'Genre']


# In[83]:


df = df[col_list]


# In[84]:


df.head()


# In[85]:


len(df.Title)


# In[86]:


len(x)


# In[87]:


# one_hot = pd.get_dummies(movie['Genre'])
# df = merge.drop('Genre', axis=1)
df2 = df['Genre'].str.join(sep='').str.get_dummies(sep='|')


# In[88]:


df2.head()


# In[89]:


df2 = df2.join(df, how = 'outer')


# In[90]:


len(df2.Action)


# In[91]:


df2 = df2.drop(['Genre'],axis = 1)


# In[92]:


cols = df2.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df2[cols]
df


# In[93]:


len(df['Title'])


# In[94]:


len(x)


# In[141]:


#file_name = "/Users/michael/Documents/UCSB_2017-2018/PSTAT 134/134MHCP/cleaned_movies.csv"


# In[142]:


#df.to_csv(file_name)


# In[113]:


a = ['Adventure', 'Documentary', 'Romance', 'Horror']
b = [x for x in df.columns if x not in a]
l = df.drop(b, axis = 1)


# In[30]:





# In[98]:


# df_test = df.drop(['Title'],axis = 1)


# In[100]:


# df_test.head()


# In[101]:


# y = df_test.values


# In[114]:


y = l.values


# In[115]:


y.shape


# In[116]:


x[0].shape


# In[117]:


x = np.stack(x)


# In[105]:


x.shape


# In[106]:


y.shape


# In[125]:


from sklearn.cross_validation import train_test_split


# In[126]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# In[127]:


input_shape = x_train[0].shape
num_class = y_train.shape[1]
print(input_shape)
print(num_class)


# In[119]:


# from Metrics import f1,recall,precision


# In[135]:


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# layer 2
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# layer 3
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# layer 4
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# layer 5
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# layer 6
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# layer 7
model.add(Dense(num_class))
model.add(Activation('relu'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:


# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(268, 182, 3)))
# model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Dropout(.25))

# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(268, 182, 3)))
# model.add(MaxPooling2D(pool_size = (2,2)))

# model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(268, 182, 3)))
# model.add(MaxPooling2D(pool_size = (2,2)))

# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(.5))
# model.add(Dense(6))
# model.add(Activation('sigmoid'))


# In[ ]:


# model.compile(loss='categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy'])


# In[129]:


x_train.shape


# In[130]:


y_train.shape


# In[ ]:


# x_train = np.array(x_train)
# y_train = np.array(y_train)


# In[ ]:


model.summary()


# In[ ]:


model.fit(x_train, y_train,
          batch_size=32, epochs=30, verbose=1)


# In[ ]:


prediction = model.predict(x_test)

