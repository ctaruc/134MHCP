import pandas as pd
import os
import numpy as np
import time
from tqdm import tqdm
import tensorflow as tf
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


movie = pd.read_csv("MovieGenre.csv", encoding='latin1')
movie2 = movie.dropna()

slash_movie_title = []
for i in range(len(movie.Title)):
    if ('/' in movie.Title[i]):
        slash_movie_title.append(movie.Title[i])
slash_movie_title

movie3 = movie2[~movie2.Title.isin(slash_movie_title)]

movie = movie3

x = []
x_titles = []
for name in tqdm((os.listdir("Images/"))):
    sb = "Images/"+name
    s = name[:-4]
    x_titles.append(s)
    img = cv2.imread(sb)
    x.append(img)

shape_trash = []

for i in range(len(x)):
    if (x[i].shape) != (x[0].shape):
        shape_trash.append(i)

shape_trash

counter = 0
for i in shape_trash:
    del x[i-counter]
    del x_titles[i-counter]
    counter+= 1

titles = pd.DataFrame(x_titles, columns = ['Title'])

df = pd.merge(titles, movie,how = 'inner', on='Title', copy = False)

df = df.drop_duplicates(subset = 'Title')


titles2 = df.Title.tolist()

excluded = []
excluded_titles = []
for i in range(len(x_titles)):
    if(x_titles[i] not in titles2):
        excluded.append(i)
        excluded_titles.append(x_titles[i])

counter = 0
for i in excluded:
    del x[i-counter]
    del x_titles[i-counter]
    counter+= 1

count = 0
for i in excluded:
    del(x_titles[i-count])
    count+=1

df = df.reset_index(drop = True)

col_list = ['Title', 'Genre']


df = df[col_list]

df2 = df['Genre'].str.join(sep='').str.get_dummies(sep='|')

df2 = df2.join(df, how = 'outer')

df2 = df2.drop(['Genre'],axis = 1)

cols = df2.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df2[cols]

a = ['Adventure', 'Documentary', 'Romance', 'Horror']
b = [x for x in df.columns if x not in a]
l = df.drop(b, axis = 1)

y = l.values

y.shape

x[0].shape

x = np.stack(x)

from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

input_shape = x_train[0].shape
num_class = y_train.shape[1]

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
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train,
          batch_size=32, epochs=15, verbose=1)

prediction = model.predict(x_test)

model.save("model0_sigmoid_binaryCE.h5")
