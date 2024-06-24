
# coding: utf-8

# In[1]:


import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
# from pyvi.pyvi import ViTokenizer

stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow.compat.v1 as tf
import random
import warnings
warnings.filterwarnings("ignore")


# In[2]:


import json
with open('D:/MonHocKy7/PL6/chatbot-demo-test/data/intents_test.json', encoding='utf-8') as json_data:
    intents = json.load(json_data)


# In[42]:


def ngrams(str, n):
    tokens = str.split(' ')
    arr = []
    for i in range(len(tokens)):
        new_str = ''
        if i == 0 and n>1:
            new_str = '_'
            for j in range(n):
                if j < n - 1:
                    if (i + j) <= len(tokens):
                        new_str += ' '+tokens[i+j]
                    else:
                        new_str += ' _'
        else:
            for j in range(n):
                if j < n:
                    if (i + j) < len(tokens):
                        if j == 0:
                            new_str += tokens[i+j]
                        else:
                            new_str += ' '+tokens[i+j]
                    else:
                        new_str += ' _'
        arr.append(new_str)
    return arr


# In[43]:


ngrams('a b c d e f g h', 4)


# In[3]:


words = []
classes = []
documents = []
ignore_words = ['?', 'và', 'à', 'ừ', 'ạ', 'vì', 'từng', 'một_cách']

for intent in intents['intents']:
    for pattern in intent['patterns']:

        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)


# In[4]:


# Khởi tạo danh sách training_data
training_data = []
output_empty = [0] * len(classes)

# Vòng lặp qua các documents để tạo dữ liệu huấn luyện
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

    # Tạo mảng bag
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    # Thêm mảng bag và mảng output_row vào training_data
    training_data.append([bag, output_row])

# Trộn ngẫu nhiên dữ liệu huấn luyện
random.shuffle(training_data)

# Tạo mảng NumPy cho dữ liệu huấn luyện
train_x = []
train_y = []

for item in training_data:
    train_x.append(item[0])
    train_y.append(item[1])

# Chuyển danh sách thành mảng NumPy
train_x = np.array(train_x)
train_y = np.array(train_y)



# In[5]:


print(train_x[1])
print(train_y[1])


# In[6]:


# tf.reset_default_graph()
tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')
# net = tflearn.regression(net, optimizer=tf.compat.v1.train.AdamOptimizer(), loss='sparse_categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('models/model.tflearn')


# In[8]:


import pickle
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "models/training_data", "wb" ) )

