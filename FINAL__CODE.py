#IMDB DataSet
from tensorflow.python.keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 10000)

class_names = ['Negative', 'Positive']

word_index = imdb.get_word_index()

#Decoding exmples from dataset to be viwed
reverse_word_index = dict((value, key) for key, value in word_index.items())
def decode(review):
    text = ''
    for i in review:
        text += reverse_word_index[i]
        text += ' '
    return text
decode(x_train[0])




#Pre-preocessing
#Showing lengths before padding
def show_lengths():
    print('Length of 1st training example: ', len(x_train[0]))
    print('Length of 2nd training example: ',  len(x_train[1]))
    print('Length of 1st test example: ', len(x_test[0]))
    print('Length of 2nd test example: ',  len(x_test[1]))
    
show_lengths()
word_index['the']

#padding
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(x_train, value = word_index['the'], padding = 'post', maxlen = 256)
x_test = pad_sequences(x_test, value = word_index['the'], padding = 'post', maxlen = 256)
#Showing lengths after padding
show_lengths()


#Creating and training model with 20 epchos and spilited to 75% for training and 25% for testing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding, Dense, GlobalAveragePooling1D

model = Sequential([
    Embedding(10000, 16),
    GlobalAveragePooling1D(),
    Dense(16, activation = 'relu'),
    Dense(1, activation = 'sigmoid')
])

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['acc']
)

model.summary()

from tensorflow.python.keras.callbacks import LambdaCallback

simple_logging = LambdaCallback(on_epoch_end = lambda e, l: print(e, end='.'))

E = 20

h = model.fit(
    x_train, y_train,
    validation_split = 0.25,
    epochs = E,
    callbacks = [simple_logging],
    verbose = False
)

#Results graph ploting
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(range(E), h.history['acc'], label = 'Training')
plt.plot(range(E), h.history['val_acc'], label = 'Validation')
plt.legend()
plt.show()

#Accuracy calculations
loss, acc = model.evaluate(x_test, y_test)
print('Test set accuracy: ', acc * 100)

import numpy as np
#Testing Predications
prediction = model.predict(np.expand_dims(x_test[50], axis = 0))
class_names = ['Negative', 'Positive']
print(class_names[round(float(prediction))])
print(float(prediction))

#Gui implementaion
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tkinter import *
import tkinter as tk
window = tk.Tk()
window.title("NLP Final Project")
window.configure(background="grey")
window.geometry('400x400') 
tk.Label(window, text="Please Enter Your Review",font=("Times New Roman", 14),width=30).grid(row=1,column=2)
e1 = Text(window,font=("Times New Roman", 14),width=40,height=10)
e1.grid(row=2, column=2)
def analyze():
    #User input pre-procssing
    e2= e1.get("1.0",'end-1c')
    review = re.sub('[^a-zA-Z]', ' ', e2)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    print(review)
    LL=[]
    for i in range(len(review)):
      if review[i] in word_index:
        LL.append(word_index[review[i]])
    print(LL)
    pp = model.predict(np.expand_dims(LL, axis = 0))
    results = class_names[round(float(pp))]
    print(float(pp)) 
    if results == "Negative":
        tk.Label(window, text=results ,width = 30, bg = "red" ).grid(row=5,column=2)
    else:
        tk.Label(window, text=results ,width = 30, bg = "green" ).grid(row=5,column=2)
btn = Button(window, text="Analyze",font=("Times New Roman", 14),width=30,command = analyze)
btn.grid(column=2, row=3)
window.mainloop()

