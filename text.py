# How to Build a Text Generator using Keras in Python
# https://www.thepythoncode.com/article/text-generation-keras-python?utm_source=newsletter&utm_medium=email&utm_campaign=newsletter

# Recurrent Neural Networks (RNNs) are very powerful sequence models for classification problems. 
# However, in this tutorial, we will use RNNs as generative models, which means they can learn
# the sequences of a problem and then generate entirely a new sequence for the problem domain.

# After reading this tutorial, you will learn how to build a LSTM model that can generate 
# text (character by character) using Keras in Python.

# In text generation, we show the model many training examples so it can learn a pattern between
# the input and output. Each input is a sequence of characters and the output is the next single character. 
# For instance, say we want to train on the sentence "python is great", the input is "python is
# grea" and output would be "t". We need to show the model as many examples as our memory can handle 
# in order to make reasonable predictions.

# Let's install the required dependencies for this tutorial:
# pip3 install tensorflow==1.13.1 keras numpy requests

import numpy as np
import os
import pickle
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint
from string import punctuation

# PREPAIRING THE DATASET 
# We are going to use a free downloadable book as the 
# dataset: Alice’s Adventures in Wonderland by Lewis Carroll. 
# (https://www.gutenberg.org/ebooks/11)

# THE REQUEST LIBRARY
# The requests library is the de facto standard for making HTTP requests in Python. 
# It abstracts the complexities of making requests behind a beautiful, simple API 
# so that you can focus on interacting with services and consuming data in your application.

# These lines of code will download it and save it in a text file:
# Make sure there is a data subfolder!!
import requests
content = requests.get("http://www.gutenberg.org/cache/epub/11/pg11.txt").text
open("data/wonderland.txt", "w", encoding="utf-8").write(content)

# CLEANING THE DATASET
# The below code reduces our vocabulary for better and faster training by removing upper 
 # case characters and punctuations as well as replacing two consecutive new line by just one.

# read the textbook
text = open("data/wonderland.txt", encoding="utf-8").read()
# remove caps and replace two new lines with one new line
text = text.lower().replace("\n\n", "\n")
# remove all punctuations
text = text.translate(str.maketrans("", "", punctuation))

# LETS PRINT SOME STATISTICS ABOUT THE DATASET
n_chars = len(text)
unique_chars = ''.join(sorted(set(text)))
print("unique_chars:", unique_chars)
n_unique_chars = len(unique_chars)
print("Number of characters:", n_chars)
print("Number of unique characters:", n_unique_chars)

print (text)




