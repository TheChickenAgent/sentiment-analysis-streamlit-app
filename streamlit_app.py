# python -m streamlit run streamlit_app.py in the right directory (cd blahblah)

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from bin.bert import Bertenizer

st.title("Pairwise sentiment analysis")
st.header("Welcome to this wonderful app which detects whether the first sentence is more positive than the right one. Powered by Streamlit.")

model = tf.keras.models.load_model('models/model0') #Load pre-trained model
dictionary_output = {0 : 'right is more positive', 1: 'left is more positive'}

sentences = st.text_area("Please enter your two sentences/text here. Each new line will be concidered as a new sentence/text.")
pressed = st.button("Calculate sentiment")

if pressed & (sentences is not None):
    #Split each line to be a new sentence
    splitted = sentences.splitlines() #this should only be 2 sentences
    
    #Now convert sentences to BERT embedding
    bert = Bertenizer()
    train_pooled_bert = bert.pooling(splitted, 'max') #the model is going to be also max

    def reshapeTens(tens):
        shape = (tens.shape[0], 2, 768)
        tens = tf.reshape(tens, shape)
        return tens

    train_pooled_bert = reshapeTens(train_pooled_bert)

    pred = model.predict(train_pooled_bert)
    pred = np.rint(pred)

    st.write("Processed and the result is:")
    st.write(dictionary_output[pred[0]]) # working with trigrams, and the first two characters are spaces, so from 2:end should be in the list. +1 because Python indexes work from 0.
    st.write(" ") # Ensures nice spacing