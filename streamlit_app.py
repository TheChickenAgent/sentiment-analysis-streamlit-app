#python -m streamlit run streamlit_app.py in the right directory (cd xxx) or in bash command

import streamlit as st
import numpy as np
import tensorflow as tf
from bin.bert import Bertenizer

st.title("Pairwise sentiment analysis")
st.header("Welcome to this wonderful app which detects which sentence is more positive. Powered by Streamlit.")
st.write('Getting everything ready for you... Please wait.')

model = tf.keras.models.load_model('models/model0') #load pre-trained model
bert = Bertenizer() #initialize BERT embeddings and load the model pre-trained

dictionary_output = {0 : 'lower text is more positive', 1: 'upper text is more positive'}

sentences = st.text_area("Please enter your two sentences/text here. Each new line will be concidered as a new sentence/text.")
pressed = st.button("Calculate sentiment")

if pressed & (sentences is not None):
    #Split each line to be a new sentence
    splitted = sentences.splitlines() #this should only be 2 sentences
    
    #Now convert sentences to BERT embedding
    train_pooled_bert = bert.pooling(splitted, 'max') #the model is trained on max pooling

    def reshapeTens(tens):
        shape = (1, 2, 768)
        tens = tf.reshape(tens, shape)
        return tens

    train_pooled_bert = reshapeTens(train_pooled_bert) #shape them in pairs

    pred = model.predict(train_pooled_bert)
    pred = np.rint(pred)

    st.write("Processed and the result is:")
    st.write(dictionary_output[pred[0][0]]) #working with trigrams, and the first two characters are spaces, so from 2:end should be in the list. +1 because Python indexes work from 0.
    st.write(" ") #ensures nice spacing