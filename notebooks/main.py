import streamlit as st
import utils
import pandas as pd


siteHeader = st.container()
dataExploration = st.container()
newFeatures = st.container()
modelTraining = st.container()

with siteHeader:
    st.title('Welcome to the Awesome project!')
    st.text('In this project I look into ...  And I try ... I worked with the dataset from ...')

with dataExploration:
    st.header('Dataset: Iris flower dataset')
    st.text('I found this dataset at...  I decided to work with it because ...')
    df_corpus_top5 = pd.read_csv('../data/df_corpus_top5.csv', sep=',')
    st.write(df_corpus_top5.head())

with newFeatures:
    st.header('New features I came up with')
    st.text('Let\'s take a look into the  features I generated.')
    #uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)

    uploaded_file = st.file_uploader("Choose an image...", type="csv")
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file) 
        st.write(dataframe.head())
    
    
    

with modelTraining:
    st.header('Model training')
    st.text('In this section you can select  the hyperparameters!')
