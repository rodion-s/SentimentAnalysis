import torch
import transformers
import shap
from streamlit_shap import st_shap
import streamlit as st
import streamlit.components.v1 as components

from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification, 
                          TextClassificationPipeline)

from transformers import BertTokenizerFast

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import pymorphy2
from nltk import word_tokenize, TweetTokenizer
import psutil
import os


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def clean_text_lemmatize(text, tokenizer, morph):
    text = text.lower()
    text = re.sub(r'#', '# ', text)
    text = tokenizer.tokenize(text)
    
    text = [morph.parse(word)[0].normal_form for word in text]
    return text


@st.cache(allow_output_mutation=True)
def init_logreg():
    print('init_logreg...')
    train_df = pd.read_csv('./app/rusentiment_train.csv')

    morph = pymorphy2.MorphAnalyzer()
    tokenizer = TweetTokenizer()
    
    train_df['lemmatized'] = train_df['lemmatized'].apply(lambda x: literal_eval(x))
    x_train = train_df['lemmatized'].values
    y_train = train_df['label_num'].values

    cv = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    sparse_train = cv.fit_transform(x_train)
    tfidf = TfidfTransformer()
    x_train = tfidf.fit_transform(sparse_train)

    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    return x_train, cv, tfidf, clf, morph, tokenizer

def get_vector(text, cv, tfidf, tokenizer, morph):
    text = clean_text_lemmatize(text, tokenizer, morph)
    sparse_t = cv.transform([text])
    x_t = tfidf.transform(sparse_t)
    return x_t


@st.cache(allow_output_mutation=True)
def init_bert():
    print('init_bert...')
    tokenizer = AutoTokenizer.from_pretrained("blanchefort/rubert-base-cased-sentiment-rusentiment")
    model = AutoModelForSequenceClassification.from_pretrained("blanchefort/rubert-base-cased-sentiment-rusentiment")
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
    
    bert_explainer = shap.Explainer(pipe)
    return tokenizer, model, pipe, bert_explainer


def main():
    # Init BERT
    tokenizer, model, pipe, bert_explainer = init_bert()
    # Init Logistic Regression
    x_train, cv, tfidf, clf, morph, tokenizer_logreg = init_logreg()

    st.title('Sentiment analysis')

    form = st.form(key='my_form')
    input_text = form.text_input('Введите текст', value='Сегодня хороший день')

    method = st.selectbox(label='Выберите классификатор',
                          options=['TFIDF + Logistic Regression', 'Fine-tuned BERT'])
    submit_button = form.form_submit_button(label='Определить тональность')


    mapping = {'NEUTRAL': 0, 'POSITIVE': 1, 'NEGATIVE': 2}
    if submit_button:
        if method == 'TFIDF + Logistic Regression':
            text = input_text
            vector = get_vector(text=text, cv=cv, tfidf=tfidf, tokenizer=tokenizer_logreg, morph=morph).toarray()
            
            pred = clf.predict_proba(vector)
            print(text)
            print(pred)
            pred_df = pd.DataFrame({'label': mapping.keys(), 'score': pred[0]})

            pred_df = pred_df.sort_values(by='score', ascending=False).reset_index(drop=True)
            pred_df_styled = pred_df.style.background_gradient(cmap='Blues')
            st.dataframe(pred_df_styled)
            

            with st.spinner('Загружаю интерпретацию...'):
                st.subheader('Интерпретация')
                explainer = shap.LinearExplainer(clf, x_train, feature_dependence="interventional")
                shap_values = explainer.shap_values(vector)

                st.write('Neutral')
                idx = 0
                st_shap(shap.force_plot(
                    explainer.expected_value[idx], shap_values[idx][0], vector,
                    feature_names=cv.get_feature_names_out()
                ))
                
                st.write('Positive')
                idx = 1
                st_shap(shap.force_plot(
                    explainer.expected_value[idx], shap_values[idx][0], vector,
                    feature_names=cv.get_feature_names_out()
                ))
                
                st.write('Negative:')
                idx = 2
                st_shap(shap.force_plot(
                    explainer.expected_value[idx], shap_values[idx][0], vector,
                    feature_names=cv.get_feature_names_out()
                ))
        else:
            text = input_text
            prediction = pipe([text])
            print(text)
            print(prediction[0])

            pred_df = pd.DataFrame(prediction[0])
            pred_df = pred_df.sort_values(by='score', ascending=False).reset_index(drop=True)
            
            pred_df_styled = pred_df.style.background_gradient(cmap='Blues')
            st.dataframe(pred_df_styled)

            with st.spinner('Загружаю интерпретацию...'):
                st.subheader('Интерпретация')
                shap_values = bert_explainer([text])
                st.write('Neutral')
                st_shap(shap.force_plot(base_value=shap_values.base_values[0][0],
                            shap_values=shap_values.values[0, :, 0],
                            feature_names=shap_values.data[0], show=False))

                st.write('Positive')
                st_shap(shap.force_plot(base_value=shap_values.base_values[0][1],
                            shap_values=shap_values.values[0, :, 1],
                            feature_names=shap_values.data[0], show=False))

                st.write('Negative:')
                st_shap(shap.force_plot(base_value=shap_values.base_values[0][2],
                            shap_values=shap_values.values[0, :, 2],
                            feature_names=shap_values.data[0], show=False))

    process = psutil.Process(os.getpid())
    print('HOW MANY:', process.memory_info().rss / 1024 / 1024) 


if __name__== '__main__':
    main()
