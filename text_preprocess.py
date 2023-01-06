import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk
nltk.download('punkt')

def removeNonAscii(text):
  return ''.join(char for char in text if ord(char) < 128)

def casefolding(text):
  text = text.lower()
  text = re.sub(r'[-+]?[0-9]+', '', text)           # Menghapus angka
  text = re.sub(r'[^\w\s]','', text)                # Menghapus karakter tanda baca
  text = text.strip()                               # removing whitespaces
  text = re.sub(r'\r?\n|\r', ' ', text)             # removing new line
  text = removeNonAscii(text)

  return text


def prepareStopWord():
    stopword = StopWordRemoverFactory().get_stop_words()
    more_stopword = [
        'dah', 
        'yg', 
        'exp', 
        'jis', 
        'ory'
        ]

    final_stopword = stopword + more_stopword
    dic_stopword = ArrayDictionary(final_stopword)
    stopword_str = StopWordRemover(dic_stopword)

    return stopword_str
    

def text_filtering(text):
  stop = prepareStopWord().remove(text)
  token = word_tokenize(stop)
  return ' '.join(token)

def prepareStemmer():
    stemmer = StemmerFactory().create_stemmer()
    return stemmer

def text_stemming(text):
  return prepareStemmer().stem(str(text))

def getTfidfVec(X):
    tf_idf_vectorizer = TfidfVectorizer(ngram_range=(1,3))
    tf_idf_vectorizer.fit(X)
    return tf_idf_vectorizer

def getTfidfText(tfidfVec, text):
  text = casefolding(text)
  text = text_filtering(text)
  text = text_stemming(text)

  return tfidfVec.transform([text]).toarray()


def make_prediction(tfidfVec, model, text):
  tfidftext = getTfidfText(tfidfVec, text)

  #* 0 -> negative
  #* 1 -> positive
  return model.predict(tfidftext)[0]

def prepareDataset():
    return pd.read_csv('dataset/clean_df.csv')

def getSentimentPrediction(df, cls):
    if cls=='mustika':
        df = df['sentiment_predict'][df['class']==1]
    elif cls=='scarlet':
        df = df['sentiment_predict'][df['class']==2]
    elif cls=='wardah':
        df = df['sentiment_predict'][df['class']==3]
    else:
        raise ValueError('class not supported')

    return df

def get_prediction(pred_res) -> tuple:
  series_count = pred_res.count()

  negative_sentiment = pred_res[pred_res==0].count()/series_count
  positive_sentiment = pred_res[pred_res==1].count()/series_count

  return (negative_sentiment, positive_sentiment)

def get_prediction_res_by_class(df, cls) -> tuple:
  pred_res = None
  if cls=='mustika':
    pred_res = getSentimentPrediction(df, 'mustika')
  elif cls=='scarlet':
    pred_res = getSentimentPrediction(df, 'scarlet')
  elif cls=='wardah':
    pred_res = getSentimentPrediction(df, 'wardah')
  else:
    raise ValueError('class is not supported')

  return get_prediction(pred_res)