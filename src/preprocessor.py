from nltk import word_tokenize
import pandas as pd
import re
import nltk
import spacy
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
spacy_model = spacy.load("en_core_web_sm")
ps = PorterStemmer()

def string_cleaner(s: str) -> str:
    lowercased = s.lower()
    cleaned_punc = re.sub(r'[^\w\s]',' ',lowercased)
    unattached_digits_removal = re.sub(r'\b\d+\b',' ',cleaned_punc)
    sentence = ' '.join(word_tokenize(unattached_digits_removal))
    return sentence

def generate_merged_df() :
    acl_df = pd.read_csv('../data/interim/acl.csv')
    arxiv_df = pd.read_csv('../data/interim/arxiv_data_210930-054931.csv')
    acl_df = acl_df[acl_df.columns[1:]]
    arxiv_df = arxiv_df[arxiv_df.columns[1:]]
    acl_df = acl_df.rename(columns={'title':'titles', 'abstract':'abstracts'})
    merged_df = pd.concat([acl_df, arxiv_df])
    merged_df = merged_df.reset_index(drop=True)
    return merged_df

def tokenized_pos_tagger(tokens) :
    pos_tag = nltk.pos_tag(tokens)
    tag = [e[1] for e in pos_tag]
    return tag

def stemmer(word_tokens):
    stemmed = [ps.stem(w) for w in word_tokens]
    return ' '.join(stemmed)

def lemmatizer(str):
    doc = spacy_model(str)
    lemmatized = [token.lemma_ for token in doc]
    return ' '.join(lemmatized)

def NER_tagging(str):
    ner_words = []
    doc = spacy_model(str)
    for ent in doc.ents:
        ner_words.append((ent.label_, ent.text))
    return ner_words

def tf_idf(sentences) :
    tfidf = TfidfVectorizer()
    tfidf.fit_transform(sentences)
    return [(feature_name, value) for feature_name, value in zip(tfidf.get_feature_names_out(), tfidf.idf_)]