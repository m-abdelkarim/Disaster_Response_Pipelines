import nltk
nltk.download(['punkt','wordnet','stopwords'])

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

def lemmatize_all(word_in):
    """
    The given word is lemmatized with all part of speech tags.

    Args:
    word_in: The word to be lemmatized.

    Returns:
    word: The lemmatized word.
    """
    word = WordNetLemmatizer().lemmatize(word_in, pos='n')
    word = WordNetLemmatizer().lemmatize(word, pos='v')
    word = WordNetLemmatizer().lemmatize(word, pos='a')
    word = WordNetLemmatizer().lemmatize(word, pos='r')
    word = WordNetLemmatizer().lemmatize(word, pos='s')
    return word

def tokenize(text_in):
    """
    normalize and tokenize the input text.

    Args:
    text_in: The text to be normalized and tokenized.

    Returns:
    tokens: The tokens of the input text.
    """
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text_in.lower())
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize and remove stop words
    tokens = [lemmatize_all(word)
              for word in tokens
              if word not in stopwords.words("english")]

    return tokens