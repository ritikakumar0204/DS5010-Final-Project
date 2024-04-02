"""
    DS 5010 - Final Project
    Class Name: TextPreprocessing - a class for the text preprocessing task
    Created by Xin Wang
"""

from error_logger import log_error

import nltk
import spacy # for lemmatization

# For stopwords removal
from nltk.corpus import stopwords
nltk.download('stopwords')

class TextPreprocessing():
    """
    Class: TextPreprocessing
        Preprocess the given text in a text file
        This class is for vectorization methods like Word2Vec, GloVe

    Methods (Process steps):
    1. read_txt: Load the text
    2. remove_punctuation:
        Lowercase the text and remove the non-alphanumeric characters
    3. tokenize:
        Tokenize the text, splitting the text into a list of words
    4. remove_stopwords:
        Use NLTK's stopwords library to remove stopwords in the list of tokens
    5. lemmatize:
        Lemmatize the tokens in the list. E.g. better -> well, cats -> cat
    6. preprocess:
        Preprocess the text, return the list of processed tokens

    Attributes:
        filename - str - the name of the text file
        text - str - the text in the file
    """

    def __init__(self, textfile):
        """
        This is the constructor of the TextPreprocessing class
        Attributes:
            filename - str - the name of the text file
            text - str - the text in the file
        """

        self.filename = textfile

        text = self.read_txt()
        if text is not None:
            self.text = text

    def read_txt(self):
        """
        Method: read_txt
            Reads text in the text file

        :return: a large text string
        """

        try:
            with open(self.filename, mode = "r", encoding = "utf-8") as file:
                whole_text = file.read()
                return whole_text

        except Exception as error:
            log_error(f"{error}", "read_txt",
                      "TextPreprocessing.py")
            return

    def remove_punctuation(self):
        """
        Method: remove_punctuation
            1. Lowercase the text
            2. Remove punctuation in the text

        :return: the lowercase text with punctuation removed
        """
        try:
            text = self.text.lower()
            return "".join([char for char in text if char.isalnum() or char.isspace()])

        except Exception as error:
            log_error(f"{error}", "remove_punctuation",
                      "TextPreprocessing.py")
            return

    def tokenize(self):
        """
        Method: tokenize
            To split the text into tokens
            by just split by the whitespace characters
        :return: a list of tokens
        """
        try:
            text = self.remove_punctuation()
            if text is not None:
                tokens = text.split()
                return tokens

        except Exception as error:
            log_error(f"{error}", "tokenize",
                      "TextPreprocessing.py")
            return

    def remove_stopwords(self):
        """
        Method: remove_stopwords
            To remove stopwords in the list of tokens
        :return: a list of tokens without stopwords
        """
        try:
            stop_words = set(stopwords.words("english"))
            tokens = self.tokenize()
            if tokens is not None:
                return [word for word in tokens if word not in stop_words]

        except Exception as error:
            log_error(f"{error}", "remove_stopwords",
                      "TextPreprocessing.py")
            return

    def lemmatize(self):
        """
        Method: lemmatize
            To lemmatize the tokens in the list without stopwords
        :return: a list of lemmatized tokens
        """
        try:
            model = spacy.load("en_core_web_sm")
            tokens = self.remove_stopwords()
            return [model(word)[0].lemma_ for word in tokens]

        except Exception as error:
            log_error(f"{error}", "lemmatize",
                      "TextPreprocessing.py")
            return

    def preprocess(self):
        """
        Method: preprocess
            Preprocess the texts and make them ready
            for downstream works like vectorization
        :return: a list of processed tokens
        """

        return self.lemmatize()