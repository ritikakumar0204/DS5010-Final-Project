"""
    DS 5010 - Final Project
    Class Name: TextPreprocessing - a class for the text preprocessing task
    Created by Xin Wang
"""

from .error_logger import log_error

import nltk

# For stopwords removal
from nltk.corpus import stopwords
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')


class TextPreprocessing():
    """
    Class: TextPreprocessing
        Preprocess the given text in a text file
        This class is for vectorization methods like Word2Vec, GloVe

    Used packages:
        1. NLTK
        2. SpaCy
            To install model for lemmatization, the command line is:
                python -m spacy download en_core_web_sm

    Methods (Process steps):
    1. read_txt: Load the text in a txt file
    2. set_text: Set a new text
    3. get_text: Get the current text
    4. remove_punctuation:
        Lowercase the text and remove the non-alphanumeric characters
    5. tokenize:
        Tokenize the text, splitting the text into a list of words
    6. remove_stopwords:
        Use NLTK's stopwords library to remove stopwords in the list of tokens
    7. lemmatize:
        Lemmatize the tokens in the list. E.g. better -> well, cats -> cat
    8. preprocess:
        Preprocess the text, return a processed text (str)

    Attributes:
        filename - str - the name of the text file
        text - str - the text in the file
    """

    def __init__(self, text, file_given=False):
        """
        This is the constructor of the TextPreprocessing class
        Attributes:
            text - str - the text ready to be processed or the txt file name
            file_given - bool
                - True if the parameter text indicates the filename
                    (text = "filename.txt")
                - (Default) False if the parameter text indicates the whole text
                    (text = "This is a text.")
        """
        try:
            if file_given:
                if len(text) < 4 or (text[-4:] != ".txt"):
                    print("Please give a correct txt file name.")
                    raise ValueError("A wrong txt filename were given " +
                                     "while initializing the " +
                                     "TextPreprocessing object.")

                current_text = self.read_txt(text)
            else:
                current_text = text

            if current_text is not None:
                self.text = current_text
            else:
                self.text = None
                raise ValueError("Did not give a valid text or filename.")

            self.lemmatizer = WordNetLemmatizer()

        except Exception as error:
            log_error(f"{error}", "__init__",
                      "TextPreprocessing.py")
            return

    def read_txt(self, filename):
        """
        Method: read_txt
            Reads text in the text file
        :param: filename - str - the name of the text file
        :return: a large text string
        """

        try:
            with open(filename, mode = "r", encoding = "utf-8") as file:
                whole_text = file.read()
                return whole_text

        except Exception as error:
            log_error(f"{error}", "read_txt",
                      "TextPreprocessing.py")
            return

    def set_text(self, text, file_given=False):
        """
        Method: set_text
            To set a new text

        :param:
            text - str - the text ready to be processed or the txt file name
            file_given - bool
                - True if the parameter text indicates the filename
                    (text = "filename.txt")
                - False if the parameter text indicates the whole text
                    (text = "This is a text.")
        """
        try:
            if file_given:
                current_text = self.read_txt(text)
            else:
                current_text = text

            if current_text is not None:
                self.text = current_text
            else:
                print("The text has not been changed.")

        except Exception as error:
            log_error(f"{error}", "set_text",
                      "TextPreprocessing.py")

    def get_text(self):
        """Returns the current text of the instance"""
        return self.text

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
            tokens = self.remove_stopwords()
            return [self.lemmatizer.lemmatize(word) for word in tokens]

        except Exception as error:
            log_error(f"{error}", "lemmatize",
                      "TextPreprocessing.py")
            return

    def preprocess(self):
        """
        Method: preprocess
            Preprocess the texts and make them ready
            for downstream works like vectorization
        :return: a string which joins a list of processed tokens together
        """

        return " ".join(self.lemmatize())
