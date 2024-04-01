"""
    DS 5010 - Final Project
    Class Name: TextPreprocessing - a class for the text preprocessing task
    Created by Xin Wang
"""

from error_logger import log_error

class TextPreprocessing():
    def __init__(self, textfile):
        self.filename = textfile

    def read_txt(self):
        """
        Method: read_txt
        Reads text in the text file
        :return: a large text string
        """

        try:
            with open(self.filename, mode='r', encoding='utf-8') as file:
                whole_text = file.read()
                return whole_text

        except Exception as error:
            log_error(f"{error}", "read_txt",
                      "TextPreprocessing.py")

    def remove_punctuation(self):
        """
        Method: remove_punctuation
            1. Lowercase the text
            2. Remove punctuation in the text
        :return: the lowercase text with punctuation removed
        """
        text = self.text.lower()
        return "".join([char for char in text if char.isalnum() or char.isspace()])

    def stemming(self):

    def remove_stopwords(self):

    def tokenize(self, model_name = None):

    def preprocess(self):