"""
    Here's the unit tests for TextPreprocessing class.
"""

import unittest

# Test the printed message
from unittest.mock import patch
import io

# Test punctuations
import string

from ..TextPreprocessing import *


class TestTextPreprocessing(unittest.TestCase):
    """
    Class: TestTextPreprocessing
        This is a class to test the constructor and methods
        in class TextPreprocessing.

    Methods:
        setUp: Sets up the
        test_init
    """

    def setUp(self):
        """
        Method: setUp
            This method is to set up some files
            and the objects for testing.
        """

        self.text = ("The quick brown fox jumps over the lazy dog," +
                     "\nwho seems uninterested...\n\n'What a curious sight!' " +
                     "exclaimed the squirrel, watching from a nearby tree." +
                     "\nThe fox, undeterred, continues its journey." +
                     "\n\nSuddenly!!! A loud noise startles them all: " +
                     "it's thunder!!\nRain begins to fall heavily, " +
                     "soaking everything in its path...\nThe animals scatter, " +
                     "seeking shelter from the storm.\n\nAmidst the chaos; " +
                     "the fox finds a cozy den and curls up,\nwaiting for the " +
                     "tempest to pass...")

        with open("./whowrotethis/tests/test.txt", mode="w", encoding = "utf-8") as file:
            file.write(self.text)

        self.text1 = TextPreprocessing("test.txt", file_given=True)
        self.text2 = TextPreprocessing(self.text, file_given=False)
        self.text3 = TextPreprocessing("", file_given=False)
        self.text4 = TextPreprocessing("Lemmatize the tokens in the list. " +
                                       "\n\n E.g. better -> well, cats -> cat",
                                       file_given=False)

    def test_init(self):
        """
        Method: test_init
            This method is to test the constructor of class TextPreprocessing

        Test case 1:
            Test the TextPreprocessing object self.text1
            initialized by giving filename

        Test case 2:
            Test the TextPreprocessing object self.text2
            initialized by giving a non-empty text string

        Test case 3:
            Test the TextPreprocessing object self.text2
            initialized by giving an empty string
        """

        # Test case 1
        self.assertEqual(self.text, self.text1.get_text(),
                         "Wrong-Test case 1 in test_init method")

        # Test case 2
        self.assertEqual(self.text, self.text2.get_text(),
                         "Wrong-Test case 2 in test_init method")

        # Test case 3
        self.assertEqual("", self.text3.get_text(),
                         "Wrong-Test case 3 in test_init method")

    def test_read_txt(self):
        """
        Method: test_read_txt
            This method is to test the read_txt method
            in class TextPreprocessing

        Test case 1:
            Read the whole text in file test.txt and
            check its contents with self.text
        """

        self.assertEqual(self.text, self.text3.read_txt("test.txt"),
                         "Test case 1 in test_read_txt method failed.")

    def test_set_text(self):
        """
        Method: test_set_text
            This method is to test the set_text method in class TextPreprocessing

        Test case 1:
            set test by giving the filename

        Test case 2:
            set test by giving the whole text string


        """

        # Test case 1
        self.text3.set_text("test.txt", file_given=True)
        self.assertEqual(self.text, self.text3.get_text(),
                         "Test case 1 in test_set_text method failed.")

        # Test case 2
        a_text = ("Lemmatize the tokens in the list. \n\n " +
                  "E.g. better -> well, cats -> cat\n")
        self.text3.set_text(a_text, file_given=False)
        self.assertEqual(a_text, self.text3.get_text(),
                         "Test case 2 in test_set_text method failed.")

    def test_get_text(self):
        """
        Method: test_get_text
            This is a method to test the get_text method
            in class TextPreprocessing

        Test case 1:
            Check the text in object self.text1

        Test case 2:
            Check the text in object self.text2
        """

        self.assertEqual(self.text, self.text1.get_text(),
                         "Test case 1 in test_get_text method failed.")

        self.assertEqual(self.text, self.text2.get_text(),
                         "Test case 2 in test_get_text method failed.")

    def test_remove_punctuation(self):
        """
        Method: test_remove_punctuation
            This is a method to test the remove_punctuation method
            in class TextPreprocessing

        Test case:
            Check if there are any punctuation and uppercase letters in the text
            after using the remove_punctuation method.
        """

        text = self.text1.remove_punctuation()

        # Check punctuations
        self.assertFalse(any(char in string.punctuation for char in text),
                         "Failed to remove punctuation.")

        # Check if there are uppercase letters
        self.assertFalse(any(char.isupper() for char in text),
                         "Failed to lower the letters.")

    def test_tokenize(self):
        """
        Method: test_tokenize
            This is a method to test the tokenize method
            in class TextPreprocessing

        Test case:
            Check if the output in object self.text4 is tokenized
            after using the tokenize method.
        """

        tokens = self.text4.tokenize()
        expected_tokens = ['lemmatize', 'the', 'tokens', 'in', 'the', 'list',
                           'eg', 'better', 'well', 'cats', 'cat']

        self.assertEqual(expected_tokens, tokens,
                         "Failed to tokenize the text.")

    def test_remove_stopwords(self):
        """
        Method: test_remove_stopwords
            This is a method to test the remove_stopwords method
            in class Textpreprocessing

        Test case:
            Check if there's no stopwords in the output of self.text4
            after using the remove_stopwords method.
        """

        tokens = self.text4.remove_stopwords()
        expected_tokens = ['lemmatize', 'tokens', 'list', 'eg',
                           'better', 'well', 'cats', 'cat']
        self.assertEqual(expected_tokens, tokens,
                         "Failed to remove stopwords.")

    def test_lemmatize(self):
        """
        Method: test_lemmatize
            This is a method to test the lemmatize method
            in class TextPreprocessing

        Test case:
            Check if the output of self.text4 is lemmatized
            after using the lemmatize method
        """

        tokens = self.text4.lemmatize()
        expected_tokens = ['lemmatize', 'tokens', 'list', 'eg',
                           'well', 'well', 'cat', 'cat']
        self.assertEqual(expected_tokens, tokens,
                         "Failed to lemmatize the tokens.")

    def test_preprocess(self):
        """
        Method: test_preprocess
            This is a method to test the preprocess method
            in class TextPreprocessing

        Test case:
            Check if the output of self.text4 is preprocessed
            after using the preprocess method
        """

        processed_text = self.text4.preprocess()
        expected_processed_text = "lemmatize tokens list eg well well cat cat"
        self.assertEqual(expected_processed_text, processed_text,
                         "Failed to preprocess the text.")

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_bad_init(self, mock_stdout):
        """
        Method: test_bad_init
        This is a method to test the constructor of class TextPreprocessing
        when giving a wrong txt filename
        """

        text1 = TextPreprocessing("file", file_given=True)
        text2 = TextPreprocessing("txt", file_given=True)
        error_messages = mock_stdout.getvalue().splitlines()

        self.assertEqual("Please give a correct txt file name.",
                         error_messages[0],
                         "Test case 1 in test_bad_init method failed.")

        self.assertEqual("Please give a correct txt file name.",
                         error_messages[1],
                         "Test case 2 in test_bad_init method failed.")

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_bad_set_text(self, mock_stdout):
        """
        Method: test_bad_set_text
            This method is to test the set_text method in class TextPreprocessing
        Test case:
            if the given file doesn't exist then keep the text unchanged
            and print out a message.
        """

        self.text3.set_text("not_exist.txt", file_given=True)

        self.assertEqual("The text has not been changed.",
                         mock_stdout.getvalue().strip(),
                         "Test case in test_bad_set_text method failed")


def main():
    unittest.main(verbosity=3)


if __name__ == "__main__":
    main()