import unittest
from whowrotethis.TextEmbedding import TextEmbedding
import os
import numpy as np
import pandas as pd


class TestTextEmbeddings(unittest.TestCase):
    def test_get_gpt2_embeddings(self):
        path = os.getcwd()
        embeddings_saved = pd.DataFrame(np.load(f'{path}\\whowrotethis\\data\\embeddings_gpt_2.npy'))
        embeddings = TextEmbedding(f'{path}\\text.txt').get_embeddings()
        self.assertEqual(embeddings_saved.mean(axis=1).item(), embeddings.mean(axis=1).item())
    def test_get_bert_embeddings(self):
        path = os.getcwd()
        embeddings_saved = pd.DataFrame(np.load(f'{path}\\whowrotethis\\data\\embeddings_bert.npy'))
        embeddings = TextEmbedding(f'{path}\\text.txt', model='bert-base-uncased').get_embeddings()
        self.assertEqual(embeddings_saved.mean(axis=1).item(), embeddings.mean(axis=1).item())

    def test_get_longformer_embeddings(self):
        path = os.getcwd()
        embeddings_saved = pd.DataFrame(np.load(f'{path}\\whowrotethis\\data\\embeddings_longformer.npy'))
        embeddings = TextEmbedding(f'{path}\\text.txt', model='longformer-base-4096').get_embeddings()
        self.assertEqual(embeddings_saved.mean(axis=1).item(), embeddings.mean(axis=1).item())

    def test_get_instructor_embeddings(self):
        path = os.getcwd()
        embeddings_saved = pd.DataFrame(np.load(f'{path}\\whowrotethis\\data\\embeddings_instructor_xl.npy'))
        embeddings = TextEmbedding(f'{path}\\text.txt', model='instructor-xl').get_embeddings()
        self.assertEqual(embeddings_saved.mean(axis=1).item(), embeddings.mean(axis=1).item())


if __name__ == '__main__':
    unittest.main()
