import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import pandas as pd
from transformers import TFGPT2Model, GPT2Tokenizer
from transformers import LongformerTokenizer, TFLongformerModel
from error_logger import log_error
from InstructorEmbedding import INSTRUCTOR
from transformers import BertTokenizer, TFBertModel


class TextEmbedding:
    """
    Class to convert text into word embeddings using various LLM models

    Attributes
    :text: Text to be embedded
    :model: default - 'gpt-2'

    Methods:
        :get_model_names(): returns list of model names()
        :get_embeddings()
    """
    def __init__(self, text_file, model='gpt-2'):
        self.filename = text_file
        self.text = self.read_txt()
        self.model = model
        self.list_of_models = ['bert-base-uncased', 'gpt-2', 'longformer-base-4096', 'instructor-xl']

    def read_txt(self):
        """
                Method: read_txt
                Reads text in the text file
                :return: a large text string
                """

        try:
            with open(self.filename, mode='r', errors="ignore") as file:
                whole_text = file.read()
                return whole_text

        except Exception as error:
            log_error(f"{error}", "read_txt",
                      "TextEmbedding.py")

    def get_model_names(self):
        """
        Method to get a list of models available
        Prints list of available models
        :return: List of model names
        """
        print('The LLM models available are:')
        for i in self.list_of_models:
            print(i)
        return self.list_of_models

    def get_gpt2_embeddings(self):
        """
        Method to get text embeddings from GPT2 model
        :return: embeddings tensor
        """
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = TFGPT2Model.from_pretrained('gpt2')
        # encode the text
        inputs = tokenizer(self.text, return_tensors="tf", truncation=True)
        # get model output
        outputs = model(inputs)
        # outputs.last_hidden_state has shape [batch_size, sequence_length, hidden_size]
        embeddings = outputs.last_hidden_state
        # get pooled embeddings
        embeddings_mean = tf.reduce_mean(embeddings, axis=1)
        return pd.DataFrame(embeddings_mean)

    def get_longformer_embeddings(self):
        """
        Method to get text embeddings from Longformer model
        :return: text embeddings tensor
        """
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        model = TFLongformerModel.from_pretrained('allenai/longformer-base-4096')
        # encode the text
        inputs = tokenizer(self.text, truncation=True, padding='max_length', max_length=4096, return_tensors='tf')
        outputs = model(inputs)
        # get embeddings
        embeddings = outputs.last_hidden_state
        embeddings = tf.reduce_mean(embeddings, axis=1)
        return pd.DataFrame(embeddings)

    def get_instructor_embeddings(self):
        """
        Method to get text embeddings from Instructor-XL model
        :return: text embeddings tensor
        """
        model = INSTRUCTOR('hkunlp/instructor-xl')
        sentence = self.text
        instruction = "AI or Human generated Text"
        # get embeddings
        embeddings = model.encode([[instruction, sentence]])
        return pd.DataFrame(embeddings)

    def get_bert_embeddings(self):
        """
        Method to get text embeddings from bert-based-uncased model
        :return: text embeddings tensor
        """
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = TFBertModel.from_pretrained("bert-base-uncased")
        text = self.text
        # encode text
        encoded_input = tokenizer(text, return_tensors='tf', truncation=True)
        output = model(encoded_input)
        # get embeddings
        embeddings = output.last_hidden_state
        embeddings_mean = tf.reduce_mean(embeddings, axis=1)
        return pd.DataFrame(embeddings_mean)

    def get_embeddings(self):
        """
        Method to get the embeddings of text from the selected LLM model
        :return: text embeddings tensors
        """
        if self.model == 'gpt-2':
            try:
                embeddings = self.get_gpt2_embeddings()
                return embeddings
            except Exception as e:
                log_error(f"{e}", "get_gp2_embeddings",
                          "TextEmbedding.py")
        if self.model == 'longformer-base-4096':
            try:
                embeddings = self.get_longformer_embeddings()
                return embeddings
            except Exception as e:
                log_error(f"{e}", "get_longformer_embeddings",
                          "TextEmbedding.py")
        if self.model == 'instructor-xl':
            try:
                embeddings = self.get_instructor_embeddings()
                return embeddings
            except Exception as e:
                log_error(f"{e}", "get_instructor_embeddings",
                          "TextEmbedding.py")
        if self.model == 'bert-base-uncased':
            try:
                embeddings = self.get_bert_embeddings()
                return embeddings
            except Exception as e:
                log_error(f"{e}", "get_bert_embeddings",
                          "TextEmbedding.py")

