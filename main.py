"""
Program to demonstrate usage of whowrotethis
python module
"""

from whowrotethis import TextPreprocessing, TextEmbedding, Classifier
# import whowrotethis


def main():
    text = "This is a test."
    processor = TextPreprocessing(text, file_given=False)
    preprocessed_text = processor.preprocess()
    print(preprocessed_text)

    embeddings = TextEmbedding('text.txt').get_embeddings()
    predict = Classifier(embeddings)
    print(predict.predict_text())


if __name__ == "__main__":
    main()
