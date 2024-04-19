"""
Program to demonstrate usage of whowrotethis
python module
"""
from whowrotethis import TextPreprocessing, TextEmbedding, Classifier

def main():
    embeddings = TextEmbedding.TextEmbedding('text.txt', model='bert-base-uncased').get_embeddings()
    predict = Classifier.Classifier(embeddings)
    print(predict.predict_text())


if __name__ == "__main__":
    main()
