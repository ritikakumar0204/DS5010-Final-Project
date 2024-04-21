"""
Program to demonstrate usage of whowrotethis
python module
"""

from whowrotethis import TextPreprocessing, TextEmbedding, Classifier, UserApp
# import whowrotethis


def main():
    # For text preprocessing
    text = "This is a test."
    processor = TextPreprocessing(text, file_given=False)
    preprocessed_text = processor.preprocess()
    print(preprocessed_text)

    # For text embeddings
    embeddings = TextEmbedding('report.txt').get_embeddings()
    print(embeddings)
    predict = Classifier(embeddings)
    print(predict.predict_text())

    # for streamlit app (uncomment below)
    UserApp().run_app()



if __name__ == "__main__":
    main()
