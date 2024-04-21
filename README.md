# Who wrote this

Python package to distinguish text data into human vs AI generated. This package provides a straightforward interface for verifying the authenticity of text, AI or Human generated.

## Features
- **TextPreprocessing** - Preprocess text data and convert it into tokens.
- **TextEmbedding** - Convert Text data into word embeddings using LLM models.
- **Classifier** - Predict whether the text data is Human or AI generated.
- **EvaluateModel** - Evaluates the model on 10,000 samples
- **EvaluateClassifier** - Evaluates the ensemble models
- **UserApp** - Runs the Streamlit app on localhost for UI of WhoWroteThis

## Usage
1. Install requirements:
```
pip install -r requirements.txt
```
2. Text Preprocessing

To preprocess text data for classification

```
from WhoWroteThis import TextPreprocessing

text = "Example text that needs to be classified."
processed_text = TextPreprocessing.preprocess(text)
```

3. Generating Embeddings

Generate embeddings using LLM model
```
from WhoWroteThis import TextEmbedding

embeddings = TextEmbedding('text.txt', model='gpt-2').get_embeddings()
```

4. Predict

Predict the text as human or AI-generated
```
from WhoWroteThis import Classifier

prediction = Classifier.predict(embeddings)
print(prediction)
```

5. Evaluate Model

Evaluation on dataset

```angular2html

# Load data
data = pd.read_csv(
f'{os.getcwd()}\\whowrotethis\\data\\10k_raw_unseen.csv')
x_test = data.loc[:, '0' : '767']
y_test = data['label']

# Get predictions
model = EnsembledModel(x_test)
y_pred_1 = model.simple_predict()
y_pred_2 = model.weighted_predict()

# Prepare figure
fig, axs = plt.subplots(1, 2, figsize=(15, 5), tight_layout=True)
count = 0
fig.suptitle("Number of wrong predictions using the ensembled model")

# Evaluate models
print("Weighted Ensemble Model:")
evaluate(y_test, y_pred_2, axs, count, "Weighted Ensembled Model")
count += 1
print("-" * 30)

print("Unweighted Ensemble Model:")
evaluate(y_test, y_pred_1, axs, count, "Unweighted Ensembled Model")

plt.show()
```

6. Interface for text detection

Web user interface for the user to input text and display prediction

```angular2html
from whowrotethis import UserApp
UserApp().run_app()
```
