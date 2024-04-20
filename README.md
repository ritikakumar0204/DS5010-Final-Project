# Who wrote this

Python package to distinguish text data into human vs AI generated. This package provides a straightforward interface for verifying the authenticity of text, AI or Human generated.

## Features
- **TextPreprocessing** - Preprocess text data and convert it into tokens.
- **TextEmbedding** - Convert Text data into word embeddings using LLM models.
- **Classifier** - Predict whether the text data is Human or AI generated

## Usage
1. Install requirements:
```
pip install -r requirements.txt
```
2. Text Preprocessing
To preprocess text data for classification:

```
from WhoWroteThis import TextPreprocessing

text = "Example text that needs to be classified."
processed_text = TextPreprocessing.preprocess(text)
```

3. Generating Embeddings
Generate embeddings using the default model (GPT-2):
```
from WhoWroteThis import TextEmbedding

embeddings = TextEmbedding.get_embeddings(processed_text)
```
4. Classification 
Classify the text as human or AI-generated:
```
from WhoWroteThis import Classifier

prediction = Classifier.predict(embeddings)
print("This text was written by:", "Human" if prediction == 0 else "AI")
```
