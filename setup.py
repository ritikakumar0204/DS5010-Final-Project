from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Human Vs AI Text Classifier'
LONG_DESCRIPTION = 'A package that allows user to identify whether a text is human generated or AI generated'

# Setting up
setup(
    name="whowroteit",
    version=VERSION,
    author='Ritika Kumar, Xin Wang, Ardavan Mehdizadeh',
    author_email='kumar.riti@northeastern.edu',
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'spacy', 'nltk', 'matplotlib', 'transformers', 'sentence-transformers==2.2.2', 'xgboost', 'InstructorEmbedding', 'pickle' ],
    keywords=['python', 'classification', 'AIvsHuman', 'text']
)
