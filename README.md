# BERT for ABSA


## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Installation](#installation)
* [Usage](#usage)
* [Authors](#authors)
* [Acknowledgments](#acknowledgments)

## About The Project
Replication of the methodology proposed in the paper [Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence](https://arxiv.org/abs/1903.09588)

In this work (Targeted) Aspect-Based Sentiment Analysis task is converted to a sentence-pair classification task and a pre-trained [BERT](https://arxiv.org/abs/1810.04805) model is fine-tuned on it. 

More details about the project in the [presentation](presentation.pdf)

### Built With

* [:hugs: Transformers](https://github.com/huggingface/transformers)
* [PyTorch](https://pytorch.org/)

## Installation

To get a local copy up and running follow these simple steps:

1. Clone the repo
```sh
git clone https://github.com/LorenzoAgnolucci/BERT_for_ABSA.git
```
2. Run ```pip install -r requirements.txt``` in the root folder of the repo to install the requirements


## Usage

1. Run ```generate_datasets.py``` to build the datasets corresponding to each model or simply use the ones provided in ```data/sentihood/``` and ```data/semeval2014/```.

2. Use the forms in ```BERT_for_ABSA.ipynb``` to choose the desired dataset type and task both for BERT-single and BERT-pair. Then fine-tune the model and evaluate it runnning the corresponding cells.

3. Run the subsequent cells in ```BERT_for_ABSA.ipynb``` to fine-tune and evalaute the model.

## Authors

* [**Lorenzo Agnolucci**](https://github.com/LorenzoAgnolucci)

## Acknowledgments
Machine Learning © Course held by Professor [Paolo Frasconi](https://scholar.google.com/citations?user=s3l225EAAAAJ&hl=it) - Computer Engineering Master Degree @[University of Florence](https://www.unifi.it/changelang-eng.html)
