# Mountain NER Model

## Overview
This repository contains a Named Entity Recognition (NER) model specifically trained to identify mountain names in text. This model is built using spaCy, a powerful library for advanced Natural Language Processing (NLP) in Python.

## Dataset
The dataset used for training this model was generated using ChatGPT and manually annotated for mountain names. The annotation process involved marking out mountain names within the text using a free open-source NER Annotator tool available at [https://tecoholic.github.io/ner-annotator/](https://tecoholic.github.io/ner-annotator/). The annotated data is saved in JSON format.

### Data Annotation
The annotation process was meticulous, ensuring that each instance of a mountain name was correctly identified and labeled. This manual annotation step was crucial to create a high-quality dataset that the model could learn from effectively.

### Data Format
The dataset is structured in JSON, where each entry contains a text segment and corresponding annotations for mountain names. The annotations include the start and end positions of each mountain name in the text.

## Model Training
The model was trained using spaCy. The training process involved reading the annotated data, processing it to a format suitable for spaCy, and then training a custom NER model. Detailed steps and code for the training process are provided in the `train_model.py` script.

## Inference
To try out the model for inference:

### Setup
Ensure you have Python installed along with spaCy and other necessary libraries. Use `requirements.txt` to install dependencies.

### Running the Inference Script
Use the `inference_script.py` script to run inference. This script loads the trained NER model and uses it to identify mountain names in a given text. You can modify the script to input your text or use the provided examples.

#### Usage
```bash
python inference_script.py
