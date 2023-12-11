# Import necessary libraries
import json
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import random

# Initialize a blank spaCy English model
nlp = spacy.blank("en")

# Function to process data and create a DocBin
def process_data(file_path, nlp):
    db = DocBin()
    with open(file_path, 'r') as f:
        data = json.load(f)
    for text, annot in data['annotations']:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None:
                ents.append(span)
        doc.ents = ents
        db.add(doc)
    return db

# Load and process the dataset
file_path = 'mountains_data.json'  # Update this path
db = process_data(file_path, nlp)

# Split the data into training and validation
train_data = list(db.get_docs(nlp.vocab))
random.shuffle(train_data)
split = int(len(train_data) * 0.8)
train_examples = [Example.from_dict(doc, {"entities": [(e.start_char, e.end_char, e.label_) for e in doc.ents]}) for doc in train_data[:split]]
valid_examples = [Example.from_dict(doc, {"entities": [(e.start_char, e.end_char, e.label_) for e in doc.ents]}) for doc in train_data[split:]]

# Add an NER pipe to the model
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Add labels to the NER model
for example in train_examples:
    for ent in example.reference.ents:
        ner.add_label(ent.label_)

# Training loop
for itn in range(10):  # Number of training iterations
    random.shuffle(train_examples)
    losses = {}
    for batch in spacy.util.minibatch(train_examples, size=2):  # Batch size
        nlp.update(
            [example for example in batch],
            drop=0.5,  # Dropout - make it harder to memorize data
            losses=losses,
        )
    print(f"Losses at iteration {itn}: {losses}")

# Save the trained model
nlp.to_disk("model_output") 

print("Model trained and saved successfully.")
