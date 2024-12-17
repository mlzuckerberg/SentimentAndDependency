import spacy
from spacy.training import Example
from typing import Iterable, List, Tuple
import numpy as np
from spacy.scorer import Scorer
from spacy.util import minibatch
import re
import random

seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)

#USEFUL LINKS
#https://spacy.io/api/doc
#https://spacy.io/api/example
#https://spacy.io/api/textcategorizer




def process_kaggle(file_path: str) -> List[Tuple[str, dict[str, dict[str, float]]]]:
    """
    Process a text file with sentences and binary sentiment labels.
    
    Args:
        file_path (str): Path to the input text file
    
    Returns:
        List of tuples, each containing:
        - Lowercase processed sentence
        - Sentiment dictionary with POSITIVE and NEGATIVE probabilities
    """
    processed_data = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Split the line into text and label
            parts = line.strip().split('\t')
            
            # Handle cases with unexpected line format
            if len(parts) != 2:
                continue
            
            # Extract text and label
            text, label = parts[0], parts[1]
            
            # Preprocess text: lowercase, remove extra whitespace
            cleaned_text = re.sub(r'\s+', ' ', text.lower()).strip()
            
            # Create sentiment dictionary based on label
            sentiment = {
                "cats": {
                    "POSITIVE": float(label),
                    "NEGATIVE": 1.0 - float(label)
                }
            }
            
            processed_data.append((cleaned_text, sentiment))
    return processed_data





TRAINING_PATH = 'training_set.txt'
DEV_PATH = "dev_set.txt"
TEST_PATH = "test_set.txt"

TRAINING_DATA = process_kaggle(TRAINING_PATH)
DEV_DATA = process_kaggle(DEV_PATH)
TEST_DATA = process_kaggle(TEST_PATH)



nlp = spacy.load("en_core_web_md") 

config = {
    "threshold": 0.5,
    "model": {
        "@architectures": "spacy.TextCatEnsemble.v2",
        "tok2vec": {
            "@architectures": "spacy.Tok2Vec.v2",
            "embed": {
                "@architectures": "spacy.MultiHashEmbed.v2",
                "width": 64,
                "rows": [2000, 2000, 500, 1000, 500],
                "attrs": ["NORM", "LOWER", "PREFIX", "SUFFIX", "SHAPE"],
                "include_static_vectors": False,
            },
            "encode": {
                "@architectures": "spacy.MaxoutWindowEncoder.v2",
                "width": 64,
                "window_size": 1,
                "maxout_pieces": 3,
                "depth": 2,
            },
        },
        "linear_model": {
            "@architectures": "spacy.TextCatBOW.v3",
            "length": 262144,
            "no_output_layer": False,
        },
    },
}



#a list of example( the datatype ) which textcat trains on. It's made of two docs
training_examples = [] #





def make_model_data(data: tuple[str, dict[str, dict[str, float]]]) -> Iterable[Example]:
    """the model trains on examples, so this is what I am feeding into it. I have to do it this way otherwise all the other features go to 0"""
    example_list = [] #the list of examples for the model
    for text, annotations in data:

        doc_gold = nlp(text)
        doc_guess = nlp(text)

        doc_gold.cats = annotations["cats"]

        an_example = Example(doc_guess, doc_gold)
        example_list.append(an_example)

    return example_list

training_examples = make_model_data(TRAINING_DATA)

#nlp: pipe names['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner', 'textcat']
#why does textcat have to come after I fill up the trainig docs Idk
textcat = nlp.add_pipe("textcat") 
textcat.add_label("POSITIVE")
textcat.add_label("NEGATIVE")

# Confirm the pipeline components
print(f"nlp: pipe names: {nlp.pipe_names}")


#BEHOLD ACTUALLY DOING THE MODEL THING
#{'learn_rate': 0.002, 'epoch': 5, 'drop': 0.1, 'batch': 2}



def train(train_example, a_batch_size, an_epoch, a_dropnum, a_learnrate):
    """this actually tains the model based on the data that is examples """
    optimizer = nlp.begin_training()  # Initialize optimizer
    optimizer.learn_rate = a_learnrate

    # Set all components to be updated during training (this is important)
    for epoch in range(an_epoch):  # Number of epochs
        losses = {}
        # Create mini-batches
        batches = minibatch(train_example, size=a_batch_size)

        for batch in batches:
            # Update the model with the optimizer for all components
            # We pass 'drop' as a regularization parameter and use the optimizer
            #textcat.update(batch, drop=a_dropnum, losses=losses, sgd=optimizer)
            nlp.update(batch, drop=a_dropnum, losses=losses, sgd=optimizer)
        
        print(f"Epoch {epoch + 1}, Losses: {losses}")


import pickle

def sweep():
    EPOCH_NUM = 7
    BATCH_SIZE = int( len(TRAINING_DATA) /EPOCH_NUM )
    DROP_NUMS = [0.01, 0.02, 0.03, 0.04]
    LEARN_RATES = [0.001, 0.002, 0.003, 0.004]

    data = []
    for drop in DROP_NUMS:
        for rate in LEARN_RATES:
            
            #does the training
            train(training_examples, BATCH_SIZE, EPOCH_NUM, drop, rate)
            
            #scores and stores the result
            dev_examples = make_model_data(DEV_DATA)
            scorer = Scorer()
            scores = scorer.score_cats(dev_examples, "cats", labels=["POSITIVE", "NEGATIVE"], multi_label=False)
            accuracy = scores['cats_score']
            datum = {"accuracy": accuracy, "lr": rate, "drop": drop}
            data.append(datum)

    file_name = "best_scores.pkl" 
    with open(file_name, "wb") as file:  # Open the file in write-binary mode
        pickle.dump(data, file)
#scores the model


sweep() #OH yeah do the gridsearch is my favorite dance move
