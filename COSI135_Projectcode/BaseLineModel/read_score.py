import pickle
from collections import defaultdict, Counter

with open('best_scores.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
def average__accuracy(dataset):
    the_sums = 0
    occurences = len(dataset)

    for datum in dataset:
        the_sums += datum["accuracy"]

    return the_sums/occurences


the_average_accuracy = average__accuracy(data)

print(the_average_accuracy)