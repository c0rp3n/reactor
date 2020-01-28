import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

HIDDEN_NODES = 500
HIDDEN_LAYERS = 2

MAX_ITERATIONS = 1000

# setting this too low makes everything change very slowly, but too high
# makes it jump at each and every example and oscillate. I found .5 to be good
LEARNING_RATE = .2

def run_network_sec3(data_train, data_test, expected_train, expected_test):
    file = open('results/sec3_ann.txt', 'w')

    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes=(HIDDEN_NODES, HIDDEN_NODES),
                        random_state=1)

    clf.fit(data_train, expected_train)

    file.write('ANN - Accuracy: {}\n'.format(clf.score(data_test, expected_test)))
    file.close()

def run_network_sec4(data, expected):
    file = open('results/sec4_ann.txt', 'w')

    for hidden_nodes in [50, 500, 1000]:
        clf = MLPClassifier(solver='lbfgs',
                            alpha=1e-5,
                            hidden_layer_sizes=(hidden_nodes, hidden_nodes),
                            random_state=1)
        
        scores = cross_val_score(clf, data, expected, cv=10)
        file.write('ANN (Hidden Nodes: {}) - Accuracy: {} (+/- {})\n'.format(hidden_nodes, scores.mean(), scores.std() * 2))

    file.close()