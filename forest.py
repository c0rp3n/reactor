import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

TREE_COUNT = 1000
MIN_SAMPLES_LEAF = 5

def run_forest_sec3(data_train, data_test, expected_train, expected_test):
    file = open('results/sec3_rforests.txt', 'w')

    min_samples_leafs = [5, 50]
    for min_samples_leaf in min_samples_leafs:
        clf = RandomForestClassifier(n_estimators=TREE_COUNT, min_samples_leaf=min_samples_leaf, random_state=1)

        clf.fit(data_train, expected_train)

        file.write('RForest (Min Samples per Leaf: {}) - Accuracy: {}\n'.format(min_samples_leaf, clf.score(data_test, expected_test)))

    set_of_trees = [10, 50, 100, 1000, 5000]
    for number_of_trees in set_of_trees:
        clf = RandomForestClassifier(n_estimators=number_of_trees, min_samples_leaf=MIN_SAMPLES_LEAF, random_state=1)

        clf.fit(data_train, expected_train)

        file.write('RForest (Number of Trees: {}) - Accuracy: {}\n'.format(number_of_trees, clf.score(data_test, expected_test)))
    
    file.close()

def run_forest_sec4(data, expected):
    file = open('results/sec4_rforests.txt', 'w')

    set_of_trees = [20,  500, 10000]
    for number_of_trees in set_of_trees:
        clf = RandomForestClassifier(n_estimators=number_of_trees, min_samples_leaf=MIN_SAMPLES_LEAF, random_state=1)

        scores = cross_val_score(clf, data, expected, cv=10)

        file.write('RForest (Number of Trees: {}) - Accuracy: {} (+/- {})\n'.format(number_of_trees, scores.mean(), scores.std() * 2))
    
    file.close()
