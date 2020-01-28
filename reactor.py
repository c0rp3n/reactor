import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from status import Status
from statistics import process_data

from network import (run_network_sec3, run_network_sec4)
from forest import (run_forest_sec3, run_forest_sec4)

def load_csv(path: str) -> [list, np.ndarray, np.ndarray]:
    import csv
    reader = csv.reader(open(path), delimiter=',')
    data = list(reader)

    headers = data[:1]
    statuses = []
    data = data[1:] # remove leading row
    for i, row in enumerate(data):
        statuses.append(Status[row[:1][0]].value) # convert status to an int
        data[i] = row[1:] # remove leading column on each remaining row
    
    return headers, np.array(statuses, dtype=np.int8), np.array(data, dtype=np.float64)

def sec3(data, expected):
    data_train, data_test, expected_train, expected_test = train_test_split(
        data, expected, test_size=0.1, shuffle=True, random_state=1)

    run_network_sec3(data_train, data_test, expected_train, expected_test)
    #run_forest_sec3(data_train, data_test, expected_train, expected_test)

def sec4(data, expected):
    from sklearn.utils import shuffle
    data, expected = shuffle(data, expected, random_state=1)

    run_network_sec4(data, expected)
    run_forest_sec4(data, expected)

def main():
    # load data
    headers, statuses, data = load_csv('reactor_data.csv')

    process_data(statuses, data)

    sec3(data, statuses)
    #sec4(data, statuses)

    return

if __name__ == "__main__":
    main()
