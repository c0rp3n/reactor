import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from status import Status
from statistics import process_data

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

def main():
    # load data
    headers, statuses, data = load_csv('reactor_data.csv')

    process_data(statuses, data)

    return

if __name__ == "__main__":
    main()
