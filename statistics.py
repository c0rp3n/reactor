import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from status import Status

def get_statistics(data : np.ndarray) -> np.ndarray:
    stats = np.ndarray([7, data.shape[1]], dtype=np.float64)
    stats[0] = np.mean(data, axis=0)
    stats[1] = np.std(data, axis=0)
    stats[2] = np.amin(data, axis=0)
    stats[3] = np.quantile(data, 0.25, axis=0)
    stats[4] = np.median(data, axis=0)
    stats[5] = np.quantile(data, 0.75, axis=0)
    stats[6] = np.amax(data, axis=0)

    return stats

def save_statistics(filename : str, stats : np.ndarray):
    stat_types = ['mean', 'std', 'min', 'lwq', 'median', 'upq', 'max']

    file = open('results/' + filename + '_statistics.txt', 'w')
    row_len = stats.shape[1]
    for i in range(len(stats)):
        file.write(stat_types[i] + ': ')
        for i, val in enumerate(stats[i]):
            file.write('{0:0.6f}'.format(val))
            if i < row_len - 1:
                file.write(', ')
        file.write('\n')
    file.close()

def get_normal_count(statuses : np.ndarray) -> int:
    for i, status in enumerate(statuses):
        if status == Status.Abnormal.value:
            return i
    
    return len(statuses)

def plot_statistics(data_normal : np.ndarray, data_abnormal : np.ndarray):
    plt.cla()
    df = pd.DataFrame({
        'Normal': data_normal[8],
        'Abnormal': data_abnormal[8]
    })
    sns.boxplot(data=df, orient='h')
    plt.title('Boxplot of Vibration Sensor 1')
    plt.savefig('images/boxplot_vs1.jpg', format='jpeg')

    plt.cla()
    sns.distplot(data_normal[9], hist = False, kde = True,
                kde_kws = {'linewidth': 3},
                label = "Normal")
    
    sns.distplot(data_abnormal[9], hist = False, kde = True,
                kde_kws = {'linewidth': 3},
                label = "Abnormal")
    plt.legend(prop={'size': 16}, title = 'Status')
    plt.title('Density Plot for Vibration Sensor 2')
    plt.xlabel('Vibration Sensor 2')
    plt.ylabel('Density')
    plt.savefig('images/density_plot_vs2.jpg', format='jpeg')

def process_data(statuses : np.ndarray, data : np.ndarray):
    normal_count = get_normal_count(statuses)

    data_normal = data[:normal_count]
    data_abnormal = data[normal_count:]

    save_statistics('all', get_statistics(data))
    save_statistics('normal', get_statistics(data_normal))
    save_statistics('abnormal', get_statistics(data_abnormal))

    plot_statistics(data_normal, data_abnormal)
