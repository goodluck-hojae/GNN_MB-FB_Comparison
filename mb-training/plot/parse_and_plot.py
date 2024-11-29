import re
import matplotlib.pyplot as plt
import numpy as np


def plot_multiple_experiments_subplot(dataset_list, model_list, time_list, accuracy_list, sampler_list, output_file='multi_experiment_plot.jpg'):
    fig, axs = plt.subplots(3, 4, figsize=(20, 12))  
    fig.subplots_adjust(hspace=0.4, wspace=0.3) 
    line_styles = ['-', '--', '-.']
    markers = ['*', '|', '.']

    subplot_index = 0  # Track which subplot we are on

    for i in range(0, len(dataset_list), 3):
        dataset = dataset_list[i]
        model = model_list[i]

        row = subplot_index // 4 
        col = subplot_index % 4 
        ax = axs[row, col] 

        for j in range(3):
            times = time_list[i + j]
            accuracies = accuracy_list[i + j]
            sampler = sampler_list[i + j]

            cumulative_time = [sum(times[:k+1]) for k in range(len(times))]

            term = 10
            cumulative_time = cumulative_time[::term]
            accuracies = accuracies[::term]

            cumulative_time.insert(0, 0)
            accuracies.insert(0, 0)

            ax.plot(cumulative_time, accuracies, label=f'{sampler}', marker=markers[j], linestyle=line_styles[j])

        ax.set_title(f'{dataset} - {model}', fontsize=14, pad=10)

        max_accuracy = max(max(accuracies) for accuracies in accuracy_list[i:i+3]) * 1.05  # Add a small margin
        ax.set_ylim(0, max_accuracy)
        ax.legend(loc='lower right', fontsize=10, prop={'size': 12})
        subplot_index += 1

    plt.savefig(output_file, format='jpg')
    plt.show()