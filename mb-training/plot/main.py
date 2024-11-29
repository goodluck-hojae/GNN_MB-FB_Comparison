import parse_and_plot as plot 
import os
from parse import parse_log
import numpy as np

results = dict()
 
folder_name = 'tta_result'
for filename in os.listdir(folder_name):
    log_text = open(folder_name + '/' + filename, 'r').read()
     
    sampler = filename.split('_a100')[0]  
    exps = log_text.split("Running parameter")
    
    for idx, exp in enumerate(exps):
        result = parse_log(exp, sampler)  
        
        if result:
            dataset, model, times, accuracies, sampler = result  
            if model == 'NSGAT':
                model = 'GAT'
            elif model == 'NSGCN':
                model = 'GCN'
            elif model == 'NSGraphSAGE':
                model = 'GraphSAGE'
            dataset = dataset.replace('_', '-')
            sampler = sampler.replace('cluster_gcn', 'Cluster GCN')
            sampler = sampler.replace('saint_sampler', 'Saint Sampler')
            sampler = sampler.replace('node_sampling', 'Neighbor Sampler')
            dataset = dataset[0].upper() + dataset[1:] 
            key = f"{model}-{dataset}" 
            if key not in results:
                results[key] = {'dataset': dataset, 'model': model, 'sampler_data': {}}
             
            if sampler not in results[key]['sampler_data']:
                results[key]['sampler_data'][sampler] = {'times': [], 'accuracies': []}
 
            results[key]['sampler_data'][sampler]['times'].extend(times)
            results[key]['sampler_data'][sampler]['accuracies'].extend(accuracies) 

all_dataset_list, all_model_list, all_time_list, all_accuracy_list, all_sampler_list = [], [], [], [], []


for idx, (key, data) in enumerate(results.items()):
    dataset_list, model_list, time_list, accuracy_list, sampler_list = [], [], [], [], []
 
    dataset = data['dataset']
    model = data['model']
 
    for sampler, sampler_data in data['sampler_data'].items(): 
        dataset_list.append(dataset)
        model_list.append(model)
 
        if model == 'GAT' and 'products' in dataset and 'cluster' in sampler.lower():
            sampler_data['times'] = sampler_data['times'][:-50]
            sampler_data['accuracies'] = sampler_data['accuracies'][:-50]

        time_list.append(sampler_data['times'][2:])
        accuracy_list.append(sampler_data['accuracies'][2:])
        sampler_list.append(sampler)
  
    if 'papers100M' not in key:
        all_dataset_list.extend(dataset_list)
        all_model_list.extend(model_list)
        all_time_list.extend(time_list) 
        all_accuracy_list.extend(accuracy_list)
        all_sampler_list.extend(sampler_list)
 
plot.plot_multiple_experiments_subplot(all_dataset_list, all_model_list, all_time_list, all_accuracy_list, all_sampler_list, folder_name + '.jpg')
  