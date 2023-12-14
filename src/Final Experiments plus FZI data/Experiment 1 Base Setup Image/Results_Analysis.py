from glob import glob 
import os 
import pickle
import numpy as np

# Get the absolute path of the script
script_directory = os.path.dirname(os.path.abspath(__file__))

# File path for the results text file
results_file_path = os.path.join(script_directory, "metrics_results.txt")

# Construct the paths of pickle files
pickle_files = glob(os.path.join(script_directory, "models", "*.pkl"))
print(pickle_files)

metric_key_list = []
for pickle_file in pickle_files:
    if os.path.exists(pickle_file):
        print(f"The file {pickle_file} exists.")
        with open(pickle_file, 'rb') as file:
            metric_dictionary = pickle.load(file)
        for metric_name, metric_instance in metric_dictionary.items():
            metric_value = metric_instance.compute()
            metric_key_list.append(metric_name)
    else:
        print(f"The file {pickle_file} does not exist.")

metrics_of_concern = ['MulticlassSpecificity', 'MulticlassF1Score', 'MulticlassCalibrationError', 'MulticlassAUROC', 'MulticlassAccuracy', 'MulticlassAveragePrecision', 'MulticlassMatthewsCorrCoef', 'MulticlassPrecision', 'MulticlassRecall']

new_metrics_dict = {metric: [] for metric in metrics_of_concern}

for pickle_file in pickle_files:
    with open(pickle_file, 'rb') as file:
        metric_dictionary = pickle.load(file)
    for metric_name in metrics_of_concern:
        metric_instance = metric_dictionary[metric_name]
        metric_value = metric_instance.compute()
        
        new_metrics_dict[metric_name].append(metric_value.item())

# Writing results to a text file
with open(results_file_path, 'w') as results_file:
    for metric_name in metrics_of_concern:
        averaged_value = np.mean(new_metrics_dict[metric_name])
        results_file.write(f"Final Averaged {metric_name}: {averaged_value}\n")
        print(f"Final Averaged {metric_name}: {averaged_value}")

print(f"Results written to {results_file_path}")
