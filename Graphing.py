import csv
import os
import pandas as pd
import matplotlib.pyplot as plt

def append_results_to_csv(model_name, results, csv_file_path):
    """
    Append evaluation results of a model to a CSV file.
    
    :param model_name: Name of the model.
    :param results: A dictionary of timesteps and their corresponding RMSE values.
    :param csv_file_path: Path to the CSV file.
    """
    file_exists = os.path.isfile(csv_file_path)
    
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write headers if the file is new
        if not file_exists:
            writer.writerow(['Model Name', 'Timestep', 'RMSE'])
        # Append the data rows
        for timestep, rmse in results.items():
            writer.writerow([model_name, timestep, rmse])
            

def plotGraphs(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Find the unique timesteps across all models
    unique_timesteps = df['Timestep'].unique()

    # Plotting
    plt.figure(figsize=(10, 6))
    for model_name in df['Model Name'].unique():
        model_data = df[df['Model Name'] == model_name]
        plt.plot(model_data['Timestep'], model_data['RMSE'], label=model_name, marker='o') # Added a marker for clarity

    # Set the positions and labels for the x-axis ticks to show every 50 units
    step_size = 5  # The step size of your timesteps, assuming it's consistent
    interval = 50  # The interval at which you want to show labels
    # Create a list of labels to show based on the interval
    labels_to_show = [label for label in unique_timesteps if int(label[1:]) % interval == 0]

    # Set the custom ticks and labels
    plt.xticks(labels_to_show, labels_to_show) # The ticks and labels are now based on the unique timesteps
    plt.xlabel('Timestep')
    plt.ylabel('RMSE')
    plt.title('RMSE Over Time for Different Models')
    plt.xticks(rotation=45) # Improve readability of the timestep labels
    plt.legend()
    plt.tight_layout() # Adjust layout to make room for the rotated x-axis labels
    plt.show()

#plotGraphs('eval_results_sig5.csv')