Before running the code, run the following to install the neccesary dependencies:
pip install -r requirements.txt

Modules Description
1. Data Loading and Processing
loadData.py
Loads temperature distribution datasets from specified file paths.
Prepares data for training by normalizing and partitioning into training and test sets.
continuousDataLoaders.py
Provides functionality to create data loaders that continuously feed data into the model during training sessions.

2. Model Training
training_script.py (Adapted from Neural Operator)
Initializes and trains a model using configurations from the Neural Operator library.
Implements a recursive training strategy where the model's outputs are used as inputs in subsequent training steps.
Custom evaluation functions are integrated to assess model performance using the recursive outputs.

3. Model Evaluation
evaluation_script.py
Evaluates the trained model on a full dataset, calculating the root mean square error (RMSE) at each timestep.
Records and visualizes the errors and discrepancies between the model predictions and the actual data.

4. Visualization Tools
dataVisualization.py
Contains various functions for visualizing temperature distributions, both statically and dynamically.
Functions include generating heatmaps, animated visualizations, and comparison plots between predicted and actual data values.

TO TRAIN A MODEL:
1.Specify datasets going to be used for training in mainMultiple.py
  -Adjust the input length, prediction length & number of epochs as needed
  -NOTE: This project only utilized input length of 1

2. callTrain.py has adjustable hyper parameters:
  -learning rate scheduler & optimizers
  -Input resolution
  -WandB initizlization

3. trainer.py is responsible for training.

To evaluate a model:
-Run fullDatasetEval.py
-Load the model from model path (there are trained models in the /Models folder
-Specify the input parameters k,w,sig
This will evaluate the model, give the mean error of all the predictions, generate and save a video of the predictions.
