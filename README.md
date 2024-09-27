-------------------------------------------------
General comments
-------------------------------------------------

- There are often three scripts with the same name, but with different extensions (.ipynb, .py, .sh). The .ipynb and .py are the same scripts in jupyter notebook and Python scripts, respectively. The shell script is the script used for sending the Python script onto the PPI queues.
- You will need to modify the shell scripts for writing the files in your own directory (check where my username is written in the script).
- For the footprint CNN, the model version with the best performances with the one called "v13_20_epochs"

-------------------------------------------------
Subdirectory "Normalization"
-------------------------------------------------

- It contains the scripts for creating the normalization and standardization statistics (minimum, maximum, mean, and standard deviation), which are calculated during the training period. 
	- Normalization_statistics.sh creates the standardization statistics in a hdf5 file.
	- Check_hdf_normalization_file.ipynb can be used for checking the standardization statistics.

-------------------------------------------------
Subdirectory "Training data"
-------------------------------------------------

- It contains the scripts for creating the training data.
	- The first script to run is "Create_training_data_domain_grid.sh". This creates training data (netCDF files) on the domain grid.
	- Then, the script "Training_patches.sh" must be run to create training data on patches of 5 x 5 grid points (netCDF files).
	- Then, the scripts "Training_patches_shuffled.sh" and "Validation_patches_shuffled.sh" must be run to concatenate and shuffle the order of the patches. This is to create batch containing data from different seasons in a random order (netCDF files).

-------------------------------------------------
Subdirectory "Train model and make predictions"
-------------------------------------------------

- It contains the scripts used for training the deep learning model, as well as to make predictions.
	- The script "CNN.py" contains the model architecture, it should not be modified unless we want to tune the model.
	- The script "Data_generator.py" contains the data generator to load the data into the neural network during training. It should not be modified unless new variables are added to the model.
	- The script "Train_model_CNN.py" is used for training a new model. The list of predictors, target variables, and classic hyperparameters that can be tuned when developing a model are listed in the "Model parameters" section of the script. 
	- The script "Make_prediction_stride_1_GPU.sh" is used to make predictions using a GPU node.
	- The script "Make_prediction_stride_1_CPU.sh" is used to make predictions using a CPU node.

-------------------------------------------------
Making predictions
-------------------------------------------------

- 1) Run the script "/lustre/storeB/users/cyrilp/CERISE/Scripts_op/Training_data/Create_training_data_domain_grid.sh" with specifying the date of interest.

- 2) Run the script "/lustre/storeB/users/cyrilp/CERISE/Scripts_op/Train_model_and_make_predictions/Make_predictions_stride_1_CPU.sh" or "Make_predictions_stride_1_GPU.sh" with specifying the date of interest.

- 3) If the predictions are created on a CPU nodes, the following lines of codes (lines 18 - 21) must be commented in the script "/lustre/storeB/users/cyrilp/CERISE/Scripts_op/Train_model_and_make_predictions/Make_predictions_stride_1.py":

#print("GPUs available: ", tf.config.list_physical_devices('GPU'))
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#print(physical_devices)
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


