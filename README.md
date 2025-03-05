-------------------------------------------------
General comments
-------------------------------------------------

- There are often three scripts with the same name, but with different extensions (.ipynb, .py, .sh). The .ipynb and .py are the same scripts in jupyter notebook and Python scripts, respectively. The shell script is the script used for sending the Python script onto the PPI queues.
- You will need to modify the shell scripts for writing the files in your own directory (check where my username is written in the script).
- For the footprint CNN, the model version with the best performances with the one called "v13_20_epochs"
- For the static GNN, the model version with the best performances is "v6"

-------------------------------------------------
Subdirectory "Normalization"
-------------------------------------------------

- It contains the scripts for creating the normalization and standardization statistics (minimum, maximum, mean, and standard deviation), which are calculated during the training period. 
	- Normalization_statistics.sh creates the standardization statistics in a hdf5 file.
	- Check_hdf_normalization_file.ipynb can be used for checking the standardization statistics.

-------------------------------------------------
Subdirectory "Normalization data"
-------------------------------------------------

- It contains the hdf file with the normalization statistics that are used for training a new model and for making predictions.

-------------------------------------------------
Footprint CNN
-------------------------------------------------
	-------------------------------------------------
	Subdirectory "CNN/Training data/"
	-------------------------------------------------

	- It contains the scripts for creating the training data.
		- The first script to run is "Create_training_data_domain_grid.sh". This creates training data (netCDF files) on the domain grid.
		- Then, the script "Training_patches.sh" must be run to create training data on patches of 5 x 5 grid points (netCDF files).
		- Then, the scripts "Training_patches_shuffled.sh" and "Validation_patches_shuffled.sh" must be run to concatenate and shuffle the order of the patches. This is to create batch containing data from different seasons in a random order (netCDF files).

	-------------------------------------------------
	Subdirectory "CNN/Train_model_and_make_predictions/"
	-------------------------------------------------

	- It contains the scripts used for training the deep learning model, as well as to make predictions.
		- The script "CNN.py" contains the model architecture, it should not be modified unless we want to tune the model.
		- The script "Data_generator.py" contains the data generator to load the data into the neural network during training. It should not be modified unless new variables are added to the model.
		- The script "Train_model_CNN.py" is used for training a new model. The list of predictors, target variables, and classic hyperparameters that can be tuned when developing a model are listed in the "Model parameters" section of the script. 
		- The script "Make_prediction_stride_1_GPU.sh" is used to make predictions using a GPU node.
		- The script "Make_prediction_stride_1_CPU.sh" is used to make predictions using a CPU node.

	-------------------------------------------------
	CNN Making predictions
	-------------------------------------------------

	- 1) Run the script "/lustre/storeB/users/cyrilp/CERISE/Scripts_op/Training_data/Create_training_data_domain_grid.sh" with specifying the date of interest.

	- 2) Run the script "/lustre/storeB/users/cyrilp/CERISE/Scripts_op/Train_model_and_make_predictions/Make_predictions_stride_1_CPU.sh" or "Make_predictions_stride_1_GPU.sh" with specifying the date of interest.

	- 3) If the predictions are created on a CPU nodes, the following lines of codes (lines 18 - 21) must be commented in the script "/lustre/storeB/users/cyrilp/CERISE/Scripts_op/Train_model_and_make_predictions/Make_predictions_stride_1.py":

	#print("GPUs available: ", tf.config.list_physical_devices('GPU'))

	#physical_devices = tf.config.experimental.list_physical_devices('GPU')

	#print(physical_devices)

	#tf.config.experimental.set_memory_growth(physical_devices[0], True)


-------------------------------------------------
GNN static
-------------------------------------------------
	-------------------------------------------------
	Subdirectory "GNN_static/Models/"
	-------------------------------------------------
        - One static graph neural network model is available "GNN_model_18GHz.pth"
        - The details of this model are stored in the file "Model_details_18.7GHz_20250303.txt"
        - The training statitics are stored in the file "Training_statistics_18GHz_20250303.txt"
	-------------------------------------------------
	Subdirectory "GNN_static/Training_data/"
	-------------------------------------------------
        - It contains the scripts for creating the training data and for creating the files to make predictions
        - The first script to run is "Training_data_static.sh". Only this script must be run in order to make predictions.
        - In order to create training data, two additional scripts must be run:
        	- First, the script "Concatenate_graphs_training_datasets_static.sh" must be run to create a big hdf file containing all the samples.
                - Then, the script "Convert_hdf_to_zarr_static.sh" must be run to create the zarr dataset that will be used for training the models.
	-------------------------------------------------
	Subdirectory "GNN_static/Train_model_and_make_predictions/"
	-------------------------------------------------
        - The script "GNN_GAT.py" contains the model architecture, it should not be modified unless we want to tune the model.
        - The script "Data_generator_GNN.py" contains the data generator to load the data into the neural network during training. 
        - The script "Data_generator_GNN_prediction.py" contains the data generator to load the data into the neural network when making predictions. 
        - The script "Train_GNN.sh" is used for training a new model. The list of predictors, target variables, and classic hyperparameters that can be tuned when developing a model are listed in the "Model parameters" section of the script.
        - The script "Long_training_GNN.sh" is used to train further a model that has already been trained in the past.
        - The script "Predictions.sh" is used to make predictions.
