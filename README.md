-------------------------------------------------
General comments
-------------------------------------------------

- There are often three scripts with the same name, but with different extensions (.ipynb, .py, .sh). The .ipynb and .py are the same scripts in jupyter notebook and Python scripts, respectively. The shell script is the script used for sending the Python script onto the PPI queues.
- You will need to modify the shell scripts for writing the files in your own directory (check where the my username is written in the script).

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
