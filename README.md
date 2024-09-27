-------------------------------------------------
General comments
-------------------------------------------------

- There are often three scripts with the same name, but with different extensions (.ipynb, .py, .sh). The .ipynb and .py are the same scripts in jupyter notebook and Python scripts, respectively. The shell script is the script used for sending the Python script onto the PPI queues.

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


-------------------------------------------------
Subdirectory "Train model and make predictions"
-------------------------------------------------

- It contains the scripts used for training the deep learning model, as well as to make predictions.
