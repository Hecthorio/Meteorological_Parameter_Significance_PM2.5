# Meteorological_Parameter_Significance_PM2.5
These codes were developed to analyze the impact of meteorological parameters and the previous states of the system on the prediction of PM2.5 using neural networks.

## Contents
This repository contains *5 scripts* and *2 folders*. Below is the description of the functionality of each file:

### Folders:
* **base_datos (database)**: contains databases for all meteorological parameters and PM2.5 concentration used for model training (prior to merging and filtering). Inside this folder, there are subfolders with information for each meteorological parameter, the final database, and data scaling information.
* **modelos (models)**: this folder contains the models obtained from the training.

### Scripts:
* **lectura_filtrado (reading_filtering)**: This script reads files with meteorological and pollutant parameter information and compiles them into a single file, then combines them all into a single dataframe (please note, modify the default file paths in the script with your own).
* **analsis_exploratorio (exploratory_analysis)**: This script generates box and whisker plots for PM2.5 to assess possible outliers, evaluates correlation coefficients between all parameters used in the models, and performs auto-correlation and partial auto-correlation analysis for PM2.5 to determine which previous states have the highest correlation with the current state of the system.
* **entrenamiento (training)**: In this script, neural network models are trained using the Keras library and compared based on various metrics. It also generates residual analysis for the models using regression model criteria. This script also creates the data normalization file.
* **sensibilidad (sensitivity)**: This script applies the Garson algorithm and the method of partial derivatives to measure the sensitivity level of each parameter used in the models.
* **topologia (topology)**: This script trains multiple models by making modifications to the number of neurons and layers in the model and evaluates the performance of each model based on the previously used metrics. The final results are presented in contour plots, and a significance analysis is conducted to verify if there is a significant difference between the base architecture of the study and the best-found architecture.

## Notes
* Data in the base_datos (database) folder was extracted from SINAICA (https://sinaica.inecc.gob.mx/) from the state of Aguascalientes, from the "Centro" monitoring station, from 2018 to 2019.
* The Python IDE used to develop the code was Spyder 5.3.3.
* If you wish to use the code, you need to modify the file paths where the databases are located.
* Python version 3.8.8.


