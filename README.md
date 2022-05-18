# Neural_Network_Charity_Analysis
Machine learning and neural networks
# Neural Network Charity Analysis

## Analysis Overview
Beks has come a long way since her first day at that boot camp five years ago—and since earlier this week, when she started learning about neural networks! Now, she is finally ready to put her skills to work to help the foundation predict where to make investments.

With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to help Beks create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, Beks received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:
•	EIN and NAME—Identification columns
•	APPLICATION_TYPE—Alphabet Soup application type
•	AFFILIATION—Affiliated sector of industry
•	CLASSIFICATION—Government organization classification
•	USE_CASE—Use case for funding
•	ORGANIZATION—Organization type
•	STATUS—Active status
•	INCOME_AMT—Income classification
•	SPECIAL_CONSIDERATIONS—Special consideration for application
•	ASK_AMT—Funding amount requested
•	IS_SUCCESSFUL—Was the money used effectively


## Resources
- Data Source: charity_data.csv
- Software: Python 
            Anaconda Navigator 
            Jupyter Notebook

## Results

### Preprocessing Data for a Neural Network Model

Pandas and the Scikit-Learn’s StandardScaler(), to preprocess the dataset in order to compile, train, and evaluate the neural network model
- Using the information we have provided in the starter code, follow the instructions to complete the preprocessing steps.

•	Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
•	What variable(s) are considered the target(s) for your model?
•	What variable(s) are considered the feature(s) for your model?
•	Drop the EIN and NAME columns.
•	Determine the number of unique values for each column.
•	For those columns that have more than 10 unique values, determine the number of data points for each unique value.
•	Create a density plot to determine the distribution of the column values.
•	Use the density plot to create a cutoff point to bin "rare" categorical variables together in a new column, Other, and then check if the binning was successful.
•	Generate a list of categorical variables.
•	Encode categorical variables using one-hot encoding, and place the variables in a new DataFrame.
•	Merge the one-hot encoding DataFrame with the original DataFrame, and drop the originals.
•	Split the preprocessed data into features and target arrays.
•	Split the preprocessed data into training and testing datasets.
•	Standardize numerical variables using Scikit-Learn’s StandardScaler class, then scale the data.



### Compile, Train, and Evaluate the Model
Using TensorFlow, design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. Included is how many inputs there are before determining the number of neurons and layers in the model. Finally compile, train, and evaluate the binary classification model to calculate the model’s loss and accuracy.

•	Continue using the AlphabetSoupCharity.ipynb file where you’ve already performed the preprocessing steps from Deliverable 1.
•	Create a neural network model by assigning the number of input features and nodes for each layer using Tensorflow Keras.
•	Create the first hidden layer and choose an appropriate activation function.
•	If necessary, add a second hidden layer with an appropriate activation function.
•	Create an output layer with an appropriate activation function.
•	Check the structure of the model.
•	Compile and train the model.
•	Create a callback that saves the model's weights every 5 epochs.
•	Evaluate the model using the test data to determine the loss and accuracy.
•	Save and export your results to an HDF5 file, and name it AlphabetSoupCharity.


## Optimize the Model
Using TensorFlow, optimize the model in order to achieve a target predictive accuracy higher than 75%.
Optimize your model in order to achieve a target predictive accuracy higher than 75% by using any or all of the following:

•	Adjusting the input data to ensure that there are no variables or outliers that are causing confusion in the model, such as:
•	Dropping more or fewer columns.
•	Creating more bins for rare occurrences in columns.
•	Increasing or decreasing the number of values for each bin.
•	Adding more neurons to a hidden layer.
•	Adding more hidden layers.
•	Using different activation functions for the hidden layers.
•	Adding or reducing the number of epochs to the training regimen.


•	Create a new Jupyter Notebook file and name it AlphabetSoupCharity_Optimzation.ipynb.
•	Import your dependencies, and read in the charity_data.csv to a Pandas DataFrame.
•	Preprocess the dataset like you did in Deliverable 1, taking into account any modifications to optimize the model.
•	Design a neural network model, taking into account any modifications that will optimize the model to achieve higher than 75% accuracy.
•	Create a callback that saves the model's weights every 5 epochs.
•	Save and export your results to an HDF5 file, and name it AlphabetSoupCharity_Optimization.h5.



## Summary
In this analysis the data preprocessing target is included in the IS_SUCCESSFUL column
However we dropped the following cloumnsin the represented model:
ORGANIZATION,STATUS,INCOME_AMT,SPECIAL_CONSIDERATIONS,ASK_AMT,APPLICATION_TYPE,AFFILIATION,CLASSIFICATION,USE_CASE.
We also needed to remove EIN and NAME.

### Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.

nodes_hidden_layer1 = 100
nodes_hidden_layer2 = 30

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 100)               5000      
_________________________________________________________________
dense_1 (Dense)              (None, 30)                3030      
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 31        
=================================================================
Total params: 8,061
Trainable params: 8,061
Non-trainable params: 0

The accuracy was 0.7310 and we made many attempts to improve it by:-
Dereasing the number of hidden nodes in layer 1 
Increasing the number of hidden layers upto 3
We also changed the activation functions
SUMMARY
the data preprocessing analysis shows a recommendation as to how various models can solve the final outcome.