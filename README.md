# Neural_Network_Charity_Analysis

## Overview of the analysis

Using my knowledge of machine learning and neural networks, I use the features in the Alphabet Soup Charity dataset to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

The dataset contains more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

- **EIN** and **NAME**—Identification columns
- **APPLICATION_TYPE**—Alphabet Soup application type
- **AFFILIATION**—Affiliated sector of industry
- **CLASSIFICATION**—Government organization classification
- **USE_CASE**—Use case for funding
- **ORGANIZATION**—Organization type
- **STATUS**—Active status
- **INCOME_AMT**—Income classification
- **SPECIAL_CONSIDERATIONS**—Special consideration for application
- **ASK_AMT**—Funding amount requested
- **IS_SUCCESSFUL**—Was the money used effectively

I provide the following technical analysis deliverables and this written report:

- Deliverable 1: Preprocessing Data for a Neural Network Model
- Deliverable 2: Compile, Train, and Evaluate the Model
- Deliverable 3: Optimize the Model

## Results
In this section, I use bulleted lists and images to support the analysis.

### Data Preprocessing
Using my knowledge of Pandas and the Scikit-Learn’s StandardScaler(), I preprocess the dataset in order to compile, train, and evaluate the neural network model later.

- The variable **IS_SUCCESSFUL** is considered the target for my model.
- The following variables are considered to be the features for my model:
    - 'APPLICATION_TYPE'
    - 'AFFILIATION'
    - 'CLASSIFICATION'
    - 'USE_CASE'
    - 'ORGANIZATION'
    - 'INCOME_AMT'
    - 'SPECIAL_CONSIDERATIONS'
- The variables 'EIN' and 'NAME' are neither targets nor features, and are removed from the input data.
- The variable 'APPLICATION_TYPE' has 17 unique values. I create a density plot for binning and group all application types that occur less than 500 times into a category 'OTHER'.
- The variable 'CLASSIFICATION' has 71 unique values. I create a density plot for binning and group all classification values that occur less than 1800 times into a category 'OTHER'.
- I encode categorical variables using one-hot encoding, and place the variables in a new DataFrame.
- I merge the one-hot encoding DataFrame with the original DataFrame, and drop the originals.
- I split the preprocessed data into features and target arrays.
- I split the preprocessed data into training and testing datasets.
- I standardize numerical variables using Scikit-Learn’s StandardScaler class, then scale the data.

The final dataframe has 44 columns which are the features in my model.

### Compiling, Training, and Evaluating the Model
I use my knowledge of TensorFlow and design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. I consider the number of inputs before determining the number of neurons and layers in your model. I compile, train, and evaluate the binary classification model to calculate the model’s loss and accuracy.

I explain how many neurons, layers, and activation functions I select for my neural network model, and why:
- There are 44 Input Features in my model. These are the columns in my final dataframe after preprocessing the data.
- The model has a feature layer, 2 hidden layers, and an output layer
- There are 80 neurons in hidden layer 1
- There are 30 neurons in hidden layer 2
- I use a sequential Keras model
- The two hidden layers use the ReLu activation function
- The output layer uses the Sigmoid activation function
- The model has 5,981 parameters that are trainable

The following table summarizes the model:
```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_3 (Dense)              (None, 80)                3520      
_________________________________________________________________
dense_4 (Dense)              (None, 30)                2430      
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 31        
=================================================================
Total params: 5,981
Trainable params: 5,981
Non-trainable params: 0
_________________________________________________________________
```

I let the model run for 100 epochs and using this model setup, I was not able to achieve the target model performance of 75%. I achieved an accurracy of 72.5% with a loss of 55.4%.

Next, I try to increase the model performance by taking the following steps:

*Optimization Attempt 1: Remove additional noisy variables*
In this optimization attempt, I remove the columns "SPECIAL_CONSIDERATIONS" and "STATUS" in addition to "EIN" and "NAME". This reduces the total parameters to 5,741 and I let the model run for 100 epochs. I achieve 72.4% accuracy which is lower than the prior model accuracy and still not above the 75% threshold.

*Optimization Attempt 2: Add an additional hidden layer*
In this optimization attempt, I add an additional hidden layer with 15 neurons and a ReLu activation function. The following table summarizes this new model:

```
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_6 (Dense)              (None, 80)                3280      
_________________________________________________________________
dense_7 (Dense)              (None, 30)                2430      
_________________________________________________________________
dense_8 (Dense)              (None, 15)                465       
_________________________________________________________________
dense_9 (Dense)              (None, 1)                 16        
=================================================================
Total params: 6,191
Trainable params: 6,191
Non-trainable params: 0
```

This increases the total parameters to 6,191 and I let the model run for 100 epochs. I achieve 72.6% accuracy which is higher than the prior model accuracy but still not above the 75% threshold.

*Optimization Attempt 3: Change activation function*
In this optimization attempt, I change the activation function in all layers from ReLu to tanh and let the model run for 100 epochs. I achieve 72.5% accuracy which is lower than the prior model accuracy and still not above the 75% threshold.


## Summary
The deep learning model did not reach the target of 75% accuracy even after 3 optimization attempts.

Since we are trying to solve a classification problem, we could use a supervised machine learning model such as the Random Forest Classifier and evaluate its performance against our deep learning model.