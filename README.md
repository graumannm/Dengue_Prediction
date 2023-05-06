# DengAI: Predicting Disease Spread Challenge

## Background
This GitHub repository contains the code and resources for the DengAI: Predicting Disease Spread challenge, hosted by DataDriven (https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/). The challenge aimed to address the problem of predicting the number of dengue fever cases in a given region. Dengue fever is a mosquito-borne viral disease that affects millions of people worldwide, especially in tropical and subtropical regions.

Accurate predictions of dengue fever cases can help authorities and healthcare professionals take proactive measures to prevent the spread of the disease, allocate resources efficiently, and plan appropriate responses. Machine learning techniques provide a powerful approach to analyse historical data and make predictions about future outbreaks.

## Objective
The primary objective of the DengAI Challenge was to develop a predictive model that could forecast the number of dengue fever cases in both San Juan, Puerto Rico and Iquitos, Peru. We were provided with historical data, such as weather information, geographical locations, and previous dengue case records, to train their models.

The challenge aimed to foster innovation in the field of ML and contribute to the ongoing efforts to combat dengue fever by providing accurate predictions. By predicting the number of cases in advance, public health officials could take appropriate preventative measures, educate the public, and allocate resources efficiently.

## Approach
To tackle the challenge, we implemented a pipeline-based approach that involved several steps. The following is an overview of our methodology:

Data preprocessing: To ensure high-quality inputs for our ML models, we performed data preprocessing tasks, including data cleaning, handling missing values, and feature engineering.

Feature selection: We carefully selected the most relevant features that could contribute to accurate predictions, considering their correlation with dengue fever cases.

Handling missing values: To counteract gaps in the data, we used interpolation to predict a value based on the next valid observation.  

Normalisation: As part of the data preparation process, we applied the min/max normalisation technique to standardise the features and bring them to a similar scale. This step ensured that no particular feature dominated the learning process.

Adding lag to features: Following research, we added a lag of two weeks to account for the incubation period of humans.

Creating interaction feature: After exploring the data, it became clear that there was a relationship between humidity, temperature and case numbers. We foudn that above 42% humidity and 24 degrees celsius is when mosquitos thrive. Data was then categorised as either meeting this criteria (1), or not (0). 

Model training: We experimented with various classifiers, such as decision trees, random forests, and support vector machines (SVM), to identify the best-performing model. We trained these models using the preprocessed and normalised data.

Model evaluation: We evaluated the performance of each model using appropriate evaluation metrics, such as mean squared error (MSE) or mean absolute error (MAE), to assess their predictive capabilities.

Prediction: Once we selected the best model based on its performance, we used it to make predictions on future case numbers.


## Plots

## Conclusion






