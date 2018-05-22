# Introduction

This project provides several examples of common machine learning models applied to financial market predictions using TensorFlow, Keras, and Sci-kit Learn. All these models use the past 500 days of data for a given forex pairs, with a number of technical indicators added to the DataFrame. All of these models are standard, i.e. there hasn't been a major effort to optimize them and they could all be improved up in one way or another.

Ideally, this project would help make these tools more accessible for those learning to apply machine learning to financial markets.

*TODO: showcase quant functions and extend database usage examples.*

## Ensemble Research

Using a basic list of six standard Sci-kit Learn ensemble methods, we can explore the effectiveness of these off-the-shelf models. Out of the box, these models can achieve 65-75% accuracy but could be improved by manipulating learning rates, increasing the number of estimators, or for some models, including a base estimator; i.e. another predictive model that improves the Booster's reliability.

*AdaBoostRegressor*

![AdaBoost png](/graphs/model1.png)

*BaggingRegressor*

![BaggingRegressor png](/graphs/model2.png)


## LSTM with Multiple Inputs 

A basic LSTM model built with Keras and using 20 input factors: open, high, low, volume, and technical indicators such as moving averages, a Stochastic Oscillator, Bollinger Bands, and others. The results are noisy and mixed, as it is unclear if more data helps or hurts the model from building meaningful connections between the data.


![LSTMulti png](/graphs/LSTMulti_Score_69.38469925904886.png)


This model could be improved in a number of ways. Increasing the number of layers, normalizing the data, reducing the amount of input data, and increasing the amount of learning data are the most obvious choices.

## Linear Models

Linear regression models are natural candidates for time series analysis. Using the standard Sci-kit learn Ridge and Linear Regression models, we can achieve roughly 80% accuracy on a single currency pair before manipulating any of the parameters.

*Ridge*

![Ridge png](/graphs/Ridge_0.8099881178871757.png)


*LinearRegression*

![LinearRegression png](/graphs/LinearRegression_0.8099495670315746.png)


The Autoregressive Integrated Moving Average (ARIMA) is not exactly a machine learning algorithm, but a linear model used in econometric analysis that can be applied to financial markets to make predictions. A quick build of this model can also produce 78% accurate predictions on the test data.

*ARIMA*

![ARIMA png](/graphs/ARIMA_0.7895470127761381.png)



## Tensorflow NN

This is a a FOREX adaptation of Sebastian Heinz's neural network for stocks from his Medium.com article ["A simple deep learning model for stock prediction using TensoFlow"](https://github.com/sebastianheinz/stockprediction). 

![TF NN Gif](/gifs/tf_nn_research.gif)

Running on the daily AUD/JPY chart, the model reaches 95% accuracy in less than 10 epochs.

