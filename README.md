# Remaining Useful Life Prediction of Aircraft Turbofan Engine for Predictive Maintenance Using AI Enabled Prognostics

# PROBLEM STATEMENT: 
Airplane engine can fail without giving any warning which is very costly and dangerous for airlines. Sometimes present maintenance techniques are not able to detect engine problem early. Nowadays prediction of remaining useful life of a turbofan engine is necessary to avoid unexpected failures. It is important to design a predictive maintenance system that can analyze engine data and provide immediate meaningful insights that can helps to the maintenance expert to schedule the maintenance early.     

# NEED OF THIS TOPIC: 
1. Passenger safety: Predictive maintenance helps maintenance expert to solve all the engine problem early and avoid failure during running flight. Early prediction of engine life helps airline to fix all the problem and keep passenger as well as staff safe.(Azyus, 2022)
2. Reduce cost: Timely maintenance of aircraft engine support airlines to avoid unwanted expenses.(Mutunga et al., 2019)
3. Reducing delays: Sometime aircraft are grounded because of unexpected maintenance leads to cancellation and delay in flight. Predictive maintenance helps airline in this situation.
4. Extension of engine life: Maintenance expert knows about airplane engine condition early. This important information enables them to great choice about how to use aircraft.

# OBJECTIVES: 
1. To preprocess the dataset by extracting the most important settings and sensor measurement for efficient prediction of remaining useful life.
2. Design and development of machine learning and deep learning models to predict the remaining useful life of an aircraft turbofan engine.
3. Evaluation and comparison of state of art machine learning and deep learning algorithms to enhance RUL prediction accuracy and computational speed.
4. Create strong user interface that offers maintenance expert with actionable insights as well as accurate prognostics data on engine health.

# CHALLENGES IN EXISTING SYSTME: 
1. Data Understanding: Multi-time series data is difficult to understand and extract meaningful patterns from the dataset is complicated.(X. Liu et al., 2023)
2. Feature Selection: Selection of important features from the dataset is difficult because of sensor measurement.(Peng et al., 2022)
3. Model Overfitting: Overfitting of models occurs because of high dimensional data.(Blstm, 2024)
4. Computational Cost: Train machine learning and deep learning model with large number of parameters required maximum computational resources.(Mutunga et al., 2019)

# INPUT OVERVIEW: 
* Operational Settings: configuration of engine that helps to control engine operation.
* Sensor Readings: Multi time series data from multiple sensors. It includes fan temperature, fan and core speed, flow of fuel and different pressure measurement of components.(Blstm, 2024)

# EXPECTED OUTPUT:
System produces the two main outputs, Remaining useful life and status of engine health. 
1. Remaining useful life: System predict the numerical value and the unit of RUL is in cycle.
2. Status of engine health: Divide RUL into three important categories.
   * Critical (RUL <= 5.2): Immediate maintenance required.
   * Warning (5.2 < RUL <= 15.7): Schedule maintenance as soon as possible.
   * Healthy (RUL > 15.7): Engine in good condition.

# METHODOLOGY
<img width="940" height="1154" alt="image" src="https://github.com/user-attachments/assets/efed6b8f-5cd7-4b38-bcbc-45fce1246779" />

# DATA COLLECTION:
In this project we utilize NASA CMAPSS (commercial modular aero propulsion system simulation) dataset. It consists different dataset like FD001, FD002, FD003 and FD004 from these set of data we use FD001 because it contains one operating condition and one fault mode.(CMAPSS Jet Engine Simulated Data - Dataset - NASA Open Data Portal, 2025) 
* Training and Testing files are present in all dataset with different operational condition and fault modes are summarize in Table 2.
* Some of features like operational settings, reading of sensors, engine cycle includes in the dataset.

### TARGET VARIABLE CREATION:
To create target feature i.e. rul, first we create a health indicator after that covert health indicator into the remaining useful life (rul).Let see how (Thakkar & Chaoui, 2022) proceed to create target variable, 

1. Health Indicator: Develop non-liner formula, these create curve start with one and decreases to zero that explains how things affect over time.
   
                                                    h(t)=1+d-exp⁡(a.t^b )            

2. Now we calculate alpha value
   
                                                     α=log⁡(1+d)/t_max                                   

 
3. Conversion of Health Indicator into RUL:
To get remaining useful life of an engine at each cycle (Thakkar & Chaoui, 2022)  just multiply health indicator with total number of cycles.

                                                       RUL=h(t)×t_max

# DATA PREPROCESSING:
### **1. Check Missing Values:**
To check missing value present in dataset, we use “isnull()” function. There are no missing value present in this dataset.

### **2. Outlier Handling:** 
Detect outliers using box plot shown in Fig 7. Handle outliers using different types of technique, if the data is normally distributed then we use empirical rule and if data is not normally distributed then we use inter quartile range to impute the outliers.

<img width="823" height="1214" alt="image" src="https://github.com/user-attachments/assets/80be8c8a-7358-4ed7-b0b6-3192493bb843" />


### **3. Feature Normalization:**
Scale features value using minmax scaling because of feature value measured in different units and scales, so we need to transform into the same scale.

# FEATURE ENGINEERING

### **1. Drop Unique and Constant Feature:**
some of the feature not provide meaningful information so we drop those features. 
* Unique feature: engine_id
* Constant features: sensor_1, sensor_3, sensor_5, sensor_10, sensor_16, sensor_18, sensor_19

### **2. Check Correlation:**
Use heatmap to check the relationship between two numerical variables shown in Fig 8. and range of correlation values between -1 to +1, where -1 means negative correlation, +1 means positive correlation and 0 means no relation between numerical variable.

<img width="780" height="654" alt="image" src="https://github.com/user-attachments/assets/3abc142a-0add-4695-b7cf-1570250c3138" />

### **3. Check Duplicates:**
There is no duplicate value present in the dataset, to check duplicates values we use pandas “duplicated()”function. 

### **4. Dimensionality Reduction:**
In this step we reduce the dimension of the dataset using principal component analysis technique. 

<img width="653" height="445" alt="image" src="https://github.com/user-attachments/assets/3bc86016-e7c2-464e-8a37-056c2c9e265a" />

## TRAIN-TEST-SPLIT: 
Load preprocess data and define independent and target feature. Split data into training and testing using “train_test_split” function.
   * 80 percent data used for training the models.
   * 20 percent data used for testing the models.

### Data Preparation for Deep Learning and Hybrid Model:
Long short-term memory model require data in three-dimensional format for that we convert tabular data into sequential form this help models to learn relationship, to convert data into sequential form for that we use sliding window method.
   * Shape of train set: (16504, 5)
   * Shape of test set: (4127, 5)

### Window size: we select window size 30 for lstm model means our model see previous 30 data points to make prediction for the next one.(Muneer et al., 2021)
   * Train sequences for lstm: (16474, 30, 5)
   * Test sequences for lstm: (4157, 30, 5)
     
For hybrid model we select window size 1 because we only extract feature using lstm model and feed extracted feature to other different model to make prediction. 
   * Train sequences for hybrid model: (16474, 1, 5)
   * Test sequences for hybrid model: (4157, 1, 5)


# MODEL TRAINING PHASE

## 1. Random Forest Regressor 

Create random forest regression model using random search hyperparameter tunning technique to get correct prediction for that we test 20 different parameter combination using Randomized Search cv with 5-fold validation to identify the precise model. 

<img width="939" height="372" alt="image" src="https://github.com/user-attachments/assets/f612a19e-2a47-4183-99ca-d479a96f2ba2" />


## 2. Extreme Gradient Boosting  

Build xgboost model with hyperparameter tunning methods that help to get a best possible parameter. Create dictionary with nine different hyperparameter and use randomized search cv to test 20 different combination along with 3-fold cross validation.(Bhat Shrinath, 2020)  


<img width="1070" height="595" alt="image" src="https://github.com/user-attachments/assets/dff3be2b-ad13-4d21-9e33-9f274b2cb0e8" />

## 3. Stacking Regressor
Combine multiple regressor models to enhance the model performance and get correct predictions, In this technique multiple base learners are trained and use base learner predictions as an input for the main model shown in Fig 10.(Brownlee Jason, 2020) Three boosting and one simple model are used as a base-learner. Let see which models we use for the base learner, 


### 1.	Extreme gradient boost 
Build Xgboost regressor as a first base learner with 500 trees with 5 maximum depth and set 0.01 learning rate.  

### 2.	Random forest 
Utilize random forest regression model as a second base learner with different parameters like number of estimator and maximum depth of tree. Here we use 200 trees along with 10 levels.  

### 3.	Gradient boosting 
Create gradient boosting regressor model as a third base learner with 100 trees and 0.1 learning rate. 

### 4.	Support vector machine
Build support vector regressor model along with radial basis function kernel and use regularization parameters c with value 10 for the minimal error.(Matuszczak et al., 2021) 


### Meta Model (Multi-layer perceptron)
After creating all base learner we define meta multilayer perceptron model with two hidden layers, first layer with 50 neurons and second layer with 25 neurons and train this model up to 500 times to identify the best possible combination of predictions of all base learner.  In final step we develop stacking regressor and call above created base learners as well as multi-layer perceptron model and use parameters like pass-through with true this actually allows mlp model to get prediction from above created base learners and initial variables of dataset. 

<img width="836" height="352" alt="image" src="https://github.com/user-attachments/assets/609cfb1a-be33-4f8a-a0e4-fb38dd4f1995" />

## 4. Long short-term memory

* Utilize first lstm layer with 128 neurons shown in Fig 11. along with tanh activation function (Sharma et al., 2020) and pass sequences to the next layer using return sequences parameter, also pass the inputs with shape (30, 5) where 30 is a time step and 5 is total feature.
* In next layer, we use dropout with value 0.5 to suppress the overfitting problem of model.
* Use second lstm layer with 64 neurons and apply tanh activation (Sharma et al., 2020) after that pass all sequences to the next layer for the processing and use dropout layer with value 0.5.
* In third lstm layer, use 32 neurons along with tanh activation function (Sharma et al., 2020) and return output from last time steps.
* Utilize another dropout layer with value 0.3 and use output layer to with one neuron. 

<img width="1093" height="451" alt="image" src="https://github.com/user-attachments/assets/77c80630-8b49-4c37-b5b3-3ec05d06f116" />

## 5. Hybrid model (LSTM + Light-GBM) 
### Extract features using long short-term memory:
In first stage we extract features using long short-term memory to capture the different patterns from the data (Hong et al., 2020) for that we use different types of layers. 
1. Take input with shape (1,5) where 1 is a window size and total features is 5.
2. First LSTM layer contains 64 neuron and pass input sequences to the next layer also we use 0.2 dropout to suppress overfitting.
3. In second LSTM layer we use 32 neurons along with dropout value 0.2.
4. Use dense layer with 32 neurons along with relu activation function (Sharma et al., 2020) to process the features.

Above model process the sequences and convert it into the meaningful information that represent the pattern and trends present in the data. 

### Flatten create features:
In second stage, we reshape the generated features to convert it into the 2D format because of we use different machine learning model for the prediction. 


### Light-GBM
Light-GBM stand for light gradient boosting machine, it creates multiple decision tree and learning done by mistake made by the previous tree. After that combine all tree to make a prediction. It is best option to work with large dataset and it uses less memory.(Jafari & Byun, 2023) In third stage we use light-gbm regressor to make a prediction on extracted features, for that we follow the below steps, 
1. Create light-gbm regressor object with 50 estimators and maximum depth of five(Jafari & Byun, 2023) and Train model using converted 2D features.
2. After training the model, we make predictions using “predict()” function on both sets of data to see the performance.

<img width="788" height="373" alt="image" src="https://github.com/user-attachments/assets/ffd8b0be-a193-4971-8718-4ee2f20647f3" />

# RESULT

Evaluate model performance using training and testing data to determine the overfitting and underfitting of models, for that we use different types of metrics. 

<img width="1025" height="403" alt="image" src="https://github.com/user-attachments/assets/079b5a98-a250-4eda-90b5-6bb93a4eb942" />

After evaluating all models, stacking regressor and XGBoost gives the best performance on training and testing data with minimum overfitting and powerful generalization. 

<img width="896" height="426" alt="image" src="https://github.com/user-attachments/assets/e0bd7fda-13f0-4d7d-ad92-358e60cbd18d" />

The LSTM+LightGBM and random forest model show decent performance but these models were slightly overfit. LSTM model has low score but it maintains the stability on training and testing set.  

<img width="1078" height="435" alt="image" src="https://github.com/user-attachments/assets/f00e4510-5f76-4125-abd2-6f87a1b189a2" />

we can clearly see that stacking regressor and XGBoost give minimum error values as compare to other models. Neural network base model gave the highest prediction error it suggests that the ensemble approach is well suited for RUL prediction. 

<img width="897" height="559" alt="image" src="https://github.com/user-attachments/assets/569484cc-dc50-4450-99c8-8ccb81c11413" />

Random forest model prediction error values better than the long short-term memory and hybrid model. 

# RUL PREDICTION USING STACKING REGRESSOR AND HYBRID MODEL 

<img width="961" height="540" alt="image" src="https://github.com/user-attachments/assets/687f1e52-2f86-4d39-ac74-b4f23b7c7a48" />

<img width="965" height="529" alt="image" src="https://github.com/user-attachments/assets/97b9d86b-2fd1-48b1-b4bf-558d41114c69" />

<img width="959" height="569" alt="image" src="https://github.com/user-attachments/assets/ac89853b-df50-43a1-bef0-993c97b295d0" />

# DISCUSSION

Predicting the remaining useful life of turbofan engine is important to avoid unnecessary accident and enhance passenger safety. For that, we utilize different machine learning and deep learning models as well as we build our own custom model with combination on neural network and machine learning to enhance the performance. Before training models, we preprocess data using different types of methods including outlier handling, normalization of feature values and perform feature engineering to select the important features from the dataset. 

The stacking regressor delivered best performance on both sets of data with 75.28% training r2 score and 72.31% testing r2 score with 13.08 mean squared error and 2.530 mean absolute error. It shows that the model is generalize well on both sets of data. Other models like random forest, xgboost and hybrid (lstm+lightgbm) gave the decent performance with minimal overfitting. Long short-term memory model very well generalize meaning gap between training and testing r2 score is very less but it gave the highest prediction error. 

Performance of stacking regressor and hybrid model was tested using different combination of real input values. It is necessary to check the model generalization on real input values because of model performance varies some time. There are several reasons for variation model performance like data variability, quality of data etc.  These models have several limitation and drawbacks, because of these models train on specific set of data that can contain specific engine operating conditions and fault modes as well as need lots of computational power to train all sets of data.

We can implement this system into the air and space industry to schedule the engine maintenance early and avoid accidents. This system helps airline to extend the life of aircraft turbofan engine and enhance the safety of passenger. 


# CONCLUSION
In conclusion, to predict the remaining useful life of turbofan engines for that we used different approaches and methods. Firstly, we preprocess data and select important features and trained our model using those features. we utilize tree based and neural network models. Stacking regressor and xgboost model delivered best performance with minimum prediction error and the gap between training and testing score is very less meaning the model generalize well on both sets of data. The r2 scores of both models are 72.32% and 72.12% with mean squared error 13.08 and 13.16. 

The performance of the Hybrid and random forest models was decent but they showed some overfitting. In other side long short-term memory was very well generalized but the error is high and r2 score is less. In predictive maintenance tree-based models perform really well compared to the other neural network models. 

Testing model performance on real world data these actually help us understand model generalization and we can implement this application on aerospace and space industry to schedule early maintenance of engine. This project has some limitations we cannot predict remaining useful life of those engines that operate under multiple operating conditions and fault modes because these models trained on specific engine conditions. In general, our project shows that the ensemble learning is really useful to predict the remaining useful life of turbofan engine. 

# REFERENCES
* Abdi, H., & Williams, L. J. (2010). Principal component analysis. wiley interdisciplinary reviews: computational statistics. Wiley Interdisplinary Reviews: Computational Statistics, 1–47.
* Azyus, A. F. (2022). Determining the Method of Predictive Maintenance for Aircraft Engine Using Machine Learning. Journal of Computer Science and Technology Studies, 4(1), 01–06. https://doi.org/10.32996/jcsts.2022.4.1.1
* Bhat Shrinath. (2020). Random Forest Regression & Hyperparameters. Medium. https://medium.com/@bhatshrinath41/a-comprehensive-guide-to-random-forest-regression-43da559342bf
* Blstm, S. M. (2024). Turbofan Engine Remaining Useful Life ( RUL ) Prediction Based on Bi-Directional Long.
* Brownlee Jason. (2020). Stacking Ensemble Machine Learning With Python. Machine Learning Mastery. https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/
* CMAPSS Jet Engine Simulated Data - Dataset - NASA Open Data Portal. (2025). PCoE. https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
* Muneer, A., Taib, S. M., Naseer, S., Ali, R. F., & Aziz, I. A. (2021). Data-driven deep learning-based attention mechanism for remaining useful life prediction: Case study application to turbofan engine analysis. Electronics (Switzerland), 10(20). https://doi.org/10.3390/electronics10202453
* Mutunga, J. M., Kimotho, D. J., & Muchiri, P. P. (2019). Estimating the Remaining Useful Lifetime of a Turbofan Engine using Ensemble of Machine Learning Algorithms.
* Peng, C., Chen, Y., Gui, W., Tang, Z., & Li, C. (2022). Remaining useful life prognosis of turbofan engines based on deep feature extraction and fusion. Scientific Reports, 12(1), 1–14. https://doi.org/10.1038/s41598-022-10191-2
* Sharma, S., Sharma, S., & Athaiya, A. (2020). Activation Functions in Neural Networks. International Journal of Engineering Applied Sciences and Technology, 04(12), 310–316. https://doi.org/10.33564/ijeast.2020.v04i12.054




 


