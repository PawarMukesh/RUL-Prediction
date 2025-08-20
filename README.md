# Remaining Useful Life Prediction of Aircraft Turbofan Engine for Predictive Maintenance Using AI Enabled Prognostics

# Problem Statement: 
Airplane engine can fail without giving any warning which is very costly and dangerous for airlines. Sometimes present maintenance techniques are not able to detect engine problem early. Nowadays prediction of remaining useful life of a turbofan engine is necessary to avoid unexpected failures. It is important to design a predictive maintenance system that can analyze engine data and provide immediate meaningful insights that can helps to the maintenance expert to schedule the maintenance early.     

# Need of This Topic: 
1. Passenger safety: Predictive maintenance helps maintenance expert to solve all the engine problem early and avoid failure during running flight. Early prediction of engine life helps airline to fix all the problem and keep passenger as well as staff safe.(Azyus, 2022)
2. Reduce cost: Timely maintenance of aircraft engine support airlines to avoid unwanted expenses.(Mutunga et al., 2019)
3. Reducing delays: Sometime aircraft are grounded because of unexpected maintenance leads to cancellation and delay in flight. Predictive maintenance helps airline in this situation.
4. Extension of engine life: Maintenance expert knows about airplane engine condition early. This important information enables them to great choice about how to use aircraft.

# Objectives: 
1. To preprocess the dataset by extracting the most important settings and sensor measurement for efficient prediction of remaining useful life.
2. Design and development of machine learning and deep learning models to predict the remaining useful life of an aircraft turbofan engine.
3. Evaluation and comparison of state of art machine learning and deep learning algorithms to enhance RUL prediction accuracy and computational speed.
4. Create strong user interface that offers maintenance expert with actionable insights as well as accurate prognostics data on engine health.

# Challenges in existing system: 
1. Data Understanding: Multi-time series data is difficult to understand and extract meaningful patterns from the dataset is complicated.(X. Liu et al., 2023)
2. Feature Selection: Selection of important features from the dataset is difficult because of sensor measurement.(Peng et al., 2022)
3. Model Overfitting: Overfitting of models occurs because of high dimensional data.(Blstm, 2024)
4. Computational Cost: Train machine learning and deep learning model with large number of parameters required maximum computational resources.(Mutunga et al., 2019)

# Inputs overview: 
* Operational Settings: configuration of engine that helps to control engine operation.
* Sensor Readings: Multi time series data from multiple sensors. It includes fan temperature, fan and core speed, flow of fuel and different pressure measurement of components.(Blstm, 2024)

# Expected Output:
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


### **3. 3.	Feature Normalization:**
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
















 


