# Remaining Useful Life Prediction of Aircraft Turbofan Engine for Predictive Maintenance Using AI Enabled Prognostics

## INTRODUCTION
Early detection of aircraft turbofan engine failures is crucial to avoid unnecessary accident.(Mutunga et al., 2019) It also helps maintenance experts to schedule the maintenance more effectively.(Mutunga et al., 2019) If an airline able to predict engine problem early then they keep passenger safe, reduce the maintenance cost and save money.(Peng et al., 2022) 
Airplane engines are very high-power machines they operate in extreme condition. Engine temperature and pressure are very high while flying. Over period of time these factor causes engine health.(H. Liu et al., 2021) If this issue not found early then the airline leads to the engine failure.(Taha et al., 2019) Old methods of maintenance depend in fixed schedule. This approach is not safe and lead to many problems. Recently, Air India Boeing 787-8 airplane crashed in Ahmedabad after take-off. Out of 242 passengers, 241 were dead. Initial investigation suggests that both engines are failed. This incident highlights the importance of predictive maintenance to avoid accidents.(Sumit Khanna, 2025)

## Problem Statement: 
Airplane engine can fail without giving any warning which is very costly and dangerous for airlines. Sometimes present maintenance techniques are not able to detect engine problem early. Nowadays prediction of remaining useful life of a turbofan engine is necessary to avoid unexpected failures. It is important to design a predictive maintenance system that can analyze engine data and provide immediate meaningful insights that can helps to the maintenance expert to schedule the maintenance early.     

## Need of This Topic: 
1. Passenger safety: Predictive maintenance helps maintenance expert to solve all the engine problem early and avoid failure during running flight. Early prediction of engine life helps airline to fix all the problem and keep passenger as well as staff safe.(Azyus, 2022)
2. Reduce cost: Timely maintenance of aircraft engine support airlines to avoid unwanted expenses.(Mutunga et al., 2019)
3. Reducing delays: Sometime aircraft are grounded because of unexpected maintenance leads to cancellation and delay in flight. Predictive maintenance helps airline in this situation.
4. Extension of engine life: Maintenance expert knows about airplane engine condition early. This important information enables them to great choice about how to use aircraft.

## Objectives: 
1. To preprocess the dataset by extracting the most important settings and sensor measurement for efficient prediction of remaining useful life.
2. Design and development of machine learning and deep learning models to predict the remaining useful life of an aircraft turbofan engine.
3. Evaluation and comparison of state of art machine learning and deep learning algorithms to enhance RUL prediction accuracy and computational speed.
4. Create strong user interface that offers maintenance expert with actionable insights as well as accurate prognostics data on engine health.

## Challenges in existing system: 
1. Data Understanding: Multi-time series data is difficult to understand and extract meaningful patterns from the dataset is complicated.(X. Liu et al., 2023)
2. Feature Selection: Selection of important features from the dataset is difficult because of sensor measurement.(Peng et al., 2022)
3. Model Overfitting: Overfitting of models occurs because of high dimensional data.(Blstm, 2024)
4. Computational Cost: Train machine learning and deep learning model with large number of parameters required maximum computational resources.(Mutunga et al., 2019)

## Inputs overview: 
* Operational Settings: configuration of engine that helps to control engine operation.
* Sensor Readings: Multi time series data from multiple sensors. It includes fan temperature, fan and core speed, flow of fuel and different pressure measurement of components.(Blstm, 2024)

## Expected Output:
System produces the two main outputs, Remaining useful life and status of engine health. 
1. Remaining useful life: System predict the numerical value and the unit of RUL is in cycle.
2. Status of engine health: Divide RUL into three important categories.
   * Critical (RUL <= 5.2): Immediate maintenance required.
   * Warning (5.2 < RUL <= 15.7): Schedule maintenance as soon as possible.
   * Healthy (RUL > 15.7): Engine in good condition.

## METHODOLOGY
<img width="940" height="1154" alt="image" src="https://github.com/user-attachments/assets/efed6b8f-5cd7-4b38-bcbc-45fce1246779" />

### DATA COLLECTION:
In this project we utilize NASA CMAPSS (commercial modular aero propulsion system simulation) dataset. It consists different dataset like FD001, FD002, FD003 and FD004 from these set of data we use FD001 because it contains one operating condition and one fault mode.(CMAPSS Jet Engine Simulated Data - Dataset - NASA Open Data Portal, 2025) 
* Training and Testing files are present in all dataset with different operational condition and fault modes are summarize in Table 2.
* Some of features like operational settings, reading of sensors, engine cycle includes in the dataset.

### TARGET VARIABLE CREATION:
To create target feature i.e. rul, first we create a health indicator after that covert health indicator into the remaining useful life (rul).Let see how (Thakkar & Chaoui, 2022) proceed to create target variable, 
Health Indicator: Develop non-liner formula, these create curve start with one and decreases to zero that explains how things affect over time. 
                                          h(t)=1+d-exp⁡(a.t^b )               
where,
	h(t): health indicator ranges from 1 to 0
	d:  is an initial degradation (constant number)
	t: present cycle number
	a and b: coefficients help to control the rate of degradation
Now we calculate alpha value, 
                                                       α=log⁡(1+d)/t_max                                     

where,
	t max: engine failure before maximum cycle 
 
**Conversion of Health Indicator into RUL:**
To get remaining useful life of an engine at each cycle (Thakkar & Chaoui, 2022)  just multiply health indicator with total number of cycles. 
                                                     RUL=h(t)×t_max                                      




 


