#!/usr/bin/env python
# coding: utf-8

# In[164]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.api import SimpleExpSmoothing
from scipy import signal
import math
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
from sklearn.model_selection import train_test_split
from numpy import linalg as LA
from datetime import datetime


# # READING DATA AND CREATING SEVERAL TIME SERIES FUNCTIONS

# In[165]:


df = pd.read_csv('household_power_consumption.txt', sep=';',
                 parse_dates={'dt': ['Date', 'Time']}, infer_datetime_format=True,
                 low_memory=False, na_values=['nan', '?'], index_col='dt')


# In[166]:


# Creating a new columns for all the engery in the rest of the house that doesn't include submeterings 1,2,&3.
# We are only given sub_metering1,sub_metering2,sub_metering3.
# (global_active_power*1000/60 - sub_metering_1 - sub_metering_2 - sub_metering_3) represents the active energy consumed
# every minute (in watt hour) in the household by electrical equipment not measured in sub-meterings 1, 2 and 3.
df['Sub_metering_4'] = (
        df['Global_active_power'] * 1000 / 60 - df['Sub_metering_1'] - df['Sub_metering_2'] - df['Sub_metering_3'])


# In[167]:


def difference(dataset, interval):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff
def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    if result[1] < 0.05:
        print("p-value is less than 0.05, reject null hypothesis thus time series data is Stationary")
    else:
        print("p-value is greater than 0.05, we failed to reject null hypothesis thus time series data is "
              "Non-Stationary")


from statsmodels.tsa.stattools import kpss


def kpss_test(timeseries):
    print('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    print(kpss_output)
def rolling_mean(x):
    roll = []
    for i in range(1, len(x) + 1):
        rolling = np.mean(x[:i])
        roll.append(rolling)
    return roll


def rolling_var(x):
    roll = []
    for i in range(1, len(x) + 1):
        rolling = np.var(x[:i])
        roll.append(rolling)
    return roll
def acf_cal(x, k):
    acf_values = []
    mn = np.mean(x)
    for k in range(0, k + 1):
        acf_values.append(sum((x - mn).iloc[k:] * (x.shift(k) - mn).iloc[k:]) / sum((x - mn) ** 2))
    return acf_values
def GPAC(ry, j0, k0):
    # get phi
    def phi(ry, j, k):
        # FIRST STEP: GET phi
        # creating 0 for placeholders for denominator
        denominator = np.zeros(shape=(k, k))
        # replacing denom matrix with ry(j) values
        for a in range(k):
            for b in range(k):
                denominator[a][b] = ry[abs(j + a - b)]
        # making a copy of denom for numerator
        numerator = denominator.copy()
        # creating last column for numerator
        numL = np.array(ry[j + 1:j + k + 1])
        numerator[:, -1] = numL
        phi = np.linalg.det(numerator) / np.linalg.det(denominator)
        return phi

    table0 = [[0 for i in range(1, k0)] for i in range(j0)]

    for c in range(j0):
        for d in range(1, k0):
            table0[c][d - 1] = phi(ry, c, d)

    pac = pd.DataFrame(np.array(table0), index=np.arange(j0), columns=np.arange(1, k0))
    return pac


def GPAC_plot(acf, j, k):
    gpac = GPAC(acf, j, k)
    plt.figure()
    sns.heatmap(gpac, annot=True)
    plt.title('GPAC Table')
    plt.xlabel('k values')
    plt.ylabel('j values')
    plt.show()
def ACF_PACF_Plot(y, lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()
def plotacf(s, k, n):
    # n = len(y)
    # k = number of lags
    t = pd.Series(s)
    acf = acf_cal(t, k)
    acf1 = acf[::-1][:-1]
    acf2 = acf1 + acf
    x = np.arange(-k, k + 1)
    insig = 1.96 / np.sqrt(len(np.arange(n)))
    plt.stem(x, acf2, markerfmt='o')
    plt.axhspan(-insig, insig, alpha=0.2, facecolor='0.5')
    plt.ylabel('Magnitude')
    plt.xlabel('Lags')
    plt.title('Autocorrelation Plot')
    plt.show()


# # CLEANING THE DATA

# In[168]:


# Checking the amount of nan values in the data
df.isnull().sum()


# In[169]:


# Using forward fill to handle nan values
df.ffill(axis='rows', inplace=True)


# In[170]:


df.isnull().sum()


# In[171]:


# Resample the data, because of computational time.
# Reduces the number of observations of 2075259 to 34589, structure is still retained
df_resample = df.resample('H').mean()


# In[172]:


# Plot of the dependent variable against time
y = df_resample['Global_active_power']
plt.figure()
plt.plot(y,)
plt.title('Active Power vs Time - Resampled')
plt.xlabel('Time')
plt.ylabel('Active Power (kilowatt)')
plt.xticks(rotation=45)
plt.show()


# # STATIONARITY CHECK

# In[173]:


# ACF/PACF Plot of the dependent variable

ACF_PACF_Plot(y, 50)


# In[174]:


# Correlation matrix
plt.figure()
corr = df_resample.corr()
ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(150, 275, s=80, l=55, n=9), square=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('Correlation Plot')
plt.tight_layout()
plt.show()


# In[175]:


# Stationary Check on raw data
# the null for ADF is that the time series non-stationary
ADF_Cal(y)


# In[176]:


# the null for KPSS is that the time series is stationary
sm.tsa.stattools.kpss(y, regression='ct')


# In[177]:


# plot of rolling mean and rolling variance of raw data

Rmean = rolling_mean(y)
Rvar = rolling_var(y)

plt.figure()
plt.plot(Rmean)
plt.title('Rolling Mean - Raw')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.grid()
plt.show()

plt.figure()
plt.plot(Rvar)
plt.title('Rolling Variance - Raw')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.grid()
plt.show()


# Stationary so we can continue

# # SPLITTING THE DATA

# In[178]:


# Splitting the data into Train and Test
# Split the dataset into train set 80% and test set 20%
y_train, y_test = train_test_split(df_resample, shuffle=False, test_size=0.2)
#X = df[['Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3','Sub_metering_4']]
#x_train, y_train = train_test_split()


# # TIME SERIES DECOMPOSITION

# In[179]:


# Time Series Decomposition
ActivePower = df_resample['Global_active_power']
ActivePower = pd.Series(np.array(df_resample['Global_active_power']),
                        index=pd.date_range('2006-12-16 17:00:00', periods=len(ActivePower)),
                        name='Activee Power (kilowatt)')


# In[180]:


from statsmodels.tsa.seasonal import STL
STL = STL(ActivePower)
res = STL.fit()

T = res.trend
S = res.seasonal
R = res.resid

# Seasonally adjusted data and plot it vs the original
sadjusted = ActivePower - S
detrended = ActivePower - T

plt.figure(figsize=(8,6))
plt.plot(ActivePower, label= 'Original')
plt.plot(sadjusted, label='Seasonally Adjusted')
plt.title("Original vs Seasonal Adjustment")
plt.xlabel("t")
plt.ylabel("Active Power (kilowatt)")
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.plot(ActivePower, label= 'Original')
plt.plot(detrended, label='Detrended')
plt.title("Original vs Detrended")
plt.xlabel("t")
plt.ylabel("Active Power (kilowatt)")
plt.legend()
plt.show()


# In[181]:


Ft = np.max([0,1 - np.var(R)/np.var(T+R)])
Fs = np.max([0,1 - np.var(R)/np.var(S+R)])
print("The strength of trend for this dataset is ", Ft)
print("The strength of seasonality for this dataset is", Fs)


# # HOLTS WINTER FORECAST

# In[182]:


# Holts winter method
# use training data to fit model
model = ets.ExponentialSmoothing(y_train['Global_active_power'], damped_trend= True,
                                 seasonal_periods=12, trend='mul', seasonal='mul').fit()
# prediction on train set
HW_train = model.forecast(steps=len(y_train['Global_active_power']))
HW_train = pd.DataFrame(HW_train, columns=['Global_active_power']).set_index(y_train.index)

# prediction on test set
HW_test = model.forecast(steps=len(y_test['Global_active_power']))
HW_test = pd.DataFrame(HW_test, columns=['Global_active_power']).set_index(y_test.index)


# In[183]:


# forecast error
HW_FE = (y_test['Global_active_power'].values).flatten() - HW_test['Global_active_power'].values.flatten()
# forecast MSE

HW_MSE = np.round(np.mean(np.square(np.subtract(HW_test['Global_active_power'].values, y_test['Global_active_power'].values))), 4)


# In[184]:


# MODEL ASSESSMENT


# In[185]:


print("The mean of the error of the HoltWinter Model is", np.mean(HW_FE))
print("The variance of error of the HoltWinter Model is :", np.var(HW_FE))
print("The MSE of the HoltWinter Model is", HW_MSE)
print("The RMSE of the HoltWinter Model is", np.sqrt(HW_MSE))


# In[186]:


plt.figure(figsize=(10, 5))
plt.plot(y_train['Global_active_power'], label='Train')
plt.plot(y_test['Global_active_power'], label='Test')
plt.plot(HW_test, label='Holt Winter Forecast')
plt.title('Holt Winter')
plt.xlabel('t')
plt.ylabel('Kilowatt')
plt.legend()
plt.show()


# In[187]:


plotacf(HW_FE, 20, len(HW_FE))


# In[188]:


print(sm.stats.acorr_ljungbox(HW_FE, lags=[20], boxpierce=True, return_df=True))


# In[189]:


# Residuals are not white, model is not a good fit


# # FEATURE SELECTION

# In[303]:


X = df_resample[['Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3','Sub_metering_4']]
X = sm.add_constant(X)
Y = df_resample[['Global_active_power']]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, shuffle = False, test_size = 0.2)
# OLS Model
model = sm.OLS(Y_train, X_train).fit()
print(model.summary())


# In[304]:


# Removing Global intensity P-value: 0.810
X.drop('Global_intensity', axis=1, inplace=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=False, test_size=0.2)
model = sm.OLS(Y_train, X_train).fit()
print(model.summary())


# In[305]:


# Removing const - P-value:0.486
X.drop('const', axis=1, inplace=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=False, test_size=0.2)
model = sm.OLS(Y_train, X_train).fit()
print(model.summary())


# # MULTIPLE LINEAR REGRESSION MODEL

# In[306]:


model = sm.OLS(Y_train, X_train).fit()
Tr_pred = model.predict(X_train)
T_pred = model.predict(X_test)
print(model.summary())


# In[307]:


X = X_train.values
print("The condition number for X is", LA.cond(X))
H = np.matmul(X.T,X)
s,d,v = np.linalg.svd(H)
print("SingularValues are", d)


# In[308]:


plt.figure(figsize=(10, 5))
plt.plot(Y_train, label='Train')  # Y_train Values
plt.plot(Y_test, label='Test')  # Y_test Values
plt.plot(Tr_pred, label='Train Pred')  # Train Predictions
plt.plot(T_pred, label='Test Predictions')  # Test Predictions using X_test
plt.legend(loc='best')
plt.xlabel('T')
plt.ylabel('Kilowatt')
plt.title('Train vs Test Predictions - Multiple Linear Regression Model')
plt.show()


# In[314]:


H_pred = model.predict(df_resample.drop(columns=['Global_active_power','Global_intensity']))


# In[316]:


plt.figure(figsize=(10, 5))
plt.plot(df_resample['Global_active_power'], label='Actual') 
plt.plot(H_pred, label='H step Predictions')  
plt.legend(loc='best')
plt.xlabel('T')
plt.ylabel('Kilowatt')
plt.title('H step - Multiple Linear Regression Model')
plt.show()


# In[313]:


df_resample


# In[196]:


#t-test
print("The p-value of t-test is: ",model.pvalues)

#f-test
print("The p-value of f-test is: ",model.f_pvalue)


# In[197]:


print("The AIC value of the model is :",model.aic)
print("The BIC value of the model is :",model.bic)
print("The R-squared value of the model is: ",model.rsquared)
print("The Adjusted R-squared value of the model is: ",model.rsquared_adj)


# In[198]:


OLS_PE = Y_train.values.flatten() - Tr_pred.values
OLS_FE = Y_test.values.flatten() - T_pred.values
OLS_MSE = round(np.square(OLS_FE).mean(),2)
print("The MSE of the residual is :", OLS_MSE)


# In[199]:


print(sm.stats.acorr_ljungbox(OLS_FE, lags=[20], boxpierce=True, return_df=True))


# In[200]:


OLS_acf = acf_cal(pd.Series(OLS_FE),20)


# In[201]:


OLS_acf2 = acf_cal(pd.Series(OLS_PE), 20)


# In[202]:


plotacf(OLS_FE, 20, len(OLS_FE))


# In[203]:


plotacf(OLS_PE, 20, len(OLS_PE))


# In[204]:


# Q value
Q_ML_FE = len(OLS_FE)*sum(np.square(OLS_acf[1:]))
print("The Q value is:", Q_ML_FE)


# In[205]:


print("The variance of Prediction error is:", np.var(OLS_PE))
print("The mean of Prediction error is:", np.mean(OLS_PE))
print("The variance of Forecast error is:", np.var(OLS_FE))
print("The mean of Forecast error is:", np.mean(OLS_FE))


# ###### BASE MODELS #######

# # AVERAGE METHOD

# In[206]:


#y_train1 = y_train.values 
#y_test1 = y_test.values
h_step = []

for i in range(len(y_test)):
    h_step.append(np.mean(y_train['Global_active_power'].values))


# In[207]:


AV_FE = y_test['Global_active_power'].values.flatten() - np.array(h_step).flatten()
AV_MSE = round(np.square(AV_FE).mean(),2)
print("The MSE of the residual for the Average Method is :", AV_MSE)
print("THe RMSE of the residual for the Average Method is:", np.sqrt(AV_MSE))


# In[208]:


print(sm.stats.acorr_ljungbox(AV_FE, lags=[20], boxpierce=True, return_df=True))


# In[209]:


AV_acf = acf_cal(pd.Series(AV_FE),20)


# In[210]:


plotacf(AV_FE, 20, len(AV_FE))


# In[211]:


# Q value
Q_AV_FE = len(AV_FE)*sum(np.square(AV_acf[1:]))
print("The Q value is:", Q_AV_FE)


# In[212]:


print("The variance of Forecast error is:", np.var(AV_FE))
print("The mean of Forecast error is:", np.mean(AV_FE))


# In[213]:


plt.figure(figsize=(10, 5))
plt.plot(Y_train,label= "Train Data")
plt.plot(Y_test,label= "Test Data")
plt.plot(y_test.index,h_step, label= "Average Method Forecast")
plt.legend(loc='best')
plt.title('Train vs Test Predictions - Average Method Model')
plt.xlabel("t")
plt.ylabel("Kilowatt")
plt.show()


# # NAIVE METHOD

# In[214]:


h_stepN = []
for i in range(len(y_test)):
    h_stepN.append(y_train['Global_active_power'][-1])


# In[215]:


N_FE = y_test['Global_active_power'].values.flatten() - np.array(h_stepN).flatten()
N_MSE = round(np.square(N_FE).mean(),2)
print("The MSE of the residual for the Average Method is :", N_MSE)
print("THe RMSE of the residual for the Average Method is:", np.sqrt(N_MSE))


# In[216]:


print(sm.stats.acorr_ljungbox(N_FE, lags=[20], boxpierce=True, return_df=True))


# In[217]:


N_acf = acf_cal(pd.Series(N_FE),20)


# In[218]:


plotacf(N_FE, 20, len(N_FE))


# In[219]:


# Q value
Q_N_FE = len(N_FE)*sum(np.square(N_acf[1:]))
print("The Q value is:", Q_N_FE)


# In[220]:


print("The variance of Forecast error is:", np.var(N_FE))
print("The mean of Forecast error is:", np.mean(N_FE))


# In[221]:


plt.figure(figsize=(10, 5))
plt.plot(Y_train,label= "Train Data")
plt.plot(Y_test,label= "Test Data")
plt.plot(y_test.index,h_stepN, label= "Naive Method Forecast")
plt.legend(loc='best')
plt.title('Train vs Test Predictions - Naive Method Model')
plt.xlabel("t")
plt.ylabel("Kilowatt")
plt.show()


# # DRIFT METHOD

# In[222]:


h_stepDR = []
for i in range(1,len(y_test) + 1):
    slope = (y_train['Global_active_power'][-1] - y_train['Global_active_power'][0]) / (len(y_train)-1)
    h_stepDR.append(y_train['Global_active_power'][-1]+ i*slope)


# In[223]:


DR_FE = y_test['Global_active_power'].values.flatten() - np.array(h_stepDR).flatten()
DR_MSE = round(np.square(DR_FE).mean(),2)
print("The MSE of the residual for the Drift Method is :", DR_MSE)
print("THe RMSE of the residual for the Drift Method is:", np.sqrt(DR_MSE))


# In[224]:


print(sm.stats.acorr_ljungbox(DR_FE, lags=[20], boxpierce=True, return_df=True))


# In[225]:


DR_acf = acf_cal(pd.Series(DR_FE),20)


# In[226]:


plotacf(DR_FE, 20, len(DR_FE))


# In[227]:


# Q value
Q_DR_FE = len(DR_FE)*sum(np.square(DR_acf[1:]))
print("The Q value is:", Q_DR_FE)


# In[228]:


print("The variance of Forecast error is:", np.var(DR_FE))
print("The mean of Forecast error is:", np.mean(DR_FE))


# In[229]:


plt.figure(figsize=(10, 5))
plt.plot(Y_train,label= "Train Data")
plt.plot(Y_test,label= "Test Data")
plt.plot(y_test.index,h_stepDR, label= "Drift Method Forecast")
plt.legend(loc='best')
plt.title('Train vs Test Predictions - Drift Method Model')
plt.xlabel("t")
plt.ylabel("Kilowatt")
plt.show()


# # SES METHOD

# In[230]:


fit = SimpleExpSmoothing(np.asarray(y_train['Global_active_power'])).fit(smoothing_level=0.5, optimized=False)
h_stepSES = fit.forecast(len(y_test))


# In[231]:


SES_FE = y_test['Global_active_power'].values.flatten() - np.array(h_stepSES).flatten()
SES_MSE = round(np.square(SES_FE).mean(),2)
print("The MSE of the residual for the Average Method is :", SES_MSE)
print("THe RMSE of the residual for the Average Method is:", np.sqrt(SES_MSE))


# In[232]:


print(sm.stats.acorr_ljungbox(SES_FE, lags=[20], boxpierce=True, return_df=True))


# In[233]:


SES_acf = acf_cal(pd.Series(SES_FE),20)


# In[234]:


plotacf(SES_FE, 20, len(SES_FE))


# In[235]:


# Q value
Q_SES_FE = len(SES_FE)*sum(np.square(SES_acf[1:]))
print("The Q value is:", Q_SES_FE)


# In[236]:


print("The variance of Forecast error is:", np.var(SES_FE))
print("The mean of Forecast error is:", np.mean(SES_FE))


# In[237]:


plt.figure(figsize=(10, 5))
plt.plot(Y_train,label= "Train Data")
plt.plot(Y_test,label= "Test Data")
plt.plot(y_test.index,h_stepSES, label= "SES Forecast")
plt.legend(loc='best')
plt.title('Train vs Test Predictions - SES Method Model')
plt.xlabel("t")
plt.ylabel("Kilowatt")
plt.show()


# # ARMA MODEL

# In[238]:


y_acf = acf_cal(pd.Series(y.values),20)


# In[239]:


GPAC_plot(y_acf, 8, 8)


# In[240]:


# potentially (2,0) or (1,0) or (2,1)


# In[242]:


model = sm.tsa.ARMA(y,(1,0)).fit(trend='nc',disp=0)


# In[243]:


print(model.summary())


# In[244]:


for i in range(1):
    print("The AR estimated coefficient a{}".format(i), "is:", model.params[i])

for i in range(1):
    print("The confidence interval for estimated coefficient a{}".format(i), "is:", model.conf_int())


# In[245]:


print("The standard deviation of the parameter estimates is: ",model.summary().tables[1])


# In[255]:


print("The corvariance of estimated parameters is:",model.cov_params())


# In[246]:


print("The standard deviation of the parameter estimate is: ",np.square(model.sigma2))


# In[247]:


# Prediction with ARMA
onestep_ARMA = model.predict(start=0, end=len(y_train)-1)
hstep_ARMA = model.predict(start=len(y_train)-1, end=len(df_resample))


# In[248]:


ARMA_PE = y_train['Global_active_power'].values.flatten() - np.array(onestep_ARMA).flatten()
ARMA_FE = y_test['Global_active_power'].values.flatten() - np.array(hstep_ARMA[2:]).flatten()
ARMA_MSE = round(np.square(ARMA_FE).mean(),2)
print("The MSE of the residual for the Average Method is :", ARMA_MSE)
print("THe RMSE of the residual for the Average Method is:", np.sqrt(ARMA_MSE))


# In[249]:


print(sm.stats.acorr_ljungbox(ARMA_FE, lags=[20], boxpierce=True, return_df=True))


# In[250]:


ARMA_acf = acf_cal(pd.Series(ARMA_FE),20)


# In[251]:


plotacf(ARMA_FE, 20, len(ARMA_FE))


# In[252]:


# Q value
Q_ARMA_FE = len(ARMA_FE)*sum(np.square(ARMA_acf[1:]))
print("The Q value is:", Q_ARMA_FE)


# In[253]:


print("The variance of Prediction error is:", np.var(ARMA_PE))
print("The mean of Prediction error is:", np.mean(ARMA_PE))
print("The variance of Forecast error is:", np.var(ARMA_FE))
print("The mean of Forecast error is:", np.mean(ARMA_FE))
print("The variance of prediction vs forecast error is:",ARMA_PE.var()/ARMA_FE.var())


# In[254]:


plt.figure(figsize=(10, 5))
plt.plot(Y_train,label= "Train Data")
plt.plot(Y_test,label= "Test Data")
plt.plot(y_test.index,hstep_ARMA[2:], label= "ARMA Forecast")
plt.legend(loc='best')
plt.title('Train vs Test Predictions - ARMA Model')
plt.xlabel("t")
plt.ylabel("Kilowatt")
plt.show()


# In[256]:


model2 = sm.tsa.ARMA(y,(2,1)).fit(trend='nc',disp=0)


# In[257]:


print(model2.summary())


# In[147]:


for i in range(2):
    print("The AR coefficient a{}".format(i), "is:", model2.params[i])
for i in range(1):
    print("The MA coefficient a{}".format(i), "is:", model2.params[i + 2])


# In[258]:


for i in range(1):
    print("The confidence interval for estimated coefficient is:", model2.conf_int())


# In[150]:


print("The standard deviation of the parameter estimates is: ",np.square(model2.sigma2))


# In[260]:


print("the covariance Matrix for the data is", model2.cov_params())


# In[261]:


# Prediction with ARMA(2,1)
onestep_ARMA2 = model2.predict(start=0, end=len(y_train)-1)
hstep_ARMA2 = model2.predict(start=len(y_train)-1, end=len(df_resample))


# In[262]:


ARMA_PE2 = y_train['Global_active_power'].values.flatten() - np.array(onestep_ARMA2).flatten()
ARMA_FE2 = y_test['Global_active_power'].values.flatten() - np.array(hstep_ARMA2[2:]).flatten()
ARMA_MSE2 = round(np.square(ARMA_FE2).mean(),2)
print("The MSE of the residual for the Average Method is :", ARMA_MSE2)
print("THe RMSE of the residual for the Average Method is:", np.sqrt(ARMA_MSE2))


# In[153]:


print(sm.stats.acorr_ljungbox(ARMA_FE2, lags=[20], boxpierce=True, return_df=True))


# In[154]:


ARMA_acf2 = acf_cal(pd.Series(ARMA_FE2),20)


# In[155]:


plotacf(ARMA_FE2, 20, len(ARMA_FE2))


# In[263]:


# Q value
Q_ARMA_FE2 = len(ARMA_FE2)*sum(np.square(ARMA_acf2[1:]))
print("The Q value is:", Q_ARMA_FE2)


# In[264]:


print("The variance of Prediction error is:", np.var(ARMA_PE2))
print("The mean of Prediction error is:", np.mean(ARMA_PE2))
print("The variance of Forecast error is:", np.var(ARMA_FE2))
print("The mean of Forecast error is:", np.mean(ARMA_FE2))
print("The variance of prediction vs forecast error is:",ARMA_PE2.var()/ARMA_FE2.var())


# In[266]:


plt.figure(figsize=(10, 5))
plt.plot(Y_train,label= "Train Data")
plt.plot(Y_test,label= "Test Data")
plt.plot(y_test.index,hstep_ARMA2[2:], label= "ARMA Forecast")
plt.legend(loc='best')
plt.title('Train vs Test Predictions - ARMA Model(2,1)')
plt.xlabel("t")
plt.ylabel("Kilowatt")
plt.show()


# # LSTM

# In[ ]:


# code for LSTM does not belong to me, it was adopted from Susan Li fro towardsdatascience
# https://towardsdatascience.com/time-series-analysis-visualization-forecasting-with-lstm-77a905180eba


# In[112]:


get_ipython().system('pip install tensorflow')


# In[121]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping


# In[275]:


data = df_resample.Global_active_power.values #numpy.ndarray
data = data.astype('float32')
data = np.reshape(data, (-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
train_size = int(len(data) * 0.80)
test_size = len(data) - train_size
train, test = data[0:train_size,:], data[train_size:len(data),:]


# In[276]:


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


# In[277]:


# reshape into X=t and Y=t+1
look_back = 30
X_train, Y_train = create_dataset(train, look_back)
X_test, Y_test = create_dataset(test, look_back)


# In[278]:


X_train.shape


# In[279]:


Y_train.shape


# In[280]:


# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


# In[281]:


model = Sequential()
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(X_train, Y_train, epochs=20, batch_size=70, validation_data=(X_test, Y_test), 
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

# Training Phase
model.summary()


# In[282]:


# make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
# invert predictions
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

print('Train Mean Absolute Error:', mean_absolute_error(Y_train[0], train_predict[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0])))
print('Test Mean Absolute Error:', mean_absolute_error(Y_test[0], test_predict[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0])))


# In[283]:


plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show();


# In[285]:


aa=[x for x in range(200)]
plt.figure(figsize=(10,5))
plt.plot(Y_test[0], marker='.', label="Test")
plt.plot(test_predict[:,0], 'r', label="Test Pred")
#plt.plot(Y_train[0], marker='.', label="Train")
#plt.plot(train_predict[:,0],'g',label="Train Pred")
# plt.tick_params(left=False, labelleft=True) #remove ticks
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Global_active_power(KW)', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.title('LSTM Forecast')
plt.show();


# In[273]:


len(y_train['Global_active_power'].values.flatten())


# In[294]:


LSTM_PE = np.subtract(Y_train[0], train_predict[:,0])
LSTM_PMSE = np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0]))
LSTM_FE = np.subtract(Y_test[0], test_predict[:,0])
LSTM_MSE = np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0]))


# In[297]:


LSTM_acf = acf_cal(pd.Series(LSTM_FE),20)


# In[298]:


# Q value
Q_LSTM_FE = len(LSTM_FE)*sum(np.square(LSTM_acf[1:]))
print("The Q value is:", Q_LSTM_FE)


# In[299]:


plotacf(LSTM_FE, 20, len(LSTM_FE))


# In[300]:


print(sm.stats.acorr_ljungbox(LSTM_FE, lags=[20], boxpierce=True, return_df=True))


# In[ ]:




