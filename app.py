import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.knn import KNN
from pyod.models.lof import LOF


st.markdown("# Main page")
st.sidebar.markdown("# Main page")

st.title("svacvdcas")


data = pd.read_excel("NASS.xlsx", parse_dates=True)


data_load_state = st.text('Loading data...')

data_load_state.text("Done! (using st.cache)")   





  
data["spread"] = data["DGS30"] - data["DGS1"]
data = data.drop(["DGS30","DGS1"],axis=1)
data['year'] = pd.DatetimeIndex(data['DATE']).year

fig = plt.figure(figsize=(40,30))

ax3 = sns.lineplot(x = 'year', y = 'spread', data = data, lw = 6, err_style=None, estimator='mean')
plt.plot([1977, 2022], [0, 0], color = '#839192', lw = 5)

plt.title('Spread of difference between 30Yrs Yield and 1Yr Yield (%)', fontsize = 25)
plt.xlabel('Year', fontsize = 20)
plt.ylabel('Spread', fontsize = 20)

ax3.set_xlim(1977, 2022)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 15);
st.pyplot(fig) 

st.subheader("Prediction Model")


data_1 = pd.read_excel("NASS.xlsx", index_col='DATE', 
                     parse_dates=True)
data_1["spread"] = data_1["DGS30"] - data_1["DGS1"]   

data_1 = data_1.drop(['DGS30', 'DGS1'], axis=1)

st.subheader("KNN Outlier Detection")

knn = KNN(contamination=0.03,

          method='mean',

          n_neighbors=5)

knn.fit(data_1)

KNN(algorithm='auto', contamination=0.05, leaf_size=30, method='mean',

  metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=5, p=2,

  radius=1.0)

predicted = pd.Series(knn.predict(data_1),index=data_1.index)

st.write("Number of outliers")
st.write(predicted.sum())

outliers = predicted[predicted == 1]
outliers = data_1.loc[outliers.index]

st.write(outliers)

def knn_anomaly(df, method='mean', contamination=0.05, k=5):
    knn = KNN(contamination=contamination,
              method=method,
              n_neighbors=5)
    knn.fit(df)    
    decision_score = pd.DataFrame(knn.decision_scores_, 
                          index=df.index, columns=['score'])
    n = int(len(df)*contamination)
    outliers = decision_score.nlargest(n, 'score')
    return outliers, knn.threshold_

for method in ['mean', 'median', 'largest']:
    o, t = knn_anomaly(data_1, method=method)
    st.write(method,t)
    st.write(o)

st.subheader("LOF Outlier")
lof = LOF(contamination=0.03, n_neighbors=5)

lof.fit(data_1)

LOF(algorithm='auto', contamination=0.03, leaf_size=30, metric='minkowski',

  metric_params=None, n_jobs=1, n_neighbors=5, novelty=True, p=2)

predicted = pd.Series(lof.predict(data_1),index=data_1.index)

st.write("Number of outliers")
st.write(predicted.sum())

outliers = predicted[predicted == 1]

outliers = data_1.loc[outliers.index]

st.write(outliers)

st.subheader("CBLOF")

tx = data_1.copy()
from pyod.models.cblof import CBLOF
cblof = CBLOF(n_clusters=4, contamination=0.03)
cblof.fit(tx)
predicted = pd.Series(lof.predict(tx), 
                      index=tx.index)
outliers = predicted[predicted == 1]
outliers = tx.loc[outliers.index] 

st.write(predicted.sum())
st.write(outliers)

st.subheader("IForest")

from pyod.models.iforest import IForest

iforest = IForest(contamination=0.03,

                 n_estimators=100,

                 random_state=0)

iforest.fit(data_1)

IForest(behaviour='old', bootstrap=False, contamination=0.05,

    max_features=1.0, max_samples='auto', n_estimators=100, n_jobs=1,

    random_state=0, verbose=0)

predicted = pd.Series(iforest.predict(tx),

                      index=tx.index)

outliers = predicted[predicted == 1]

outliers = tx.loc[outliers.index]

st.write(predicted.sum())
st.write(outliers)

st.subheader("OCSVM")

from pyod.models.ocsvm import OCSVM

ocsvm = OCSVM(contamination=0.03, kernel='rbf')

ocsvm.fit(tx)

OCSVM(cache_size=200, coef0=0.0, contamination=0.03, degree=3, gamma='auto',

   kernel='rbf', max_iter=-1, nu=0.5, shrinking=True, tol=0.001,

   verbose=False)

predicted = pd.Series(ocsvm.predict(tx),

                      index=tx.index)

outliers = predicted[predicted == 1]

outliers = tx.loc[outliers.index]                

st.write(predicted.sum())
st.write(outliers)

from pyod.utils.utility import standardizer

scaled = standardizer(tx)

for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    ocsvm = OCSVM(contamination=0.03, kernel=kernel)
    predict = pd.Series(ocsvm.fit_predict(scaled), 
                      index=tx.index, name=kernel)
    outliers = predict[predict == 1]
    outliers = tx.loc[outliers.index]
    st.write(kernel)
    st.write(outliers)

st.subheader("COPOD")

from pyod.models.copod import COPOD

copod = COPOD(contamination=0.03)

copod.fit(tx)

COPOD(contamination=0.5, n_jobs=1)

predicted = pd.Series(copod.predict(tx),

                      index=tx.index)

outliers = predicted[predicted == 1]

outliers = tx.loc[outliers.index]


st.write(predicted.sum())
st.write(outliers)

st.subheader("MAD")

from pyod.models.mad import MAD
mad = MAD(threshold=3)
predicted = pd.Series(mad.fit_predict(tx), 
                      index=tx.index)
outliers = predicted[predicted == 1]
outliers = tx.loc[outliers.index]
st.write(predicted.sum())
st.write(outliers)


st.subheader("Anomaly Detection")

result = '''abod

              
DATE                                        
1978-12-01 
1979-05-01
1979-06-01 
1979-09-01 
1979-11-01 
1980-03-01 
1980-10-01 
1980-12-01 
1981-01-01 
1981-04-01 
1981-07-01 
1981-08-01 
1982-02-01 
1982-04-01 
1992-10-01  
2011-02-01  
------
cluster
            
DATE                                        
1979-01-01 
1979-09-01 
1979-10-01 
1979-11-01 
1979-12-01 
1980-02-01 
1980-03-01 
1980-04-01 
1980-11-01 
1980-12-01 
1981-01-01 
1981-02-01 
1981-05-01 
1981-06-01   
1981-07-01 
1981-08-01 
1981-09-01 
------
cof
              
DATE                                        
1979-07-01 
1979-10-01 
1980-03-01
1981-05-01 
1981-10-01 
1982-02-01 
1982-04-01 
1982-11-01  
1986-08-01 
1988-07-01 
1988-12-01  
1992-01-01 
2000-02-01  
2002-05-01  
2013-04-01  
2014-04-01  
2020-10-01  
------
iforest
             
DATE                                        
1978-12-01 
1979-01-01 
1979-09-01 
1979-10-01 
1979-11-01 
1980-01-01 
1980-03-01 
1980-04-01 
1980-12-01 
1981-01-01 
1981-02-01 
1981-05-01 
1981-06-01 
1981-07-01 
1981-08-01 
2010-02-01  
2011-02-01  
------
histogram
              
DATE                                        
1979-10-01
1979-11-01 
1980-03-01 
1980-12-01 
1981-01-01 
1981-05-01
1981-07-01 
1981-08-01 
------
knn
             
DATE                                        
1979-01-01 
1979-07-01 
1979-10-01 
1979-11-01 
1980-01-01 
1980-03-01 
1980-12-01 
1981-05-01 
1981-07-01 
1981-08-01 
------
lof
          
DATE                                        
1978-10-01 
1979-06-01 
1979-07-01 
1980-03-01 
1980-10-01 
1981-10-01 
1982-02-01 
1982-03-01 
1982-04-01 
1989-03-01 
2000-08-01 
2011-02-01  
------
svm
              
Date                                       
1979-01-01 
1979-09-01 
1979-10-01 
1979-11-01 
1979-12-01 
1980-02-01 
1980-03-01 
1980-04-01 
1980-11-01 
1980-12-01 
1981-01-01 
1981-02-01 
1981-05-01 
1981-06-01
1981-07-01 
1981-08-01 
1981-09-01 '''

st.write(result)

st.subheader("Statistical Graphs")

import seaborn as sns
fig = plt.figure(figsize=(10, 4))
sns.histplot(tx)
st.pyplot(fig)

fig2 = plt.figure(figsize=(10, 4))
sns.displot(tx, kind='hist', height=3, aspect=4)
st.pyplot(fig2)

fig3 = plt.figure(figsize=(10, 4))
sns.boxplot(tx['spread'])
st.pyplot(fig3)

fig4 = plt.figure(figsize=(10, 4))
sns.boxplot(tx['spread'], whis=1.5)
st.pyplot(fig4)

fig5 = plt.figure(figsize=(10, 4))
sns.boxenplot(tx['spread'])
st.pyplot(fig5)

fig6 = plt.figure(figsize=(10, 4))
sns.violinplot(tx['spread'])
st.pyplot(fig6)

from pandas.plotting import lag_plot
fig7 = plt.figure(figsize=(10, 4))
lag_plot(tx)
st.pyplot(fig7)

percentiles = [0, 0.05, .10, .25, .5, .75, .90, .95, 1]

st.write(tx.describe(percentiles= percentiles))

percentiles = [0, 5, 10, 25, 50, 75, 90, 95, 100]

np.percentile(tx, percentiles)

def iqr_outliers(data):

  q1, q3 = np.percentile(data, [25, 75])

  IQR = q3 - q1

  lower_fence = q1 - (1.5 * IQR)

  upper_fence = q3 + (1.5 * IQR)

  return data[(data.spread > upper_fence) | (data.spread < lower_fence)]

outliers = iqr_outliers(tx)

st.write(outliers)

def iqr_outliers(data, p):
    q1, q3 = np.percentile(data, [25, 75])
    IQR = q3 - q1
    lower_fence = q1 - (p * IQR)
    upper_fence = q3 + (p * IQR)
    return data[(data.spread > upper_fence) | (data.spread < lower_fence)]

for p in [1,1.3, 1.5,1.7,1.9, 2.0,2.3, 2.5]:
    st.write(p)
    st.write(iqr_outliers(tx, p))
    st.write('-'*15)

def zscore(df, degree=3):

  data = df.copy()

  data['zscore'] = (data - data.mean())/data.std()

  outliers = data[(data['zscore'] <= -degree) | (data['zscore'] >= degree)]

  return outliers['spread'], data

threshold = 2.5

outliers, transformed = zscore(tx, threshold)

fig8 = plt.figure(figsize=(10, 4))
transformed.hist()
st.pyplot(fig8)
st.write(outliers)

import matplotlib.pyplot as plt
def plot_zscore(data, d=3):

  n = len(data)

  fig = plt.figure(figsize=(8,8))

  plt.plot(data,'k^')

  plt.plot([0,n],[d,d],'r--')

  plt.plot([0,n],[-d,-d],'r--')

  st.pyplot(fig)

data = transformed['zscore'].values

fig9 = plt.figure(figsize=(10, 4))
plot_zscore(data, d=2.5)
st.pyplot(fig9)

from statsmodels.stats.diagnostic import kstest_normal
def test_normal(df):
    t_test, p_value = kstest_normal(df)
    if p_value < 0.05:
        st.write("Reject null hypothesis. Data is not normal")
    else:
       st.write("Fail to reject null hypothesis. Data is normal")

test_normal(tx)       

import scipy.stats as stats
st.write(stats.norm.ppf(0.75))

def modified_zscore(df, degree=3):

    data = df.copy()

    s = stats.norm.ppf(0.75)

    numerator = s*(data - data.median())

    MAD = np.abs(data - data.median()).median()

    data['m_zscore'] = numerator/MAD

    outliers = data[(data['m_zscore'] > degree) | (data['m_zscore'] < -degree)]

    return outliers['spread'], data

threshold = 3

outliers, transformed = modified_zscore (tx, threshold)

fig9 = plt.figure(figsize=(10, 4))
transformed.hist()
st.pyplot(fig9)
st.write(outliers)

def plot_m_zscore(data, d=3):

  n = len(data)

  fig = plt.figure(figsize=(8,8))

  plt.plot(data,'k^')

  plt.plot([0,n],[d,d],'r--')

  plt.plot([0,n],[-d,-d],'r--')

  st.pyplot(fig)


data = transformed['m_zscore'].values

plot_m_zscore(data, d=2.5)

import scipy
import matplotlib.pyplot as plt

fig10 = plt.figure(figsize=(8,8))
res = scipy.stats.probplot(tx.values.reshape(-1), plot=plt)
st.write(fig10)

from statsmodels.graphics.gofplots import qqplot
fig11 = plt.figure(figsize=(8,8))
qqplot(tx.values.reshape(-1), line='s')
st.write(fig11)