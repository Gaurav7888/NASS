import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


st.markdown("# Page 2 ❄️")
st.sidebar.markdown("# Page 2 ❄️")

data_1 = pd.read_excel("NASS.xlsx", index_col='DATE', 
                     parse_dates=True)
data_1["spread"] = data_1["DGS30"] - data_1["DGS1"]   

data_1 = data_1.drop(['DGS30', 'DGS1'], axis=1)

tx = data_1.copy()

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