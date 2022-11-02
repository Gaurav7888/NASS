import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.knn import KNN
from pyod.models.lof import LOF
import scipy as sp


st.markdown("# Main page")
st.sidebar.markdown(" :)( ")

st.title("The Great Depression")

st.subheader("Spread of Difference (30yrs - 1yr)")


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

st.subheader("Yield Curves in Time")

data = pd.read_csv("Data Yield Curve.csv")
data.rename(columns = {'CHHUSD':'CHFUSD'}, inplace = True)
data['Year'] = data['Date'].apply(lambda i: i.split('-')[0]).astype(int)

fig = plt.figure(figsize=(16,9))

ax1 = sns.lineplot(x = 'Year', y = '1 yr', data=data, err_style=None, lw = 5, estimator='mean',
                  palette = "Dark2")
ax2 = sns.lineplot(x = 'Year', y = '30YR', data=data, err_style=None, lw = 5, estimator='mean',
                  palette = "Dark2")

plt.title('Yield values for 1 and 30 years maturity - from 1977 to 2019', fontsize = 25)
plt.xlabel('Year', fontsize = 20)
plt.ylabel('Yield value(%)', fontsize = 20)
plt.legend(['1yr','30yrs'], ncol=2, loc='upper right', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

ax1.set_xlim(1977,2019);

st.pyplot(fig)

st.subheader("Impact on goods")

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex = True)
fig.set_figheight(13)
fig.set_figwidth(16)
plt.subplots_adjust(hspace = 0.1)

ax1.plot(data['Year'], data['GOLD'], lw = 4)
ax1.plot([1980, 1980], [90, 1800], '#FF4026', lw = 2)
ax1.plot([1989, 1989], [90, 1800], '#FF4026', lw = 2)
ax1.plot([2000, 2000], [90, 1800], '#FF4026', lw = 2)
ax1.plot([2006, 2006], [90, 1800], '#FF4026', lw = 2)
ax1.plot([2008, 2008], [90, 1800], '#F39C12', lw = 3)

ax2.plot(data['Year'], data['OIL'], lw = 4)
ax2.plot([1980, 1980], [0, 134], '#FF4026', lw = 2)
ax2.plot([1989, 1989], [0, 134], '#FF4026', lw = 2)
ax2.plot([2000, 2000], [0, 134], '#FF4026', lw = 2)
ax2.plot([2006, 2006], [0, 134], '#FF4026', lw = 2)
ax2.plot([2008, 2008], [0, 134], '#F39C12', lw = 3)

ax3.plot(data['Year'], data['CHFUSD'], lw = 4)
ax3.plot([1980, 1980], [0.78, 2.8], '#FF4026', lw = 2)
ax3.plot([1989, 1989], [0.78, 2.8], '#FF4026', lw = 2)
ax3.plot([2000, 2000], [0.78, 2.8], '#FF4026', lw = 2)
ax3.plot([2006, 2006], [0.78, 2.8], '#FF4026', lw = 2)
ax3.plot([2008, 2008], [0.78, 2.8], '#F39C12', lw = 3)

ax4.plot(data['Year'], data['JPYUSD'], lw = 4)
ax4.plot([1980, 1980], [76.64, 281], '#FF4026', lw = 2)
ax4.plot([1989, 1989], [76.64, 281], '#FF4026', lw = 2)
ax4.plot([2000, 2000], [76.64, 281], '#FF4026', lw = 2)
ax4.plot([2006, 2006], [76.64, 281], '#FF4026', lw = 2)
ax4.plot([2008, 2008], [76.64, 281], '#F39C12', lw = 3)

ax1.set_xlim(1977,2019) 
ax1.set_ylim(data['GOLD'].min(),data['GOLD'].max()) 
ax2.set_xlim(1977,2019)
ax2.set_ylim(data['OIL'].min(),data['OIL'].max()) 
ax3.set_xlim(1977,2019) 
ax3.set_ylim(data['CHFUSD'].min(),data['CHFUSD'].max()) 
ax4.set_xlim(1977,2019)
ax4.set_ylim(data['JPYUSD'].min(),data['JPYUSD'].max()) 

plt.xlabel('YEAR',fontsize = 20)
ax1.set_ylabel('GOLD', fontsize = 20)
ax2.set_ylabel('OIL', fontsize = 20)
ax3.set_ylabel('CHFUSD', fontsize = 20)
ax4.set_ylabel('JPYUSD', fontsize = 20)

fig.suptitle('Impact of the Yield Curve inversion on price of goods', fontsize = 25);

st.pyplot(fig)

st.subheader("Correlation betweeen Factors")

data['Flag is neg'] = np.where(data['SPREAD'] < 0, 1, 0)
corr_gold = sp.stats.pearsonr(x = data['SPREAD'], y = data['GOLD'])
corr_oil = sp.stats.pearsonr(x = data['SPREAD'], y = data['OIL'])
corr_sp500 = sp.stats.pearsonr(x = data['SPREAD'], y = data['SP500'])
corr_CHHUSD = sp.stats.pearsonr(x = data['SPREAD'], y = data['CHFUSD'])
corr_JPYUSD = sp.stats.pearsonr(x = data['SPREAD'], y = data['JPYUSD'])

st.write('Gold:') 
st.write(corr_gold)
st.write('Oil:') 
st.write(corr_oil) 
st.write('SP500:') 
st.write(corr_sp500)
st.write('CHFUSD:') 
st.write(corr_CHHUSD)
st.write('JPYUSD:') 
st.write(corr_JPYUSD) 


st.subheader("HeatMap")
corr = data[['SP500', 'GOLD', 'OIL', 'CHFUSD', 'JPYUSD', 'Flag is neg']].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(16, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});

st.pyplot(f)


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
