
# coding: utf-8

# In[6]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import datetime as dt
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from matplotlib import style
import os
import bs4 as bs
import pickle
import pandas as pd
import numpy as np
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import warnings

style.use('seaborn-ticks')

start= dt.datetime(2010,1,1)
end=dt.datetime(2017,12,31)

tickers=['TSLA','AAPL','MSFT','GOOG','FB','TWTR','AMZN','IBM','NFLX','WMT','KO']

def get_data_from_quandl():
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
    for ticker in tickers:
        print(ticker)
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df=web.DataReader(ticker,'quandl',start,end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
            

def compile_data():
    main_df=pd.DataFrame()
    for ticker in tickers:
        df=pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date',inplace=True)
        df.drop(['Open','High','Low','Close','Volume','ExDividend','SplitRatio','AdjOpen','AdjHigh','AdjLow','AdjVolume'],1,inplace=True)
        df.rename(columns={'AdjClose':ticker},inplace=True)
        if main_df.empty:
            main_df=df
        else:
            main_df=main_df.join(df,how='outer')
        
    print(main_df.tail())
    main_df.to_csv('joined_closes.csv')

def visualise_data():
    df=pd.read_csv('joined_closes.csv')
    #df['AAPL'].plot()
    #plt.show()
    df_corr=df.corr()
    print(df_corr.head())
    data=df_corr.values
    fig=plt.figure()
    ax=fig.add_subplot(111)
    
    heatmap=ax.pcolor(data,cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    
    ax.set_xticks(np.arange(data.shape[0])+0.5,minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5,minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    column_labels=df_corr.columns
    row_labels=df_corr.index
    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    plt.show()

def process_data(ticker):
    df=pd.read_csv('joined_closes.csv',index_col=0)
    days=7
    tickers=df.columns.values.tolist()
    df.fillna(0,inplace=True)
    for i in range(1,days+1):
        df['{}_{}d'.format(ticker,i)]=(df[ticker].shift(-i)-df[ticker])/df[ticker]
    df.fillna(0,inplace=True)
    return tickers,df

def buy_sell_hold(*args):
    requirement=0.02
    cols=[c for c in args]
    for col in cols:
        if col>requirement:
            return 1
        if col<-requirement:
            return -1
    return 0

from collections import Counter
def extract_featuresets(ticker):
    tickers, df = process_data(ticker)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)]))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    return X, y, df



def do_ml(ticker):
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    X, y, df = extract_featuresets(ticker)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.20)

    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                                ('knn', neighbors.KNeighborsClassifier()),
                                ('rfor', RandomForestClassifier())])

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test , y_test)
    print('accuracy :', confidence*100, '%')
    predictions = clf.predict(X_test)
    print('predicted class counts:', Counter(predictions))
    plt.plot(predictions)
    plt.show()


# In[7]:


get_data_from_quandl()


# In[8]:


compile_data()


# In[9]:


visualise_data()


# In[10]:


do_ml('GOOG')


# In[12]:


do_ml('TWTR')


# In[15]:


do_ml('AAPL')


# In[16]:


do_ml('AMZN')


# In[17]:


do_ml('NFLX')


# In[18]:


do_ml('WMT')


# In[19]:


do_ml('KO')


# In[20]:


do_ml('TSLA')


# In[21]:


do_ml('FB')


# In[22]:


do_ml('IBM')


# In[23]:


do_ml('MSFT')


# In[ ]:




