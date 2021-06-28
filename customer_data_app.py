# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 14:11:01 2021

@author: bmussa
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
import seaborn as sns
import matplotlib
from matplotlib.figure import Figure
import base64
from io import BytesIO

st.set_page_config(layout="wide")

matplotlib.use("agg")

_lock = RendererAgg.lock

sns.set_style('white')

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.beta_columns(
    (.1, 2, .2, 2, .1))

row0_1.title('Analyse your customer data')
#row0_1.write('')
row0_2.subheader('by Bilal Mussa (Linkedin: https://www.linkedin.com/in/bilalmussa/ )')

row1_spacer1, row1_1, row1_spacer2 = st.beta_columns((.1, 3.2, .1))

with row1_1:
    st.markdown("Hi, welcome to my quick analysis app - you only need 4 columns of data to profile your customers. The purpose of this app is to give you a quick overview into your customer data using the fewest columns. This app does not store any data. It simply reads it in, does the analysis and outputs the results")

row2_spacer1, row2_1, row2_spacer2 = st.beta_columns((.1, 3.2, .1))
with row2_1:
    st.write('Your data needs to look like the below - The columns names must be the same and the date format must be DD/MM/YYYY')

example_dict = {'OrderDateTime': ['31/05/2021','29/05/2021','05/05/2021'],
                'ItemCost': [5.6,7.7,10],
                'CustomerID': [1111,222222,333333],
                'OrderID' : ['abc1', 'efg3', 'hij4']
                }

example_data = pd.DataFrame(example_dict)

row3_spacer1, row3_1, row3_spacer2 = st.beta_columns((.1, 3.2, .1))
with row3_1:
    example_data 

@st.cache
def tidy_data(data):
    #clean up some of the data where needed
    data['OrderDate'] = pd.to_datetime(data['OrderDateTime']).dt.date
    #fill blank order IDs with 1
    data['OrderID'] = data['OrderID'].replace('nan', np.nan).fillna(1)
    data['ItemCost'] = pd.to_numeric(data['ItemCost'],errors='coerce')
    return data


row4_spacer1, row4_1, row4_spacer2 = st.beta_columns((.1, 3.2, .1))
with row4_1:
    user_input = st.file_uploader("Upload CSV",type=['csv'])

if not user_input:
        user_input = "C:\\Users\\bmussa\\Desktop\\Personal\\CB\\BI\BI.csv"

row5_spacer1, row5_1, row5_spacer2 = st.beta_columns((.1, 3.2, .1))
with row5_1:
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    trans_data = tidy_data(pd.read_csv(user_input,parse_dates=['OrderDateTime'], dayfirst=True))
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(trans_data)
        
        # Notify the reader that the data was successfully loaded.
    data_load_state.text('Loading data...done!')

#get max date from data series
max_date = trans_data['OrderDate'].max()
last_year = (datetime.strptime(str(trans_data['OrderDate'].max()),"%Y-%m-%d")+ timedelta(days=-365)).date()
last_year_1 = (datetime.strptime(str(trans_data['OrderDate'].max()),"%Y-%m-%d")+ timedelta(days=-730)).date()

row6_spacer1, row6_1, row6_spacer2 = st.beta_columns((.1, 3.2, .1))
with row6_1:
    if st.checkbox('Would you like to view some stats?'):
        st.subheader('Data stats')
        st.write('Max date in this data is:', max_date, )
        st.write('You have :', len(trans_data), 'many records' )
        st.write('You have :', trans_data['CustomerID'].nunique(), 'many unique customers' )
            
        st.subheader('Below is a summary of the data - item cost field only')
        data_description = trans_data['ItemCost'].agg(['count','mean', 'sum', 'min', 'max','median']).reset_index()
        st.write(data_description)

#%% select data from transdata for last 12 months on

@st.cache
def data_calcs(trans_data):
    last12data = trans_data[trans_data['OrderDate']>=last_year]
    #%% add months columns to last 12 monts
    last12data['months'] = pd.to_datetime(trans_data['OrderDate']).dt.strftime('%Y-%m')
    #%% pivot the data out and fill with 1 or 0
    last12months_pivot = pd.pivot_table(data = last12data, values='OrderID', index='CustomerID', columns='months', aggfunc='count')
    last12months_pivot = last12months_pivot.replace('nan', np.nan).fillna(0)
    #%% code it up so that there is only 1 or 0
    list_of_cols = list(last12months_pivot.columns)
    last12months_pivot= pd.DataFrame(last12months_pivot)
    for col in list_of_cols:
        last12months_pivot[col] = last12months_pivot[col].apply(lambda x: 1 if x>0 else 0).astype(np.uint8)
        #%% create aggregated dataset
    agg_data = trans_data.groupby(['CustomerID']).agg({'ItemCost': ['sum']
                                                      ,'OrderID': ['count']
                                                      ,'OrderDate': ['min','max']}
                                                      ).reset_index()
    
    agg_data.columns = ['CustomerID', 'TotalSpend', 'TotalOrders', 'FirstOrder','LastOrder']
    
    agg_data['ATV'] = agg_data['TotalSpend']/agg_data['TotalOrders']
    
    agg_data['AvgTimePOrder'] = ((agg_data['LastOrder'] -agg_data['FirstOrder'])/np.timedelta64(1,'D'))/agg_data['TotalOrders']
    
    #agg_data[agg_data['FirstOrder'].between(last_year, max_date)]
        
    condlist = [(agg_data.FirstOrder>=last_year),            
                (agg_data.LastOrder>=last_year)&(agg_data.FirstOrder<last_year_1),
                (agg_data.FirstOrder<last_year)&(agg_data.LastOrder>=last_year),
                (agg_data.LastOrder<last_year)
                ]
    
    choicelist = ['New'
                  , 'Reactivated'
                  , 'Active'
                  ,'Lapsed']
    
    agg_data['CustStatus'] = np.select(condlist, choicelist, default='unknown')
    
    condlist = [(agg_data.AvgTimePOrder<=7),            
                (agg_data.AvgTimePOrder<=30),
                (agg_data.AvgTimePOrder<=180),
                (agg_data.AvgTimePOrder<=365),
                (agg_data.AvgTimePOrder>365),
                ]
    
    choicelist = ['1. Weekly'
                  , '2. Monthly'
                  ,'3. Bi Annually'
                  , '4. Annually'
                  , '5. More than a year']
    
    agg_data['AvgTimepOrderBand'] = np.select(condlist, choicelist, default='unknown')
    
    condlist = [(agg_data.ATV<20),            
                (agg_data.ATV<50),
                (agg_data.ATV<=100),
                (agg_data.ATV<=250),
                (agg_data.ATV<=500),
                (agg_data.ATV>500),
                ]
    
    choicelist = ['1. < £20'
                  , '2. £20 - £50'
                  , '3. £50 - £100'
                  , '4. £100 - £250'
                  , '5. £250 - £500'
                  , '6. £500+']
    
    agg_data['ATVBand'] = np.select(condlist, choicelist, default='unknown')
    
    condlist = [(agg_data.TotalSpend<50),            
                (agg_data.TotalSpend<100),
                (agg_data.TotalSpend<200),
                (agg_data.TotalSpend<500),
                (agg_data.TotalSpend<1000),
                ]
    
    choicelist = ['1. < £50'
                  , '2. £50 - £100'
                  , '3. £100 - £200'
                  , '4. £200 - £500'
                  , '5. £500 - £1000']
    
    agg_data['TotalSpendBand'] = np.select(condlist, choicelist, default='6. More than £1000')
    
    #%%merge last 12 month pivot onto aggregated data set
    
    agg_data = pd.merge(agg_data, last12months_pivot, how='left', left_on='CustomerID', right_on='CustomerID')
    
    #%% last 12 months spend
    
    last12data_spend = last12data.groupby(last12data['CustomerID']).agg({'ItemCost': ['sum']}).reset_index()
    
    last12data_spend.columns = ['CustomerID', 'Spend12m']
    
    #%% last12m spend back onto merged df
    agg_data = agg_data.merge(last12data_spend,how='left', on='CustomerID')
    
    #%% months spent
    agg_data['monthsSpent'] = agg_data[list_of_cols].sum(axis=1)
    
    #%% loyalty band
    
    agg_data['monthsSpent'] = agg_data['monthsSpent'].replace('nan', np.nan).fillna(0)
    
    condlist = [(agg_data.CustStatus=='New'),
                (agg_data.monthsSpent>9),            
                (agg_data.monthsSpent>3),
                ((agg_data.monthsSpent<=3) & (agg_data.monthsSpent>0)),
                ]
    
    choicelist = ['0. New Customer'
                  ,'1. High Loyal'
                  , '2. Med Loyal'
                  , '3. Low Loyal'
                  ]
    
    agg_data['LoyaltyBand'] = np.select(condlist, choicelist, default='4. No Spender')
    
    #%% create column for last 12m spend
    condlist = [(agg_data.Spend12m<250),            
                (agg_data.Spend12m<1000),
                (agg_data.Spend12m>1000),            
                ]
    
    choicelist = ['1. < £250'
                  , '2. £250 - £1000'
                  , '3. >£1000'
                  ]
    
    agg_data['Last12mSpendBand'] = np.select(condlist, choicelist, default='7. No Spender')
    return agg_data

agg_data = data_calcs(trans_data)

#%% charts for analysis

#width = st.sidebar.slider("plot width", 1, 25, 3)
#height = st.sidebar.slider("plot height", 1, 25, 1)

#fig, ax = plt.subplots(figsize=(width, height)) #solved by add this line 
#ax = agg_data[agg_data['CustomerID']!=0][['LoyaltyBand']].value_counts().sort_index().plot.bar()

#st.subheader('Number of customers by loyalty band')
#plt.show()
#st.pyplot(fig)

row7_spacer1, row7_1,row7_spacer2 = st.beta_columns((.1, 3.2, .1))
with row7_1:
    option = st.selectbox(
     'Which field would you like to cut the data by?',
     ('LoyaltyBand', 'TotalSpendBand', 'CustStatus','Last12mSpendBand','AvgTimepOrderBand','ATVBand'))
    st.write('You selected:', option)

row8_spacer1, row8_1, row8_spacer2 = st.beta_columns((.1, 3.2, 0.1))
with row8_1, _lock:
    data_cut = agg_data[agg_data['CustomerID']!=0]
    data_cut = agg_data.groupby([option]).agg({'CustomerID': ['count']
                                               ,'TotalSpend': ['sum']
                                               ,'TotalOrders': ['sum']
                                               ,'Spend12m': ['sum']
                                               }).sort_index().reset_index()
    data_cut.columns = [option
                        , 'Counts'
                        ,'Total Spend'
                        , 'Total Orders'
                        , 'Total Spend Last12m'
                        ]

    data_cut['Avg Spend']= data_cut['Total Spend']/data_cut['Counts']
    data_cut['Avg Orders']= data_cut['Total Orders']/data_cut['Counts']
    data_cut['Avg Spend Last12m']= data_cut['Total Spend Last12m']/data_cut['Counts']
    data_cut['ATV']= data_cut['Total Spend']/data_cut['Total Orders']
    
    data_cut
if st.checkbox('Would you like to see the charts?'):
    st.header('Below are charts of each metric by the selected dimension')
    row9_spacer1, row9_1, row9_2, row9_spacer2 = st.beta_columns((.1, 3.2,3.2, 0.1))
    with row9_1, _lock:
        st.subheader('Count of customers by '+ option)
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(data = data_cut, x = data_cut[option], y=data_cut['Counts'],
                    color="goldenrod", ax=ax)
        ax.set_ylabel('Counts')
        ax.set_xlabel(option)
        ax.label_outer()
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        st.pyplot(fig)
    with row9_2, _lock:
        st.subheader('Total Spend of customers by '+ option)
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(data = data_cut, x = data_cut[option], y=data_cut['Total Spend'],
                    color="goldenrod", ax=ax)
        ax.set_ylabel('Total Spend')
        ax.set_xlabel(option)
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        st.pyplot(fig)
    
    st.write('')
    
    row10_spacer1, row10_1,row10_2, row10_spacer2 = st.beta_columns((.1, 3.2,3.2, 0.1))
    with row10_1, _lock:
        st.subheader('Total Orders of customers by '+ option)
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(data = data_cut, x = data_cut[option], y=data_cut['Total Orders'],
                    color="goldenrod", ax=ax)
        ax.set_ylabel('Total Orders')
        ax.set_xlabel(option)
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        st.pyplot(fig)
    with row10_2, _lock:
        st.subheader('Avg Number of Orders by '+ option)
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(data = data_cut, x = data_cut[option], y=data_cut['Avg Orders'],
                    color="goldenrod", ax=ax)
        ax.set_ylabel('Avg Orders')
        ax.set_xlabel(option)
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        st.pyplot(fig)
    
    st.write('')
    
    row11_spacer1, row11_1,row11_2, row11_spacer2 = st.beta_columns((.1, 3.2,3.2, 0.1))
    with row11_1, _lock:
        st.subheader('Avg Spend of customers by '+ option)
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(data = data_cut, x = data_cut[option], y=data_cut['Avg Spend'],
                    color="goldenrod", ax=ax)
        ax.set_ylabel('Avg Spend')
        ax.set_xlabel(option)
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        st.pyplot(fig)
    with row11_2, _lock:
        st.subheader('Total Spend of customers by '+ option)
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(data = data_cut, x = data_cut[option], y=data_cut['Total Spend Last12m'],
                    color="goldenrod", ax=ax)
        ax.set_ylabel('Total Spend')
        ax.set_xlabel(option)
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        st.pyplot(fig)
    
    st.write('')
    
    row12_spacer1, row12_1,row12_2, row12_spacer2 = st.beta_columns((.1, 3.2,3.2, 0.1))
    with row12_1, _lock:
        st.subheader('ATV of customers by '+ option)
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(data = data_cut, x = data_cut[option], y=data_cut['ATV'],
                    color="goldenrod", ax=ax)
        ax.set_ylabel('ATV')
        ax.set_xlabel(option)
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        st.pyplot(fig)
    with row12_2, _lock:
        st.subheader('Avg Spend Last12m of customers by '+ option)
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(data = data_cut, x = data_cut[option], y=data_cut['Avg Spend Last12m'],
                    color="goldenrod", ax=ax)
        ax.set_ylabel('Avg Spend Last12m')
        ax.set_xlabel(option)
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        st.pyplot(fig)
    
    st.write('') 
    
#%% final aggregated dataset

row13_spacer1, row13_1, row13_spacer2 = st.beta_columns((0.1, 3.2, .1))
with row13_1:
    if st.checkbox('Show final aggregated dataset'):
        st.subheader('Aggregated data')
        st.write(agg_data)


#%% download file
def get_table_download_link_csv(df):
    csv = df.to_csv(index=False).encode('utf-8-sig')
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="agg_data.csv" target="_blank">Download aggregated data as a csv file</a>'
    return href

st.markdown(get_table_download_link_csv(agg_data), unsafe_allow_html=True)