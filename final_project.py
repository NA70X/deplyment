#!/usr/bin/env python
# coding: utf-8

# # Dataset Details

# ### Columns Mapping
# 
# InvoiceNo: Unique identifier for each retail invoice or transaction.  
# StockCode: Code for the specific product or item being sold.  
# Description: Name of the product or item being sold.  
# Quantity: Number of units of the product purchased in each transaction.  
# InvoiceDate: Date and time of each retail transaction.  
# UnitPrice: Price per unit of the product being sold.  
# CustomerID: Unique identifier for each customer who made a purchase.  
# Country: Name of the country where the customer is located.  

# ### Quick info
# The Data talks about Online Retail business, Our columns in the data are InvoiceNo	StockCode	Description	Quantity	InvoiceDate	UnitPrice	CustomerID	Country
# which are explained above

# ### Our goal is:
# -  deep cleaning and analysis to our data and answering related business questions helped by visualizations and drawings.
# -  using statistical methods and unsupervised machine learning algorithm to segment customers into clusters, and help the business make business-decisions related to each cluster
# -  deploying the model and making an API that can be interacted by the user and segmenting new customers based on different data.

# # Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# # Importing Data

# In[2]:


data=pd.read_excel("Online Retail.xlsx")
data.head()


# In[3]:


data.info()


# # Data Cleaning

# In[4]:


data.isnull().sum()
#CustomerID=24% ...Description 0.3%


# In[5]:


data.fillna(method='ffill',inplace=True)


# In[6]:


num_duplicates = data.duplicated().sum()
print(num_duplicates)


# In[7]:


data = data.drop_duplicates()
data.duplicated().sum()


# In[8]:


delete = ['amazon adjust', 'Discount', 'amazon', '?', 'found', 'counted', 'Given away', 'Dotcom', 'label mix up', 'thrown away'
, 'Adjustment', 'AMAZON FEES', 'wrongly sold as sets', 'Amazon sold sets', 'dotcom sold sets' , 'wrongly sold sets', '? sold as sets?', '?sold as sets?',
'Thrown away', 'damages/display', 'damaged stock', 'broken', 'throw away', 'wrong barcode (22467)', 'wrongly sold (22719) barcode', 'wrong barcode',
'barcode problem', '?lost', "thrown away-can't sell.", "thrown away-can't sell'", 'damages?', 're dotcom quick fix.', 'sold in set?', 'cracked sold as 22467',
'Damaged', 'DAMAGED', 'damaged', 'did  a credit  and did not tick ret', 'adjustment', 'returned', 'wrong code?', 'wrong code', 'crushed',
'damages/showroom etc', 'samples', 'damages/credits from ASOS.', 'damages/dotcom?', 'wet/rusty', 'incorrectly credited C550456 see 47', 'wet damaged',
'missing', 'sold as set on dotcom', 'water damage', 'to push order througha s stock was', 'found some more on shelf', 'Show Samples', 'FOUND', 'mix up with c',
'mouldy, unsaleable.', 'wrongly marked. 23343 in box', 'came coded as 20713', 'alan hodge cant mamage this section', 'dotcom', 'stock creditted wrongly', 'ebay',
'incorrectly put back into stock', 'Damages/samples', 'Crushed', 'taig adjust', 'allocate stock for dotcom orders ta', 'Amazon', 'found box',
'OOPS ! adjustment', 'Found in w/hse', 'website fixed', 'Dagamed', 'wrongly coded', 'stock check', 'crushed boxes', "can't find", 'mouldy', 'Sale error',
'Breakages', 'Marked', 'Damages', 'CHECK', 'Unsaleable, destroyed.', 'marked', 'damages', 'damaged', 'Wrongly', 'wrong', 'wet', 'rusty', 'lost', 'faulty', 'cracked',
'sold as 22467', "Dotcom sold in 6's", 'Missing', 'Adjust bad debt', 'taig adjust no stock', 'WET/MOULDY', 'wrongly coded 20713', 'wrongly coded-23343',
'Marked as 23343', '20713', 'wrongly coded 23343', 'wrongly marked', 'dotcom sales' , 'had been put aside', 'damages wax', 'water damaged',
'Wrongly mrked had 85123a in box', 'wrongly marked carton 22804', 'missing?', 'wet rusty', 'dotcom adjust', 'rusty thrown away', 'rusty throw away', 'check?',
'???lost', 'dotcomstock','?? missing', 'wet pallet', '????missing', '???missing', 'AMAZON','wet?',
'lost??','???', 'wet boxes', '????damages????', 'mixed up', 'lost in space', 'Water damaged', 'smashed', '??', "thrown away-can't sell", 'Thrown away.',
'DOTCOM POSTAGE', 'Dotcom sales', 'Dotcomgiftshop Gift Voucher £40.00', 'Dotcomgiftshop Gift Voucher £30.00', 'Dotcomgiftshop Gift Voucher £20.00'
'Dotcom set', 'Dotcomgiftshop Gift Voucher £50.00', 'Dotcomgiftshop Gift Voucher £10.00', 'sold as set on dotcom and amazon', 'sold as set on dotcom',
'sold as set/6 by dotcom', 'Sold as 1 on dotcom', 'Dotcomgiftshop Gift Voucher £100.00', 'sold with wrong barcode', 're-adjustment',
'Amazon Adjustment','wrongly marked 23343','20713 wrongly marked', 'test','temp adjustment', 'code mix up? 84930', '?display?', 'sold as 1','?missing', 'crushed ctn'
'CRUK Commission', 'amazon sales', 'mouldy, thrown away.', 'AMAZON FEE']


# In[9]:


data.drop(data[data['Description'].isin(delete)].index, inplace=True)
data.reset_index(drop=True, inplace=True)


# In[10]:


data
#541909 ---> 401604------25% has been filled


# # Converting data and Feature Engineering

# In[11]:


#replacing negative values with its abs values
data['Quantity'] = data['Quantity'].abs()


# In[12]:


data['InV_Year'] = data.InvoiceDate.dt.year
data['InV_Month'] = data.InvoiceDate.dt.month
data['InV_Day'] = data.InvoiceDate.dt.day
data['WDay'] = data['InvoiceDate'].dt.weekday
weekday_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
data['WeekDay'] = data['WDay'].map(weekday_names)


# In[13]:


data['Profit'] = data['Quantity'] * data['UnitPrice']


# In[14]:


data.head()
#For visualisation


# In[15]:


data.info()


# In[16]:


#country, StockCode is dropped
df=data[['Description', 'Quantity',
       'UnitPrice', 'CustomerID', 'InvoiceDate', 'InvoiceNo', 'Profit']]


# **Making Features out of the Data**

# In[17]:


#aim is to make it easier for modeling if not needed please delete it..!
df1 =  data.groupby('CustomerID').agg({'InvoiceDate': lambda x: (data['InvoiceDate'].max() - x.max()).days,
                                            'InvoiceNo': 'count',
                                            'Profit': 'sum'})
df1 = df1.rename(columns={'InvoiceDate':'Recency','InvoiceNo':'Frequency','Profit':'Monetary'})
#Recency is diffirence between 1st and last purchase day..
#Monetary is summation of totalprice for one customer..
#Frequency is the count of orders done by the customer..


# In[18]:


df_viz =  data.groupby(['CustomerID','Country']).agg({'InvoiceDate': lambda x: (data['InvoiceDate'].max() - x.max()).days,
                                            'InvoiceNo': 'count',
                                            'Profit': 'sum'})
df_viz['Country'] = df_viz.index.get_level_values('Country')
df_viz.index = df_viz.index.get_level_values('CustomerID')
df_viz = df_viz.sort_values('InvoiceNo',ascending=False).rename(columns={"InvoiceDate":"Days since last purchase","InvoiceNo":"Count of Orders","Profit":"Sum of Profit"})


# In[19]:


df1.head()


# In[20]:


df_viz.head()


# # Outliers Detection and Removal

# In[21]:


df1.plot(kind="box", subplots=True, figsize=(100,100), layout=(50,50))


# In[22]:


df.plot(kind="box", subplots=True, figsize=(100,100), layout=(50,50))


# In[23]:


for col in (df_viz.select_dtypes(include=np.number)).columns:
    q1 = df_viz[col].quantile(0.25)
    q3 = df_viz[col].quantile(0.75)
    iqr = q3-q1
    upper_limit = q3 + (1.5 * iqr)
    lower_limit = q1 - (1.5 * iqr)
    df_viz=df_viz.loc[(df_viz[col] <= upper_limit) & (df_viz[col] >= lower_limit)]


# In[24]:


for col in df1.columns:
    q1 = df1[col].quantile(0.25)
    q3 = df1[col].quantile(0.75)
    iqr = q3-q1
    upper_limit = q3 + (1.5 * iqr)
    lower_limit = q1 - (1.5 * iqr)
    df2=df1.loc[(df1[col] <= upper_limit) & (df1[col] >= lower_limit)]
    print('before removing outliers:', len(df1))
print('after removing outliers:',len(df2))
print('outliers:', len(df1)-len(df2))


# In[25]:


df_numerics_only = df.select_dtypes(include=np.number)

for col in df_numerics_only:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3-q1
    upper_limit = q3 + (1.5 * iqr)
    lower_limit = q1 - (1.5 * iqr)
    df3=df.loc[(df[col] <= upper_limit) & (df[col] >= lower_limit)]
print('before removing outliers:', len(data))
print('after removing outliers:',len(df3))
print('outliers:', len(data)-len(df3))


# # Visualization and Business Questions

# - What are our 5 most selling products?
# - What are our 5 most profitable products?
# - Our top 10 Profitable countries? 
# - Our top 10 countries with high demands? 
# - Top 5 Products with High Demand?
# - Top 5 Countries with more customers?
# - Top 5 Countries with less customers?
# - What is the highest profitable month?
# - What is the month with most demands?
# - Do Frequent Customers bring much profit?

# In[26]:


df_viz = df_viz.sort_values("Sum of Profit", ascending=False)
df_viz.head()


# In[27]:


by_profit = data.groupby("Country")["Profit"].sum().reset_index().sort_values('Profit', ascending=False).reset_index(drop=True)
by_orders = data.groupby("Country")["InvoiceNo"].count().reset_index().sort_values('InvoiceNo', ascending=False).reset_index(drop=True).rename(columns={'InvoiceNo':'Count of Orders'})
viz = by_profit.head(11)
viz2 = by_orders.head(11) 
viz_wuk_profit = by_profit.drop(index=0).head(10)
viz_wuk_order = by_orders.drop(index=0).head(10)


# In[28]:


viz3 = data.groupby("Country")['CustomerID'].count().reset_index().sort_values('CustomerID', ascending=False)
viz3 = viz3.rename(columns={'CustomerID':'No. of Customers'}).reset_index(drop=True)
least_customers = viz3.tail(5)
high_customers = viz3.head(5)


# In[29]:


viz4 = data.groupby(['Description','UnitPrice'])['InvoiceNo'].count().reset_index().sort_values('InvoiceNo', ascending=False)
viz4_1 = data.groupby(['Description','UnitPrice'])['Profit'].sum().reset_index().sort_values('Profit', ascending=False)
viz4 = viz4.rename(columns={'InvoiceNo':'Count of Orders'}).reset_index(drop=True)
high_order = viz4.head(6)
high_profit = viz4_1.head(6)


# In[30]:


monthly_profit = data.groupby(['InV_Month', 'InV_Year'])['Profit'].sum().reset_index()
highest_profit_month = monthly_profit.loc[monthly_profit['Profit'].idxmax()]


# In[31]:


monthly_demand = data.groupby(['InV_Month','InV_Year'])['Quantity'].sum().reset_index()
highest_demand_month = monthly_demand.loc[monthly_demand['Quantity'].idxmax()]


# In[32]:


product_quantity_sold = df.groupby('Description')['Quantity'].sum().reset_index()
product_quantity_sold_sorted = product_quantity_sold.sort_values(by='Quantity', ascending=False)
top_5_selling_products = product_quantity_sold_sorted.head(5)


# In[33]:


# sns.pairplot(df_viz)


# In[34]:


# fig = px.scatter(df_viz, x="Days since last purchase", y="Sum of Profit",
# 	         size="Count of Orders", color="Country", size_max=60, log_y=True, title="Decoding Online Retail Success: Is it about Recency, Monetary, or Frequency?")
# fig.show()
# # Model idea introduction


# In[35]:


# fig = px.scatter(df_viz, x="Sum of Profit", y="Count of Orders",
# 	         size="Count of Orders", color="Country", size_max=60, log_y=True, title="Decoding Online Retail Success: Is it about Recency, Monetary, or Frequency?")
# fig.show()


# In[36]:


# # Top 10 Profitable Countries
# fig = px.bar(viz, x='Country', y='Profit', log_y=True, title='Top 10 Profitable Countries')
# fig.show()


# In[37]:


# # Top 10 Profitable Countries(w/UK)
# fig = px.bar(viz_wuk_profit, x='Country', y='Profit', title='Top 10 Profitable Countries (UK excluded)')
# fig.show()


# In[38]:


# #Top 10 countries with highest demands
# fig = px.bar(viz2, x='Country', y="Count of Orders", log_y=True, title='Top 10 Countries with Highest demand')
# fig.show()


# In[39]:


# # Top 10 Countries with Highest demand(w/UK)
# fig = px.bar(viz_wuk_order, x='Country', y='Count of Orders', title='Top 10 Countries with Highest demand (UK excluded)')
# fig.show()


# In[40]:


# fig = px.bar(least_customers, x='Country', y='No. of Customers', title='Top 5 Countries with Least number of Customers')
# fig.show()


# In[41]:


# fig = px.bar(high_customers, x='Country', y='No. of Customers', log_y=True, title='Top 5 Countries with highest number of Customers')
# fig.show()


# In[42]:


# fig = px.bar(high_order, x='Count of Orders', y='Description', title='Top 5 Products with High Demand', orientation='h', hover_data=["UnitPrice"])
# fig.show()


# In[43]:


# fig = px.bar(top_5_selling_products, x='Quantity', y='Description',title='Top 5 Most Selling Products',orientation='h', hover_data=["Quantity"])
# fig.show()


# In[44]:


# fig = px.bar(high_profit, x='Profit', y='Description', title='Top 5 Profitable Products', orientation='h', hover_data=["UnitPrice"])
# fig.show()


# In[45]:


# fig = px.bar(monthly_profit, x='InV_Month', y='Profit', title='Total Profit by month', hover_data='InV_Year')
# fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',marker_line_width=1.5, opacity=0.6)
# fig.update_layout(xaxis_title="month", yaxis_title="Total Profit")
# fig.show()


# In[46]:


# fig = px.bar(monthly_demand, x='InV_Month', y='Quantity', title='Total Demand by Month', hover_data='InV_Year')
# fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',marker_line_width=1.5, opacity=0.6)
# fig.update_layout(xaxis_title="Month", yaxis_title="Total Quantity Sold")
# fig.show()


# # Encoding

# In[47]:


df.head()


# In[48]:


data['Description'] = data['Description'].astype(str)
le=LabelEncoder()
df['Description']=le.fit_transform(data['Description'])
df.dtypes


# In[49]:


#data after encoding
df.head()


# **Scaling**

# In[50]:


scale = StandardScaler()
df_scaled = scale.fit_transform(df2)
df_scaled = pd.DataFrame(df_scaled, columns=df2.columns)


# In[51]:


df_scaled.head()


# ## Modeling

# **Feature Extraction using PCA**

# In[52]:


import matplotlib.pyplot as plt

def pca_plot(cumulative_explained_variance, df):
    """
    Plot the cumulative explained variance to select the number of components.

    Parameters:
    cumulative_explained_variance (array): Cumulative explained variance ratio
    dfk (pandas dataframe): Original data
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.show()


# In[54]:


from sklearn.decomposition import PCA

# Initialize PCA with the number of components equal to the number of scaled features
pca = PCA(n_components=df_scaled.shape[1])

# Fit PCA to the scaled data
pca.fit(df_scaled)

# Calculate the explained variance and cumulative explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

# Visualize the explained variance to select the number of components
pca_plot(cumulative_explained_variance, df_scaled)


# In[55]:


pca = PCA(n_components=2)
pca.fit(df_scaled)


# In[56]:


scores_pca = pca.transform(df_scaled)


# In[57]:


scores_pca


# **Selecting K number of Clusters**

# In[58]:


from sklearn.cluster import KMeans
sum_of_square_distance=[]
for k in range(1,11):
    km=KMeans(n_clusters =k,init="k-means++",max_iter=300, random_state=33)
    km=km.fit(scores_pca)
    sum_of_square_distance.append(km.inertia_)


# In[ ]:


# plt.figure(figsize=(10,7))
# plt.plot(range(1,11),sum_of_square_distance)
# plt.title('Elbow Method')
# plt.xlabel('Number of Clusters')
# plt.ylabel("Sum of Squared Distance")


# #### From previous graph the expected number of clusters is 3
# 

# **Building the ML model using K-means Algorithm**

# In[59]:


Model = KMeans (n_clusters=3,
                init='k-means++',
                max_iter=300)
Model.fit(scores_pca)


# In[60]:


dfk = df2.copy()
dfk["Cluster"] = Model.labels_

dfk.head()


# In[61]:


df_clustered = dfk.copy() 
df_clustered['CustomerID'] = df_clustered.index.get_level_values('CustomerID')
df_clustered = df_clustered.reset_index(drop=True)

df_clustered.head()


# In[62]:


dfk.groupby('Cluster').agg({
    'Recency':'mean',
    'Frequency':'mean',
    'Monetary':['mean', 'count']}).round(2)


# In[63]:


from sklearn.metrics import silhouette_score

preds = Model.fit_predict(scores_pca)
silhouette = silhouette_score(scores_pca, preds)
print("Silhouette Coefficient:", silhouette)


# **Visualizing Clusters**

# In[ ]:


# import plotly.express as px

# # Create a 3D scatter plot matrix with Plotly
# fig = px.scatter_3d(dfk, x='Recency', y='Frequency', z='Monetary', color='Cluster')

# fig.update_layout(
#     scene=dict(
#         xaxis_title='Recency',
#         yaxis_title='Frequency',
#         zaxis_title='Monetary',
#     ),
#     title='Clustering by Recency, Frequency, and TotalPrice',
#     width=800,
#     height=600
# )

# fig.show()


# In[ ]:


# # Visualising the Cluster Chartecteristics

# df_normalized = pd.DataFrame(df_scaled, columns=['Recency', 'Frequency', 'Monetary'])
# df_normalized['ID'] = df2.index
# df_normalized['Cluster'] = Model.labels_
# # Melt The Data
# df_nor_melt = pd.melt(df_normalized.reset_index(),
#                       id_vars=['ID', 'Cluster'],
#                       value_vars=['Recency','Frequency','Monetary'],
#                       var_name='Attribute',
#                       value_name='Value')
# df_nor_melt.head()
# # Visualize it
# sns.lineplot(x='Attribute', y='Value', hue='Cluster', data=df_nor_melt)
# plt.show()


# We deduce from the chart that cluster (1) is loyal:
# * Monetary value is high --> They tend to ___spend more___
# * Frequency is high --> ___More orders___ is placed by them
# * Recency attribute is low --> ___less days___ between each purchase which means they bought the products ___more recently___
# 
# -they pay more, order more and they bought more recently, which means they are loyal to us and can be a good research data for making potential customers loyal.

# We deduce from the chart that cluster (0) is Potential Customer:
# * Monetary value is low --> They tend to ___spend less___
# * Frequency is low --> ___less orders___ is placed by them
# * Recency attribute is low --> ___less days___ between each purchase which means they bought the products ___more recently___
# 
# -they pay less, and order less but, they bought more recently which means we can convert them into loyal customers with good marketing and shopping deals.  
# -they are also a good research data for attracting customers to our business.

# We deduce from the chart that cluster (2) is Churn customers:
# * Monetary value is low --> They tend to ___spend less___
# * Frequency is low --> ___less orders___ is placed by them
# * Recency attribute is high --> ___more days___ between each purchase which means they bought the products ___a while ago___
# 
# -they pay less, order less and they are old customers, this cluster can be depercated.

# In[65]:


import pickle
pickle_out=open("Model.pkl","wb")
pickle.dump(Model,pickle_out)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




