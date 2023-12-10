#!/usr/bin/env python
# coding: utf-8

# In[1]:
from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns.fpgrowth import fpgrowth
import numpy as np
import pandas as pd
import random
import os
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime  


# In[2]:
current_directory = os.getcwd()
os.chdir('C:\\Users\\lokna\\OneDrive\\Desktop')


# In[3]:
df1 = pd.read_excel("Assignment-1_Data.xlsx")

# In[4]:
df1.describe


# In[5]:
df1

# In[7]:
seed_value = 44
random.seed(seed_value)

# # DATA CLEANING -

# In[8]:
Dataset1=df1


# In[9]:
print(Dataset1['Date'].dtype)

date_format = 'your_actual_date_format_here'
Dataset1['Date'] = pd.to_datetime(Dataset1['Date'], format=date_format)

# In[10]:
missing_values = Dataset1.isnull().sum()
print("Missing values in each variable of Dataset1:")
print(missing_values)


# In[11]:

Dataset1.dropna(inplace=True)

duplicate_count = df1.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

# Remove duplicate rows
Dataset1.drop_duplicates(inplace=True)

# Reset the index if needed
Dataset1.reset_index(drop=True, inplace=True)

# List of buzzwords to delete
buzzwords = ["WRONG", "LOST", "CRUSHED", "SMASHED", "DAMAGED", "FOUND", "THROWN", "MISSING",
             "AWAY", "\\?", "CHECK", "POSTAGE", "MANUAL", "CHARGES", "AMAZON", "FEE",
             "FAULT", "SALES", "ADJUST", "COUNTED", "LABEL", "INCORRECT", "SOLD", "BROKEN",
             "BARCODE", "CRACKED", "RETURNED", "MAILOUT", "DELIVERY", "MIX UP", "MOULDY",
             "PUT ASIDE", "ERROR", "DESTROYED", "RUSTY"]

# Filter out rows that contain buzzwords in the 'Itemname' column
Dataset1 = Dataset1[~Dataset1['Itemname'].str.contains('|'.join(buzzwords), case=False, na=False)]

# Reset the index if needed
Dataset1.reset_index(drop=True, inplace=True)

Dataset1

# Create the new column "TotalPrice"
Dataset1['TotalPrice'] = Dataset1['Quantity'] * Dataset1['Price']

# Filter Quantity > 0, Price > 0, and TotalPrice > 0
Dataset1 = Dataset1[(Dataset1['Quantity'] > 0) & (Dataset1['Price'] > 0) & (Dataset1['TotalPrice'] > 0)]

# Display the updated DataFrame
print(Dataset1)

# In[20]:
Dataset2 = Dataset1

# In[22]:
# Extract the Year and Month from the 'Date' column
Dataset2['Year'] = Dataset2['Date'].dt.year
Dataset2['Month'] = Dataset2['Date'].dt.month

# View the updated DataFrame
print(Dataset2.head())


# In[23]:
# Convert 'Date' column to datetime format
Dataset2['Date'] = pd.to_datetime(Dataset2['Date'])

# Create a new column 'Date1' containing only the date part
Dataset2['Date1'] = Dataset2['Date'].dt.date

# Print the updated DataFrame
print(Dataset2)

#####################uk and seasons######################


# In[24]:

# Filter the DataFrame to include only rows where 'Country' is 'United Kingdom'
Dataset2 = pd.DataFrame(Dataset2)
Dataset2 = Dataset2[Dataset2['Country'] == 'United Kingdom']

# Convert the 'Date' column to pandas datetime type
Dataset2['Date'] = pd.to_datetime(Dataset2['Date'])

# Define a function to get the season based on the month
def get_season(month):
    if 3 <= month <= 5:
        return 'Spring'
    elif 6 <= month <= 8:
        return 'Summer'
    elif 9 <= month <= 11:
        return 'Autumn'
    else:
        return 'Winter'

# Create the 'Season' column based on the month from the 'Date' column using .loc
Dataset2.loc[:, 'Season'] = Dataset2['Date'].dt.month.map(get_season)

# Print the updated DataFrame
print(Dataset2)

####################################### DESCRIPTIVE ANALYSIS ##########################################################


# In[27]:
#################TO CHECK THE TOP TRANSCATIONS BY COUNTRY #######################3
# Assuming Dataset2 is already loaded as a DataFrame
country_counts = Dataset1['Country'].value_counts()

print(country_counts)

import matplotlib.pyplot as plt

country_counts.plot(kind='bar', figsize=(12, 7))
plt.title('Counts of Each Country in Dataset2')
plt.xlabel('Country')
plt.ylabel('Count')
plt.show()


# In[28]:
# Group the data by year and calculate transaction counts for each year
yearly_transaction_counts = Dataset1.groupby('Year')['BillNo'].count()

print(yearly_transaction_counts)

yearly_transaction_counts.plot(kind='bar', figsize=(12, 7))
plt.title('Number of Transactions in Each Year')
plt.xlabel('Year')
plt.ylabel('Transaction Count')
plt.show()


# In[50]:
# Create a line plot
plt.figure(figsize=(12, 7))
plt.plot(Sales_weekly.index, Sales_weekly.values, linewidth=2)
plt.title('Number of Sales Weekly')
plt.xlabel('Date')
plt.ylabel('Number of Sales')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[49]:
# Create a line plot
plt.figure(figsize=(12, 7))
plt.plot(Unique_customer_weekly.index, Unique_customer_weekly.values, linewidth=2)
plt.title('Number of Customers Weekly')
plt.xlabel('Date')
plt.ylabel('Number of Customers')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[48]:
# Calculate the sales per customer ratio
Sales_per_Customer = Sales_weekly / Unique_customer_weekly

# Create a line plot
plt.figure(figsize=(12, 7))
plt.plot(Sales_per_Customer.index, Sales_per_Customer.values, linewidth=2)
plt.title('Sales per Customer Weekly')
plt.xlabel('Date')
plt.ylabel('Sales per Customer Ratio')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[46]:
# Group by Season and ItemName, then get the size/count of each group
grouped_counts = Dataset2.groupby(['Season', 'Itemname']).size().reset_index(name='Count')

# Get the top 10 products for each season
top_10_by_season = grouped_counts.groupby('Season').apply(lambda x: x.nlargest(10, 'Count')).reset_index(drop=True)

# Visualization
fig, ax = plt.subplots(figsize=(15, 7))
for season in top_10_by_season['Season'].unique():
    subset = top_10_by_season[top_10_by_season['Season'] == season]
    ax.bar(subset['Itemname'], subset['Count'], label=season)

ax.set_title('Top 10 Purchased Products in Different Seasons')
ax.set_xlabel('Product Name')
ax.set_ylabel('Count')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[45]:
# Group by month and count the items sold
monthly_sales = Dataset1.groupby(pd.Grouper(key='Date', freq='M'))['Itemname'].count()

# Plotting the monthly sales
plt.figure(figsize=(20, 8))
plt.grid(True)
plt.plot(monthly_sales.index, monthly_sales.values, marker='o', linestyle='-')
plt.title("Number of Items Sold by Month")
plt.xlabel("Date")
plt.ylabel("Number of Items Sold")
plt.show()


# In[44]:
Dataset3=Dataset2


# In[51]:
Dataset3
season_spring = Dataset3[Dataset3['Season'] == 'Spring'] #68494 
season_summer = Dataset3[Dataset3['Season'] == 'Summer'] #69002 
season_autumn = Dataset3[Dataset3['Season'] == 'Autumn'] #134671 
season_winter = Dataset3[Dataset3['Season'] == 'Winter'] #73577


# In[52]:
# Group by 'Itemname' and count the frequency of each item
Frequency_of_items = Dataset3.groupby('Itemname').size().reset_index(name='count')

# Create a treemap plot
fig = px.treemap(Frequency_of_items, path=['Itemname'], values='count', labels={'count': 'Frequency'})
fig.update_layout(title='Frequency of the Items Sold', title_x=0.5, title_font=dict(size=18))
fig.update_traces(textinfo="label+value")
fig.show()


# In[53]:
# Group by 'Itemname' and count the frequency of each item
Frequency_of_items = Dataset3.groupby('Itemname').size().reset_index(name='count')

# Sort the DataFrame by count in descending order and select top 50
Frequency_of_items = Frequency_of_items.sort_values(by='count', ascending=False).head(50)

# Generate the treemap
fig = px.treemap(Frequency_of_items, path=['Itemname'], values='count', labels={'count': 'Frequency'})
fig.update_layout(title='Frequency of the Top 50 Items Sold', title_x=0.5, title_font=dict(size=18))
fig.update_traces(textinfo="label+value")
fig.show()


# In[54]:

# Grouping data to create cleandata1
cleandata1 = Dataset3.groupby(['CustomerID', 'Date']).BillNo.nunique().reset_index(name='count_order')

# Further grouping data to create cleandata2
cleandata2 = cleandata1.groupby('CustomerID').agg(
    last_order_date=('Date', 'max'),
    transaction_number=('Date', 'size'), 
    count_order_sum=('count_order', 'sum')
).reset_index()

# Printing the resultant DataFrames

print(cleandata2)

# In[55]:
# Create snapshot date
snapshot_date = cleandata2['last_order_date'].max() + datetime.timedelta(days=1)
print(snapshot_date)


# In[56]:
# Create snapshot date
snapshot_date = cleandata2['last_order_date'].max() + datetime.timedelta(days=1)

# Aggregate data by each customer
customers = cleandata2.groupby(['CustomerID']).agg({
    'last_order_date': lambda x: (snapshot_date - x.max()).days,
    'count_order_sum': 'sum',
    'transaction_number': 'sum'
})

# Rename columns
customers.rename(columns={'last_order_date': 'Recency',
                          'count_order_sum': 'MonetaryValue',
                          'transaction_number': 'Frequency'}, inplace=True)

# Reset the index to make 'CustomerID' a regular column
customers.reset_index(inplace=True)

# Display the resulting DataFrame
print(customers)


# In[62]:
#distrubution analysis for recency
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

fig, ax = plt.subplots()
sns.histplot(data=customers, x='Recency', kde=True)
ax.set_title('Distribution of Recency')

customers_fix = pd.DataFrame()
customers_fix["Recency"] = customers['Recency']
plt.figure(figsize=(12,10))


# In[63]:
# Create a histogram with KDE plot
plt.figure(figsize=(12, 6))
sns.histplot(data=customers, x='Frequency', kde=True)
plt.title('Distribution of Frequency')
plt.xlabel('Frequency')
plt.ylabel('Frequency Count')

plt.show()


# In[64]:
#distrubution analysis for monetaryvalue
fig, ax = plt.subplots()
sns.histplot(data=customers, x='MonetaryValue', kde=True)
ax.set_title('Histogram and MonetaryValue')


# In[65]:
#Box-cox transformation
customers_fix["Frequency"] = stats.boxcox(customers['Frequency'])[0]
customers_fix["MonetaryValue"] = stats.boxcox(customers['MonetaryValue'])[0]


# Plot distributions after transformation
plt.subplot(3, 1, 1); sns.distplot(customers_fix['Recency'])
plt.subplot(3, 1, 2); sns.distplot(customers_fix['Frequency'])
plt.subplot(3, 1, 3); sns.distplot(customers_fix['MonetaryValue'])

#Before rfm analysis put everything into normalized, same mean variance 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(customers_fix)
customers_normalized = scaler.transform(customers_fix)

# Assert that it has mean 0 and variance 1
print(customers_normalized.mean(axis = 0).round(2)) # [0. -0. -0.]
print(customers_normalized.std(axis = 0).round(2)) # [1. 1. 1.]


# In[67]:

# RFM segmentation
cleandata2['rfm_recency'] = pd.qcut(cleandata2['last_order_date'], q=4, labels=False, duplicates='drop') + 1
cleandata2['rfm_frequency'] = pd.qcut(cleandata2['transaction_number'], q=4, labels=False, duplicates='drop') + 1
cleandata2['rfm_monetary'] = pd.qcut(cleandata2['count_order_sum'], q=4, labels=False, duplicates='drop') + 1

rfm_segment = cleandata2[['CustomerID', 'rfm_recency', 'rfm_frequency', 'rfm_monetary']]

def rfm_level(rfm_segment):
  if((rfm_segment['rfm_recency']>= 4) and (rfm_segment['rfm_frequency']>=4) 
      and (rfm_segment['rfm_monetary']>= 4)): 
      return 'Best Customers'
  elif ((rfm_segment['rfm_recency']>= 3) and (rfm_segment['rfm_frequency']>= 3) 
      and (rfm_segment['rfm_monetary']>= 3)): 
      return 'Loyal'
  elif ((rfm_segment['rfm_recency']>= 3) and (rfm_segment['rfm_frequency']>= 1)
      and (rfm_segment['rfm_monetary']>= 2)):
      return 'Potential Loyalist'
  elif ((rfm_segment['rfm_recency']>= 3) and (rfm_segment['rfm_frequency']>= 1)
      and (rfm_segment['rfm_monetary']>= 1)):
      return 'Promising'
  elif ((rfm_segment['rfm_recency']>= 2) and (rfm_segment['rfm_frequency']>= 2)
      and (rfm_segment['rfm_monetary']>= 2)):
      return 'Customers Needing Attention'
  elif ((rfm_segment['rfm_recency']>= 1) and (rfm_segment['rfm_frequency']>= 2)
      and (rfm_segment['rfm_monetary']>= 2)):
    return 'At Risk'
  elif ((rfm_segment['rfm_recency']>= 1) and (rfm_segment['rfm_frequency']>= 1)
      and (rfm_segment['rfm_monetary']>= 2)):
    return 'Hibernating'
  else:
    return 'Lost'

rfm_segment['rfm_level'] = rfm_segment.apply(rfm_level, axis=1)

# Calculating total customers in each segment
rfm_agg = rfm_segment.groupby('rfm_level').agg({'CustomerID':'count'})
print(rfm_agg)

# In[68]:
# Data
data = {
    'Customers ': ['At Risk', 'Customers Needing Attention', 'Lost', 'Loyal', 'Potential Loyalist', 'Promising'],
    'No of customers': [125, 391, 1442, 675, 629, 654]
}
rfm_agg = pd.DataFrame(data).set_index('Customers ')

# Extract data for plotting
labels = rfm_agg.index
sizes = rfm_agg['No of customers']
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFD700', '#FF6347']
explode = (0.1, 0, 0, 0, 0, 0)  # explode 1st slice for emphasis if you want

# Plot
plt.figure(figsize=(12, 7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distribution of Customers across RFM Levels')
plt.show()

# In[69]:

# Initialize an empty dictionary to store SSE values
sse = {}

# Define a range of K values to test
k_values = range(1, 11)

# Perform K-Means clustering for each K value and calculate SSE
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(customers_normalized)
    sse[k] = kmeans.inertia_  # SSE to closest cluster centroid

# Create the Elbow Method plot
plt.figure(figsize=(10, 6))
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Distances (SSE)')
plt.plot(list(sse.keys()), list(sse.values()), marker='o')
plt.grid(True)
plt.show()

# In[70]:

# K-Means clustering with K=5 and random_state=0
model = KMeans(n_clusters=5, random_state=0)
model.fit(customers_normalized)

# Assign cluster labels to customers
customers["Cluster"] = model.labels_

# Calculate mean values for Recency, Frequency, and MonetaryValue for each cluster
cluster_summary = customers.groupby('Cluster').agg({
   'Recency':'mean',
   'Frequency':'mean',
   'MonetaryValue':['mean', 'count']
}).round(2)

print(cluster_summary)

# In[71]:

# First normalize the data
# Create the dataframe
df_normalized = pd.DataFrame(customers_normalized, columns=['Recency', 'Frequency', 'MonetaryValue'])
df_normalized['ID'] = customers.index
df_normalized['Cluster'] = model.labels_

# Melt The Data
df_nor_melt = pd.melt(df_normalized.reset_index(),
                      id_vars=['ID', 'Cluster'],
                      value_vars=['Recency','Frequency','MonetaryValue'],
                      var_name='Attribute',
                      value_name='Value')

# Visualize it
sns.lineplot(x='Attribute', y='Value', hue='Cluster', data=df_nor_melt)

################################################# Mba analysis ######################################

# In[72]:
# Assuming df1_selected22 is your DataFrame containing 'BillNo' and 'Itemname'
df_fp = Dataset3[['BillNo', 'Itemname']]
df_fp = df_fp.groupby('BillNo')['Itemname'].apply(list).to_list()

# Instantiate a transaction encoder
transEncoder = TransactionEncoder()

# Fit the transaction encoder using the list of transactions
transEncoder.fit(df_fp)

# Transform the transactions into a one-hot encoded matrix
enctrans = transEncoder.transform(df_fp)

# Convert the array of encoded transactions into a DataFrame
df_fp_final = pd.DataFrame(enctrans, columns=transEncoder.columns_)
df_fp_final.head()

# In[75]:
# List of frequency itemset
frequent_itemsets = fpgrowth(df_fp_final, min_support=0.01, use_colnames=True)
frequent_itemsets.head(10)

sorted_frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

print(sorted_frequent_itemsets)
count_of_itemsets = sorted_frequent_itemsets.shape[0]
print("Count of Itemsets:", count_of_itemsets)

association_rules(frequent_itemsets,metric='lift',min_threshold=0.6)


# In[78]:
Dataset66=Dataset3 ####(converting main dataset to dataset 66)


# In[79]:
###for main dataset ,,no of association rules + freuqent pattenr
# In[91]:


# Assuming df1_selected22 is your DataFrame containing 'BillNo' and 'Itemname'
df_fp22 = Dataset66[['BillNo', 'Itemname']]
df_fp22 = df_fp22.groupby('BillNo')['Itemname'].apply(list).tolist()

# Instantiate a transaction encoder
transEncoder22 = TransactionEncoder()

# Fit the transaction encoder using the list of transactions
transEncoder22.fit(df_fp22)

# Transform the transactions into a one-hot encoded matrix
enctrans22 = transEncoder22.transform(df_fp22)

# Convert the array of encoded transactions into a DataFrame
df_fp_final22 = pd.DataFrame(enctrans22, columns=transEncoder22.columns_)

# Perform association rule mining
frequent_itemsets22 = apriori(df_fp_final22, min_support=0.05, use_colnames=True)
association_rules_df22 = association_rules(frequent_itemsets22, metric="lift", min_threshold=0.6)

# Count the number of association rules and patterns
num_association_rules22 = len(association_rules_df22)
num_patterns22 = len(frequent_itemsets22)

print("Number of Association Rules:", num_association_rules22)
print("Number of Patterns:", num_patterns22)

# Displaying the top 10 rules based on 'lift'
top_10_rules = association_rules_df22.sort_values(by="lift", ascending=False).head(10)

for index, row in top_10_rules.iterrows():
    antecedents = ', '.join(list(row['antecedents']))
    consequents = ', '.join(list(row['consequents']))
    support = row['support']
    confidence = row['confidence']
    lift = row['lift']
    
    print(f"Rule: {antecedents} -> {consequents}")
    print(f"Support: {support:.4f}, Confidence: {confidence:.4f}, Lift: {lift:.4f}")
    print("=" * 20)

# Plotting frequent patterns
plt.figure(figsize=(10, 6))
frequent_patterns_plot = sns.barplot(x='support', y='itemsets', data=frequent_itemsets22)
frequent_patterns_plot.set_title("Frequent Patterns")
plt.xlabel("Support")
plt.ylabel("Itemsets")
plt.show()

# Plotting association rules
plt.figure(figsize=(10, 6))
association_rules_plot = sns.scatterplot(x='support', y='confidence', data=association_rules_df22)
association_rules_plot.set_title("Association Rules")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.show()


# In[98]:


# Assuming df1_selected22 is your DataFrame containing 'BillNo' and 'Itemname'
df_fp_season_spring = season_spring[['BillNo', 'Itemname']]
df_fp_season_spring = df_fp_season_spring.groupby('BillNo')['Itemname'].apply(list).tolist()

# Instantiate a transaction encoder
trans_encoder = TransactionEncoder()

# Fit the transaction encoder using the list of transactions
trans_encoder.fit(df_fp_season_spring)

# Transform the transactions into a one-hot encoded matrix
encoded_transactions = trans_encoder.transform(df_fp_season_spring)

# Convert the array of encoded transactions into a DataFrame
df_fp_final1 = pd.DataFrame(encoded_transactions, columns=trans_encoder.columns_)

# Apply FP-Growth to find frequent itemsets
frequent_itemsets1 = fpgrowth(df_fp_final1, min_support=0.05, use_colnames=True)

# Generate association rules from frequent itemsets
association_rules1 = association_rules(frequent_itemsets1, metric="lift", min_threshold=0.6)

# Sort association rules by lift
sorted_association_rules1 = association_rules1.sort_values(by='lift', ascending=False)

# Displaying the top 10 rules based on 'lift'
top_10_rules1 = sorted_association_rules1.head(10)
print("Top 10 Association Rules Based on Lift:")
print(top_10_rules1)

# Plotting frequent patterns
plt.figure(figsize=(10, 6))
frequent_patterns_plot = sns.barplot(x='support', y='itemsets', data=frequent_itemsets1)
frequent_patterns_plot.set_title("Frequent Patterns")
plt.xlabel("Support")
plt.ylabel("Itemsets")
plt.show()

# Plotting association rules
plt.figure(figsize=(10, 6))
association_rules_plot = sns.scatterplot(x='support', y='confidence', data=association_rules1)
association_rules_plot.set_title("Association Rules")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.show()


# In[104]:


# Assuming df1_selected22 is your DataFrame containing 'BillNo' and 'Itemname'
df_fp_season_summer = season_summer[['BillNo', 'Itemname']]
df_fp_season_summer = df_fp_season_summer.groupby('BillNo')['Itemname'].apply(list).tolist()

# Instantiate a transaction encoder
trans_encoder = TransactionEncoder()

# Fit the transaction encoder using the list of transactions
trans_encoder.fit(df_fp_season_summer)

# Transform the transactions into a one-hot encoded matrix
encoded_transactions = trans_encoder.transform(df_fp_season_summer)

# Convert the array of encoded transactions into a DataFrame
df_fp_final2 = pd.DataFrame(encoded_transactions, columns=trans_encoder.columns_)

# Apply FP-Growth to find frequent itemsets
frequent_itemsets2 = fpgrowth(df_fp_final2, min_support=0.05, use_colnames=True)

# Generate association rules from frequent itemsets
association_rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=0.6)

# Sort association rules by lift
sorted_association_rules2 = association_rules2.sort_values(by='lift', ascending=False)

# Displaying the top 10 rules based on 'lift'
top_10_rules2 = sorted_association_rules2.head(10)
print("Top 10 Association Rules Based on Lift for Summer Season:")
print(top_10_rules2)

# Plotting frequent patterns
plt.figure(figsize=(10, 6))
frequent_patterns_plot = sns.barplot(x='support', y='itemsets', data=frequent_itemsets2)
frequent_patterns_plot.set_title("Frequent Patterns for Summer Season")
plt.xlabel("Support")
plt.ylabel("Itemsets")
plt.show()

# Plotting association rules
plt.figure(figsize=(10, 6))
association_rules_plot = sns.scatterplot(x='support', y='confidence', data=association_rules2)
association_rules_plot.set_title("Association Rules for Summer Season")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.show()

# Calculate the number of association rules
count_of_association_rules2 = association_rules2.shape[0]
print("Count of Association Rules for Summer Season:", count_of_association_rules2)

# Calculate the number of frequent itemsets
count_of_frequent_itemsets2 = frequent_itemsets2.shape[0]
print("Count of Frequent Itemsets for Summer Season:", count_of_frequent_itemsets2)


# In[107]:


# Assuming df1_selected22 is your DataFrame containing 'BillNo' and 'Itemname'
df_fp_season_winter = season_winter[['BillNo', 'Itemname']]
df_fp_season_winter = df_fp_season_winter.groupby('BillNo')['Itemname'].apply(list).tolist()

# Instantiate a transaction encoder
trans_encoder = TransactionEncoder()

# Fit the transaction encoder using the list of transactions
trans_encoder.fit(df_fp_season_winter)

# Transform the transactions into a one-hot encoded matrix
encoded_transactions = trans_encoder.transform(df_fp_season_winter)

# Convert the array of encoded transactions into a DataFrame
df_fp_final4 = pd.DataFrame(encoded_transactions, columns=trans_encoder.columns_)

# Apply FP-Growth to find frequent itemsets
frequent_itemsets4 = fpgrowth(df_fp_final4, min_support=0.02, use_colnames=True)

# Generate association rules from frequent itemsets
association_rules4 = association_rules(frequent_itemsets4, metric="lift", min_threshold=0.6)

# Sort association rules by lift
sorted_association_rules4 = association_rules4.sort_values(by='lift', ascending=False)

# Displaying the top 10 rules based on 'lift'
top_10_rules4 = sorted_association_rules4.head(10)
print("Top 10 Association Rules Based on Lift for Winter Season:")
print(top_10_rules4)

# Plotting frequent patterns
plt.figure(figsize=(10, 6))
frequent_patterns_plot = sns.barplot(x='support', y='itemsets', data=frequent_itemsets4)
frequent_patterns_plot.set_title("Frequent Patterns for Winter Season")
plt.xlabel("Support")
plt.ylabel("Itemsets")
plt.show()

# Plotting association rules
plt.figure(figsize=(10, 6))
association_rules_plot = sns.scatterplot(x='support', y='confidence', data=association_rules4)
association_rules_plot.set_title("Association Rules for Winter Season")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.show()

# Calculate the number of association rules
count_of_association_rules4 = association_rules4.shape[0]
print("Count of Association Rules for Winter Season:", count_of_association_rules4)

# Calculate the number of frequent itemsets
count_of_frequent_itemsets4 = frequent_itemsets4.shape[0]
print("Count of Frequent Itemsets for Winter Season:", count_of_frequent_itemsets4)


# In[108]:


# Assuming df1_selected22 is your DataFrame containing 'BillNo' and 'Itemname'
df_fp_season_autumn = season_autumn[['BillNo', 'Itemname']]
df_fp_season_autumn = df_fp_season_autumn.groupby('BillNo')['Itemname'].apply(list).tolist()

# Instantiate a transaction encoder
trans_encoder = TransactionEncoder()

# Fit the transaction encoder using the list of transactions
trans_encoder.fit(df_fp_season_autumn)

# Transform the transactions into a one-hot encoded matrix
encoded_transactions = trans_encoder.transform(df_fp_season_autumn)

# Convert the array of encoded transactions into a DataFrame
df_fp_final_autumn = pd.DataFrame(encoded_transactions, columns=trans_encoder.columns_)

# Apply FP-Growth to find frequent itemsets
frequent_itemsets_autumn = fpgrowth(df_fp_final_autumn, min_support=0.05, use_colnames=True)

# Generate association rules from frequent itemsets
association_rules_autumn = association_rules(frequent_itemsets_autumn, metric="lift", min_threshold=0.6)

# Sort association rules by lift
sorted_association_rules_autumn = association_rules_autumn.sort_values(by='lift', ascending=False)

# Displaying the top 10 rules based on 'lift'
top_10_rules_autumn = sorted_association_rules_autumn.head(10)
print("Top 10 Association Rules Based on Lift for Autumn Season:")
print(top_10_rules_autumn)

# Plotting frequent patterns
plt.figure(figsize=(10, 6))
frequent_patterns_plot_autumn = sns.barplot(x='support', y='itemsets', data=frequent_itemsets_autumn)
frequent_patterns_plot_autumn.set_title("Frequent Patterns for Autumn Season")
plt.xlabel("Support")
plt.ylabel("Itemsets")
plt.show()

# Plotting association rules
plt.figure(figsize=(10, 6))
association_rules_plot_autumn = sns.scatterplot(x='support', y='confidence', data=association_rules_autumn)
association_rules_plot_autumn.set_title("Association Rules for Autumn Season")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.show()

# Calculate the number of association rules
count_of_association_rules_autumn = association_rules_autumn.shape[0]
print("Count of Association Rules for Autumn Season:", count_of_association_rules_autumn)

# Calculate the number of frequent itemsets
count_of_frequent_itemsets_autumn = frequent_itemsets_autumn.shape[0]
print("Count of Frequent Itemsets for Autumn Season:", count_of_frequent_itemsets_autumn)


# In[109]:


import matplotlib.pyplot as plt

# Data
min_support = [0.01, 0.02, 0.03, 0.04, 0.05]
patterns_spring = [1224, 280, 111, 65, 29]
patterns_summer = [2268, 383, 139, 63, 36]
patterns_autumn = [1606, 389, 165, 86, 48]
patterns_winter = [870, 223, 93, 39, 23]

association_spring = [2372, 150, 22, 6, 0]
association_summer = [12748, 512, 84, 20, 0]
association_autumn = [2550, 204, 30, 4, 2]
association_winter = [786, 46, 12, 2, 0]

# Plotting the number of frequent patterns against MIN support
plt.figure(figsize=(10, 6))
plt.plot(min_support, patterns_spring, marker='o', label='Spring')
plt.plot(min_support, patterns_summer, marker='o', label='Summer')
plt.plot(min_support, patterns_autumn, marker='o', label='Autumn')
plt.plot(min_support, patterns_winter, marker='o', label='Winter')

plt.title('Number of Frequent Patterns vs. MIN Support')
plt.xlabel('MIN Support')
plt.ylabel('Number of Patterns')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the number of association rules against MIN support
plt.figure(figsize=(10, 6))
plt.plot(min_support, association_spring, marker='o', label='Spring')
plt.plot(min_support, association_summer, marker='o', label='Summer')
plt.plot(min_support, association_autumn, marker='o', label='Autumn')
plt.plot(min_support, association_winter, marker='o', label='Winter')

plt.title('Number of Association Rules vs. MIN Support')
plt.xlabel('MIN Support')
plt.ylabel('Number of Association Rules')
plt.legend()
plt.grid(True)
plt.show()
