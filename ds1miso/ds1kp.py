# importing module
import pandas as pd
import numpy as np
import streamlit as st
from mlxtend.frequent_patterns import association_rules, fpgrowth, apriori
from mlxtend.preprocessing import TransactionEncoder



st.write("<h1>Front-end web</h1>", unsafe_allow_html=True)
st.write("Move the slider to change the values of support and confidence variable")



# dataset
dataset = pd.read_csv("Parkinson.csv")

# Gather All Items of Each Transactions into Numpy Array
transaction = []
for i in range(0, dataset.shape[0]):
    for j in range(0, dataset.shape[1]):
        transaction.append(dataset.values[i,j])

# converting to numpy array
transaction = np.array(transaction)

#  Transform Them a Pandas DataFrame
df = pd.DataFrame(transaction, columns=["items"])

# Put 1 to Each Item For Making Countable Table, to be able to perform Group By
df["incident_count"] = 1

#  Delete NaN Items from Dataset
indexNames = df[df['items'] == "NaN" ].index
df.drop(indexNames , inplace=True)

# Making a New Appropriate Pandas DataFrame for Visualizations
df_table = df.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()

# initializing the transactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transaction).transform(transaction)
dataset = pd.DataFrame(te_ary, columns=te.columns_)

# Extracting the most frequest itemsets via Mlxtend.
# The length column has been added to increase ease of filtering.
frequent_itemsets = apriori(dataset, min_support=0.01, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

#sliders
support = st.slider("Support", min_value=0.01, max_value=0.5, value=0.01)
confidence = st.slider("Confidence", min_value=0.01, max_value=0.95, value=0.01)

# creating asssociation rules
rules=apriori(dataset,min_support=support, use_colnames=True)
rules=association_rules(rules, metric="confidence", min_threshold=confidence)

#st layout adjustments
rules["antecedents"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequents"] = rules["consequents"].apply(lambda x: len(x))

st.write(rules)