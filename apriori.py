import time
from numpy import product
import pandas as pd

from mlxtend.frequent_patterns import association_rules, apriori

print("Importing data...")
start_time = time.time()
df = pd.read_excel('invoices.xlsx')
print("Data imported successfully! (took %.2f seconds)" % (time.time() - start_time))

print("Preprocessing data...")
start_time = time.time()
df.dropna(inplace=True)

df_Invoice = pd.DataFrame({"Invoice":[row for row in df["Invoice"].values if "C"  not in str(row)]})
df_Invoice = df_Invoice.drop_duplicates("Invoice")
df = df.merge(df_Invoice, on = "Invoice")

df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

df_product = (
    df[["Description", "StockCode"]].drop_duplicates()
    .groupby(["Description"])["StockCode"].count().reset_index()
    .rename(columns={'StockCode': 'StockCode_Count'})
)
df_multiple_stock_code = df_product[df_product["StockCode_Count"] > 1]
df = df[~df["Description"].isin(df_multiple_stock_code["Description"])]

df_product = (
    df[["Description", "StockCode"]].drop_duplicates()
    .groupby(["StockCode"])["Description"].count().reset_index()
    .rename(columns={'Description': 'Description_Count'})
)
df_multiple_desctiption = df_product[df_product["Description_Count"] > 1]
df = df[~df["StockCode"].isin(df_multiple_desctiption["StockCode"])]


df = df[~df["StockCode"].str.contains("POST", na=False)]

df = df[df["Country"] == "United Kingdom"]

print("Finish preprocessing! (took %.2f seconds)" % (time.time() - start_time))

print("Building matrix...")
start_time = time.time()

matrix = df.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
print("Finish bulding matrix! (took %.2f seconds)" % (time.time() - start_time))

print("Building association rules...")
start_time = time.time()
frequent_itemsets = apriori(matrix, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("lift", ascending=False, inplace=True)
print("Finish bulding association rules! (took %.2f seconds)" % (time.time() - start_time))

def get_product_name(dataframe, stockcode):
    product_name = dataframe[dataframe["StockCode"] == stockcode]["Description"].unique()[0]
    return str(stockcode), product_name

def get_recommend_products(code, rules, limit):
    recommendation_list = set()
    for idx, product in enumerate(rules["antecedents"]):
        for j in list(product):
            if str(j) == str(code):
                consequents = rules.iloc[idx]["consequents"]
                recommendation_list.update(consequents)
        if len(recommendation_list) >= limit:
            break
    return list(recommendation_list)


def recommendation_system_func(product_id, dataframe, rules, limit):
    if product_id in list(dataframe["StockCode"].astype("str").unique()):
        print("Finding recommend products...")
        start_time = time.time()
        product_list = get_recommend_products(product_id, rules, limit)
        print("Finish finding recommend products! (took %.2f seconds)" % (time.time() - start_time))
        if len(product_list) == 0:
            print("There is no product can be recommended!")
        else:
            print("Recommend products for " , get_product_name(dataframe, product_id) , " can be seen below:")
        
            for i in range(max(len(product_list), limit)):
                print(get_product_name(dataframe, product_list[i]))
    else:
        print("Invalid Product Id, try again!")

while True:
    print('<<----------------------------------------------------->>')
    product_id = input("Enter a product id (type 'exit' to quit): ")
    if product_id == 'exit': break
    recommendation_system_func(product_id, df, rules, 10)