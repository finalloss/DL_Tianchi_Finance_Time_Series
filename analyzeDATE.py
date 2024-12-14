import pandas as pd

# 读取数据
df_product = pd.read_csv('data/product_info_simple_final_train.csv')
df_predict = pd.read_csv('data/predict_table.csv')

# 检查两个数据集的日期范围
df_product['transaction_date'] = pd.to_datetime(df_product['transaction_date'], format='%Y%m%d')
df_predict['transaction_date'] = pd.to_datetime(df_predict['transaction_date'], format='%Y%m%d')

# 获取 product_info_simple_final_train 中的日期范围
min_date_product = df_product['transaction_date'].min()
max_date_product = df_product['transaction_date'].max()

# 获取 predict_table 中的日期范围
min_date_predict = df_predict['transaction_date'].min()
max_date_predict = df_predict['transaction_date'].max()

print(f"Date range in product_info_simple_final_train.csv: {min_date_product} to {max_date_product}")
print(f"Date range in predict_table.csv: {min_date_predict} to {max_date_predict}")
