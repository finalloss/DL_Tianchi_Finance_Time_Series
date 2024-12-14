import pandas as pd

# 读取数据
df_product = pd.read_csv('data/product_info_simple_final_train.csv')
df_predict = pd.read_csv('data/predict_table.csv')

# 查看两者的 product_pid 列的唯一值
product_pids_product = df_product['product_pid'].unique()
product_pids_predict = df_predict['product_pid'].unique()

# 打印唯一的 product_pid 值
print("Unique product_pid in product_info_simple_final_train.csv:")
print(product_pids_product)

print("\nUnique product_pid in predict_table.csv:")
print(product_pids_predict)

# 找出两个文件中共同的 product_pid
common_product_pids = set(product_pids_product).intersection(set(product_pids_predict))
print("\nCommon product_pid between both datasets:")
print(common_product_pids)

# 也可以检查不重合的 product_pid
unique_in_product_info = set(product_pids_product) - set(product_pids_predict)
unique_in_predict = set(product_pids_predict) - set(product_pids_product)

print("\nproduct_pid only in product_info_simple_final_train.csv:")
print(unique_in_product_info)

print("\nproduct_pid only in predict_table.csv:")
print(unique_in_predict)
