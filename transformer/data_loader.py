# data_loader.py
import pandas as pd
import numpy as np
from datetime import datetime

class DataLoader:
    def __init__(self, product_path, yield_path, time_path, predict_path):
        self.df_product = pd.read_csv(product_path)
        self.df_yield = pd.read_csv(yield_path)
        self.df_time = pd.read_csv(time_path)
        self.df_predict = pd.read_csv(predict_path)

    def convert_datetime(self):
        self.df_product['transaction_date'] = pd.to_datetime(
            self.df_product['transaction_date'], format='%Y%m%d', errors='coerce')
        self.df_yield['enddate'] = pd.to_datetime(
            self.df_yield['enddate'], format='%Y%m%d', errors='coerce')
        self.df_time['stat_date'] = pd.to_datetime(
            self.df_time['stat_date'], format='%Y%m%d', errors='coerce')

    def drop_nat(self):
        for df, date_col in zip([self.df_product, self.df_yield, self.df_time], 
                                ['transaction_date', 'enddate', 'stat_date']):
            if df[date_col].isnull().sum() > 0:
                df.dropna(subset=[date_col], inplace=True)

    def fill_missing_values(self):
        for feature in ['during_days', 'total_net_value']:
            self.df_product[feature] = self.df_product.groupby('product_pid')[feature].transform(
                lambda x: x.fillna(x.median())
            )

    def align_train_test_pids(self):
        train_pids = self.df_product['product_pid'].unique()
        test_pids = self.df_predict['product_pid'].unique()
        pids_in_train_not_in_test = set(train_pids) - set(test_pids)
        self.df_product = self.df_product[~self.df_product['product_pid'].isin(pids_in_train_not_in_test)]

        # 删除total_net_value仍然缺失的pid
        missing_pid = self.df_product[self.df_product['total_net_value'].isnull()]['product_pid'].unique()
        self.df_product = self.df_product[~self.df_product['product_pid'].isin(missing_pid)]

    def merge_data(self):
        # 与time_info合并
        self.df_product = self.df_product.merge(
            self.df_time[['stat_date', 'is_trade', 'is_week_end', 'is_month_end', 
                          'is_quarter_end', 'is_year_end']], 
            left_on='transaction_date', right_on='stat_date', how='left'
        )
        self.df_product.drop(columns=['stat_date'], inplace=True)

        # 与yield合并
        self.df_product = self.df_product.merge(
            self.df_yield[['enddate', 'yield']], 
            left_on='transaction_date', right_on='enddate', how='left'
        )
        self.df_product.drop(columns=['enddate'], inplace=True)

    def get_product_data(self):
        return self.df_product, self.df_predict
