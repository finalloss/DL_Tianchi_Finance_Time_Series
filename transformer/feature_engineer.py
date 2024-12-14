# feature_engineer.py
from datetime import datetime
import numpy as np
import pandas as pd

class FeatureEngineer:
    def __init__(self, df):
        self.df = df

    def create_time_features(self):
        self.df['year'] = self.df['transaction_date'].dt.year
        self.df['month'] = self.df['transaction_date'].dt.month
        self.df['day'] = self.df['transaction_date'].dt.day
        self.df['weekday'] = self.df['transaction_date'].dt.weekday
        self.df['is_weekend'] = self.df['weekday'].isin([5, 6])  # 周末

        # 周期编码
        self.df['weekday_sin'] = np.sin(2 * np.pi * self.df['weekday'] / 7)
        self.df['weekday_cos'] = np.cos(2 * np.pi * self.df['weekday'] / 7)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)

    def add_normalized_time_feature(self):
        # 引入base和change函数对日期进行标准化
        base = int(datetime.strptime(str(20210104), "%Y%m%d").timestamp()) + 58320000
        change = lambda x: (int(datetime.strptime(str(x), "%Y%m%d").timestamp()) - base)/86400
        # 将日期列转为字符串格式YYYYMMDD再应用change
        self.df['time_normalized'] = self.df['transaction_date'].apply(lambda d: change(d.strftime('%Y%m%d')))

    def create_lag_features(self, lags=[1,3,7], features=['apply_amt', 'redeem_amt', 'net_in_amt']):
        for lag in lags:
            for feature in features:
                self.df[f'{feature}_lag_{lag}'] = self.df[feature].shift(lag)
        self.df.dropna(inplace=True)

    def scale_uv_features(self):
        for col in ['uv_fundown', 'uv_stableown', 'uv_fundopt', 'uv_fundmarket', 'uv_termmarket']:
            self.df[f'{col}_scaled'] = self.df[col] / (self.df[col].max() + 1e-5)

    def log_transform_yield(self):
        self.df['log_yield'] = np.log(self.df['yield'] + 1)

    def fill_missing_mean(self):
        self.df.fillna(self.df.mean(), inplace=True)

    def get_df(self):
        return self.df
