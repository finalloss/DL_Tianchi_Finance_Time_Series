# preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class DataPreprocessor:
    def __init__(self, df, df_predict, features_to_scale, target, train_start_date, train_end_date):
        self.df = df
        self.df_predict = df_predict
        self.features_to_scale = features_to_scale
        self.target = target
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.le = LabelEncoder()

    def split_train_data(self):
        train_df = self.df[
            (self.df['transaction_date'] >= self.train_start_date) & 
            (self.df['transaction_date'] <= self.train_end_date)
        ]
        #train_df = self.df
        return train_df

    def fit_scalers(self, train_df):
        self.feature_scaler.fit(train_df[self.features_to_scale])
        self.target_scaler.fit(train_df[self.target])

    def handle_outliers_iqr(self, df, columns):
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        return df

    def encode_pid(self, df):
        df['pid_encoded'] = self.le.fit_transform(df['product_pid'])
        return df

    def split_train_val(self, df, validation_days=10, window_size=10, forecast_horizon=10):
        X_train, y_train, X_val, y_val = [], [], [], []
        train_pids_encoded = []
        val_pids_encoded = []

        for pid in df['product_pid'].unique():
            pid_data = df[df['product_pid'] == pid].sort_values(by='transaction_date')
            if len(pid_data) <= (window_size + forecast_horizon + 1):
                continue

            train_data = pid_data.iloc[:-validation_days]
            val_input_data = pid_data.iloc[-validation_days-window_size:-validation_days]
            val_output_data = pid_data.iloc[-validation_days:][self.target]

            for i in range(window_size, len(train_data) - forecast_horizon + 1):
                X_train.append(self.feature_scaler.transform(train_data.iloc[i-window_size:i][self.features_to_scale].values))
                y_train.append(self.target_scaler.transform(train_data.iloc[i:i+forecast_horizon][self.target].values))
                train_pids_encoded.append(self.le.transform([pid])[0])

            X_val.append(self.feature_scaler.transform(val_input_data[self.features_to_scale].values))
            y_val.append(self.target_scaler.transform(val_output_data.values))
            val_pids_encoded.append(self.le.transform([pid])[0])

        return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(train_pids_encoded), np.array(val_pids_encoded)

    def prepare_test_set_with_pid(self, df, window_size=10, forecast_horizon=10):
        X_test = []
        test_pids_encoded = []
        pids = df['product_pid'].unique()
        for pid in pids:
            pid_data = df[df['product_pid'] == pid].sort_values(by='transaction_date')
            test_input = pid_data.iloc[-window_size:][self.features_to_scale].values
            test_input_scaled = self.feature_scaler.transform(test_input)
            X_test.append(test_input_scaled)
            test_pids_encoded.append(self.le.transform([pid])[0])
        return np.array(X_test), np.array(test_pids_encoded)
