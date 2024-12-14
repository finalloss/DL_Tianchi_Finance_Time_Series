# config.py
import torch

class Config:
    INPUT_WINDOW = 10
    OUTPUT_WINDOW = 10
    VALIDATION_DAYS = 10
    BATCH_SIZE = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TRAIN_START_DATE = '2022-05-01'
    TRAIN_END_DATE = '2022-11-09'

    FEATURES_TO_SCALE = [
        'apply_amt', 'redeem_amt', 'net_in_amt', 
        'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos', 
        'apply_amt_lag_1', 'redeem_amt_lag_1', 'net_in_amt_lag_1', 
        'apply_amt_lag_3', 'redeem_amt_lag_3', 'net_in_amt_lag_3', 
        'apply_amt_lag_7', 'redeem_amt_lag_7', 'net_in_amt_lag_7', 
        'uv_fundown_scaled', 'uv_stableown_scaled', 'uv_fundopt_scaled', 
        'uv_fundmarket_scaled', 'uv_termmarket_scaled', 'log_yield',
        'time_normalized' 
    ]

    TARGET = ['apply_amt', 'redeem_amt', 'net_in_amt']
    
    IFLSTM = True