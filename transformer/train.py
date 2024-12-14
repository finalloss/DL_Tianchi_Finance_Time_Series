# train.py
import warnings
warnings.filterwarnings("ignore")

from config import Config
from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from preprocessing import DataPreprocessor
from model_builder import TransformerModelBuilder
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import numpy as np

# 新增导入Evaluator
from Evaluate import Evaluator

# 数据加载与初步处理
loader = DataLoader(
    product_path='data/product_info_simple_final_train.csv',
    yield_path='data/cbyieldcurve_info_final.csv',
    time_path='data/time_info_final.csv',
    predict_path='data/predict_table.csv'
)
loader.convert_datetime()
loader.drop_nat()
loader.fill_missing_values()
loader.align_train_test_pids()
loader.merge_data()
df_product, df_predict = loader.get_product_data()

# 特征工程
fe = FeatureEngineer(df_product)
fe.create_time_features()
fe.add_normalized_time_feature()  # 调用新增的时间标准化特征方法
fe.scale_uv_features()
fe.log_transform_yield()
fe.create_lag_features()  # 包含 dropna
fe.fill_missing_mean()
product_data = fe.get_df()

# 数据预处理
dp = DataPreprocessor(
    df=product_data,
    df_predict=df_predict,
    features_to_scale=Config.FEATURES_TO_SCALE,
    target=Config.TARGET,
    train_start_date=Config.TRAIN_START_DATE,
    train_end_date=Config.TRAIN_END_DATE
)

train_df = dp.split_train_data()

# 拟合缩放器
dp.fit_scalers(train_df)

# 异常值处理
train_df_balanced = dp.handle_outliers_iqr(train_df.copy(), Config.TARGET)

# PID编码
train_df_balanced = dp.encode_pid(train_df_balanced)

# 划分训练与验证集
X_train, y_train, X_val, y_val, train_pids_encoded, val_pids_encoded = dp.split_train_val(
    train_df_balanced, 
    validation_days=Config.VALIDATION_DAYS, 
    window_size=Config.INPUT_WINDOW, 
    forecast_horizon=Config.OUTPUT_WINDOW
)

# 构建模型
model_builder = TransformerModelBuilder()
pid_vocab_size = len(dp.le.classes_)

if Config.IFLSTM == True:
    model = model_builder.build_lstm_attention_model_with_pid(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        pid_vocab_size=pid_vocab_size,
        forecast_horizon=Config.OUTPUT_WINDOW,
        target_len=len(Config.TARGET),
        pid_embedding_dim=10,
        lstm_units=64,
        attention_units=64,
        dropout=0.1
    )
else:
    model = model_builder.build_transformer_model_with_pid(
        input_shape=(X_train.shape[1], X_train.shape[2]), 
        pid_vocab_size=pid_vocab_size, 
        forecast_horizon=Config.OUTPUT_WINDOW, 
        target_len=len(Config.TARGET)
    )

optimizer = Adam(learning_rate=1e-4, clipvalue=1.0)
model.compile(optimizer=optimizer, loss='mae', metrics=[model_builder.wmape, 'mae', 'mse'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

history = model.fit(
    [X_train, train_pids_encoded],
    y_train,
    epochs=100,
    batch_size=Config.BATCH_SIZE,
    validation_data=([X_val, val_pids_encoded], y_val),
    callbacks=[early_stopping, lr_scheduler]
)

# 使用Evaluator绘制训练过程
Evaluator.plot_history(history)

val_metrics = model.evaluate([X_val, val_pids_encoded], y_val)
print(f"验证集 - 损失 (MAE): {val_metrics[0]}")
print(f"验证集 - WMAPE: {val_metrics[1]}")
print(f"验证集 - MSE: {val_metrics[3]}")

# 预测验证集数据用于可视化（可根据需求调整）
y_pred_val_scaled = model.predict([X_val, val_pids_encoded])
y_pred_val = dp.target_scaler.inverse_transform(y_pred_val_scaled.reshape(-1, y_pred_val_scaled.shape[-1]))
y_val_original = dp.target_scaler.inverse_transform(y_val.reshape(-1, y_val.shape[-1]))

def wmape_metric(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8)

wmape = wmape_metric(y_val_original, y_pred_val)

print(f"验证集反归一化后 WMAPE: {wmape}")

# 对单个目标列例如apply_amt进行可视化对比(这里假设目标列顺序与TARGET一致)
Evaluator.plot_predictions(y_val_original[:, 0], y_pred_val[:, 0], metric='apply_amt')
Evaluator.plot_predictions(y_val_original[:, 1], y_pred_val[:, 1], metric='redeem_amt')
Evaluator.plot_predictions(y_val_original[:, 2], y_pred_val[:, 2], metric='net_in_amt')

# 准备测试集并预测
X_test, test_pids_encoded = dp.prepare_test_set_with_pid(product_data, window_size=Config.INPUT_WINDOW, forecast_horizon=Config.OUTPUT_WINDOW)
y_pred_test_scaled = model.predict([X_test, test_pids_encoded])
y_pred_test = dp.target_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, y_pred_test_scaled.shape[-1]))

apply_amt_pred = y_pred_test[:,0]
redeem_amt_pred = y_pred_test[:,1]
net_in_amt_pred = y_pred_test[:,2]
net_in_amt_pred_transformer = apply_amt_pred - redeem_amt_pred

df_predict['apply_amt_pred'] = apply_amt_pred
df_predict['redeem_amt_pred'] = redeem_amt_pred
df_predict['net_in_amt_pred'] = net_in_amt_pred * 0.5 + net_in_amt_pred_transformer * 0.5

""" count = 0
for pid in df_predict['product_pid'].unique():  # Loop through unique pids
    df_predict.loc[df_predict['product_pid'] == pid, 'apply_amt_pred'] = apply_amt_pred[count]
    df_predict.loc[df_predict['product_pid'] == pid, 'redeem_amt_pred'] = redeem_amt_pred[count]
    df_predict.loc[df_predict['product_pid'] == pid, 'net_in_amt_pred'] = (net_in_amt_pred[count] * 0.5 + net_in_amt_pred_transformer[count] * 0.5)
    count += 1 """

df_predict.to_csv('test_data_with_predictions.csv', index=False)
