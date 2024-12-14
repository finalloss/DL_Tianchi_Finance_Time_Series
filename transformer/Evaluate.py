# evaluate.py
import matplotlib.pyplot as plt
import numpy as np

class Evaluator:
    @staticmethod
    def wmape(y_true, y_pred):
        return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8)

    @staticmethod
    def plot_history(history):
        # 临时配置 Matplotlib 使用支持中文的字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='训练损失 (MAE)')
        plt.plot(history.history['val_loss'], label='验证损失 (MAE)')
        plt.plot(history.history['wmape'], label='训练 WMAPE')
        plt.plot(history.history['val_wmape'], label='验证 WMAPE')
        plt.legend()
        plt.title('训练与验证损失曲线')
        plt.xlabel('Epochs')
        plt.ylabel('损失')
        plt.show()

    @staticmethod
    def plot_predictions(y_true, y_pred, metric='apply_amt'):
        # 临时配置 Matplotlib 使用支持中文的字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label=f'实际值 ({metric})', linestyle='-', alpha=0.7)
        plt.plot(y_pred, label=f'预测值 ({metric})', linestyle='--', alpha=0.7)
        plt.legend()
        plt.title(f'验证集 {metric} 实际值 vs 预测值')
        plt.xlabel('时间序列索引')
        plt.ylabel(metric)
        plt.grid(True)
        plt.show()
