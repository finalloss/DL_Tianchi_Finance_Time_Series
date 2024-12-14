import argparse
import os
import yaml
import logging
import sys
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

### 读取数据集

# 申赎信息
product_info = pd.read_csv('dataset/product_info_simple_final_train.csv')
# 时间信息
time_info = pd.read_csv('dataset/time_info_final.csv')
# 行情信息
yield_info = pd.read_csv('dataset/cbyieldcurve_info_final.csv')
# 测试信息
test_info = pd.read_csv('dataset/predict_table.csv')

### 分析基金特征
print(product_info.info())
print(product_info.isnull().sum())
print(product_info.describe().transpose())
product_pids = product_info['product_pid'].unique()
print(f'基金种类: {len(product_pids)}种')

### 处理数据
# 处理申赎信息, 将日期命名为date, 删除product_pid中的字符串, 移除net_in_amt、during_days、total_net_value
product_info.rename(columns={'transaction_date':'date'}, inplace=True)
product_info['product_pid'] = product_info['product_pid'].apply(lambda x: int(x[7:]))
product_info.drop('net_in_amt', axis=1, inplace=True)
product_info.drop('during_days', axis=1, inplace=True)
product_info.drop('total_net_value', axis=1, inplace=True)

# 处理时间信息, 将日期命名为date, 将日期相关转化为整数
time_info.rename(columns={'stat_date':'date'}, inplace=True)
time_info['next_trade_date'] = pd.to_datetime(time_info['next_trade_date'], format = '%Y%m%d') - pd.to_datetime(time_info['date'], format = '%Y%m%d')
time_info['last_trade_date'] = pd.to_datetime(time_info['date'], format = '%Y%m%d') - pd.to_datetime(time_info['last_trade_date'], format = '%Y%m%d')
time_info['next_trade_date'] = time_info['next_trade_date'].apply(lambda x: x.days)
time_info['last_trade_date'] = time_info['last_trade_date'].apply(lambda x: x.days)

# 处理行情信息, 将日期命名为date
yield_info.rename(columns={'enddate':'date'}, inplace=True)

# 处理测试信息, 将日期命名为date, 删除product_pid中的字符串
# test_info.rename(columns={'transaction_date':'date'}, inplace=True)
# test_info['product_pid'] = test_info['product_pid'].apply(lambda x: int(x[7:]))

# 根据日期整合申赎信息、时间信息、行情信息
total_dataset = product_info.copy()
total_dataset = pd.merge(total_dataset, time_info, how='left', on='date')
total_dataset = pd.merge(total_dataset, yield_info, how='left', on='date')

# 对下列feature进行归一化
scaler = StandardScaler()
feature_lis_need_scalar = ['uv_fundown', 'uv_stableown', 'uv_fundopt', 'uv_fundmarket', 'uv_termmarket',
                'is_trade', 'next_trade_date','last_trade_date','is_week_end', 'is_month_end','is_quarter_end','is_year_end','trade_day_rank','yield']
feature_scaler_data = scaler.fit_transform(total_dataset[feature_lis_need_scalar])
total_dataset[feature_lis_need_scalar] = feature_scaler_data

# 输出数据
total_dataset.to_csv('total_dataset.csv',index=False)
test_info.to_csv("test_data.csv", index=False)

feature_list = ['product_pid',
                'uv_fundown', 'uv_stableown', 'uv_fundopt', 'uv_fundmarket', 'uv_termmarket',
                'is_trade', 'next_trade_date','last_trade_date','is_week_end', 'is_month_end','is_quarter_end','is_year_end','trade_day_rank','yield']
label_list = ['apply_amt', 'redeem_amt']

# 使用十天数据预测后十天
window_config = {
    'feature_size': 10,
    'label_size': 10
}


### 处理参数
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser('time series forecasting in the financial sector')

    parser.add_argument('--cpu', action='store_true', help='wheather to use cpu , if not set, use gpu')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optim', type=str, default='adam', choices=['sgd','adam','adamw'])
    parser.add_argument('--loss_method', type=str, default='wmape', choices=['wmape','mse'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--lr_decay', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--best_loss', type=float, default=math.inf)

    parser.add_argument('--saved_dir', type=str, default='./saved_dir')

    args = parser.parse_args()

    if not args.cpu and not torch.cuda.is_available():
        logger.error('cuda is not available, try running on CPU by adding param --cpu')
        sys.exit()

    args.begin_time = datetime.datetime.now().strftime('%H%M%S')
    args.saved_dir = Path(args.saved_dir) / args.begin_time
    args.saved_dir.mkdir(exist_ok=True,parents=True)
    logger.info(f'Save Dir : {str(args.saved_dir)}')    

    with open(args.saved_dir / 'opt.yaml','w') as f:
        yaml.dump(vars(args),f,sort_keys=False)

    return args

### 划分数据集、验证集

def train_valid_split(data_set):
    valid = pd.DataFrame()         
    train = pd.DataFrame() 
    product_pids = data_set['product_pid'].unique().tolist()

    for pid in product_pids:
        pid_data = data_set[(data_set['product_pid'] == pid)]
        valid_p = pid_data.tail(window_config['feature_size'] + window_config['label_size'])
        train_p = pid_data.head(pid_data.shape[0] - window_config['label_size'])  
        valid = pd.concat([valid, valid_p], ignore_index=True)
        train = pd.concat([train, train_p], ignore_index=True)

    return train, valid

### Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.product_pids = data['product_pid'].unique()
        self.features = []
        self.labels = []
        for pid in self.product_pids:
            same_product_data = data[data['product_pid'] == pid]
            for i in range(0, len(same_product_data)-window_config['feature_size']-window_config['label_size']+1):
                self.labels.append(np.array(same_product_data[label_list].iloc[(i+window_config['feature_size']):(i+window_config['feature_size']+window_config['label_size'])]))
                feature = []
                for j in range(0, window_config['feature_size']):
                    feature.append(same_product_data[feature_list].iloc[i+j])
                self.features.append(np.array(feature))

    def __getitem__(self, index):
        feature = torch.from_numpy(self.features[index]).float().unsqueeze(0)
        label = torch.from_numpy(self.labels[index]).float().unsqueeze(0).permute(0, 2, 1).reshape(1, -1)
        return feature, label

    def __len__(self):
        return len(self.features)
    
### Net
class Net(nn.Module):
    def __init__(self, num_label):
        super(Net, self).__init__()
        self.num_label = num_label
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 5, 1, 1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(64, 16)
        self.norm1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, self.num_label*10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.unsqueeze(1)
        return x

### 保存损失
plt.rcParams['font.family'] = 'SimHei'
def save_loss(output_dir, train_loss_record, valid_loss_record):
    plt.figure()
    plt.plot(np.arange(len(train_loss_record)), train_loss_record, label='train')
    plt.plot(np.arange(len(valid_loss_record)), valid_loss_record, label='valid')
    plt.legend()
    plt.grid(True)
    plt.title('train_valid_loss')
    plt.savefig(str(output_dir / 'train_valid_loss.png'))
    plt.close()

### 构建模型
def _make_model(device):
    model = Net(len(label_list)).to(device)
    model_parameters = filter(lambda p:p.requires_grad,model.parameters())
    n_params = sum([p.numel() for p in model_parameters])
    logger.info('Model Setting ...')
    logger.info(f'Number of model params: {n_params}')
    return model

### 构建优化器
def _make_optimizer(args, model):
    logger.info(f'Using {args.optim} Optimizer ......')
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.epsilon)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, nesterov=True, weight_decay=args.weight_decay)
    return optimizer

### 构建损失
def _make_criterion(args):
    if args.loss_method == 'mse':
        criterion =nn.MSELoss(reduction='mean')
    elif args.loss_method == 'wmape':
        criterion = lambda pred, y: torch.mean(torch.sum(torch.abs(pred-y), dim = 2)/torch.sum(torch.abs(y), dim = 2))
    return criterion

### 保存模型
def save_checkpoint(checkpoint_path, model, optimizer, epoch, best_loss):
    if isinstance(model,(torch.nn.parallel.DistributedDataParallel,torch.nn.DataParallel)):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    checkpoint = {
        'modelstate':model_state_dict,
        'optimstate':optimizer.state_dict(),
        'epoch_id':epoch,
        'beat_loss':best_loss
        }
    torch.save(checkpoint, checkpoint_path)

### 训练过程
def train(args):
    device = torch.device('cuda' if not args.cpu else 'cpu')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.cpu:
        torch.cuda.manual_seed(args.seed)

    model = _make_model(device)
    optimizer = _make_optimizer(args, model)
    criterion = _make_criterion(args)

    train_data, valid_data = train_valid_split(total_dataset)
    train_dataset, valid_dataset = TimeSeriesDataset(train_data), TimeSeriesDataset(valid_data)
    logger.info(f'Number of train samples: {len(train_dataset)}')
    logger.info(f'Number of valid samples: {len(valid_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

    epoch_id = -1
    best_loss = args.best_loss

    train_loss_record = []
    valid_loss_record = []
    early_stop_count = 0

    try:
        for epoch in range(epoch_id + 1, args.epoch):
            model.train()
            loss_record = []
            train_pbar = tqdm(train_loader, position=0, leave=True)
            for x, y in train_pbar:
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)  
                pred = model(x)    
                loss = criterion(pred, y)
                loss.backward()                   
                optimizer.step()
                l_ = loss.detach().item()
                loss_record.append(l_)
                train_pbar.set_description(f'Epoch [{epoch+1}/{args.epoch}]')
                train_pbar.set_postfix({'loss': f'{l_:.5f}'})
            mean_train_loss = sum(loss_record)/len(loss_record)
            train_loss_record.append(mean_train_loss)

            print(f'Epoch [{epoch+1}/{args.epoch}]: Train loss: {mean_train_loss:.4f}')

            model.eval()
            loss_record = []
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    pred = model(x)
                    loss = criterion(pred, y)
                    loss_record.append(loss.item())
            
            mean_valid_loss = sum(loss_record)/len(loss_record)
            valid_loss_record.append(mean_valid_loss)

            print(f'Epoch [{epoch+1}/{args.epoch}]: Valid loss: {mean_valid_loss:.4f}')

            if mean_valid_loss < best_loss:
                best_loss = mean_valid_loss
                checkpoint_path = args.saved_dir / 'model.pth'
                save_checkpoint(checkpoint_path, model, optimizer, epoch, best_loss)
                print('Saving model with loss {:.3f}...'.format(best_loss))
                early_stop_count = 0
            else:
                early_stop_count += 1
            
            if early_stop_count >= args.early_stop:
                print('\nModel is not improving, so we halt the training session.')
                break

            save_loss(args.saved_dir, train_loss_record, valid_loss_record)
    
    except KeyboardInterrupt:
        logger.info('Catch a KeyboardInterupt')


### 预测
def predict(args):
    device = torch.device('cuda' if not args.cpu else 'cpu')
    model = Net(len(label_list)).to(device)
    checkpoint = torch.load(args.saved_dir/'model.pth')
    model.load_state_dict(checkpoint['modelstate'])

    product_pids = test_info['product_pid'].unique()
    total_data_group_product = total_dataset[feature_list].groupby('product_pid')

    model.eval()
    with torch.no_grad():
        for pid in product_pids:
            pid_id = int(pid[7:])
            feat_np = np.array(total_data_group_product.get_group(pid_id).tail(window_config['feature_size']))
            feature = torch.from_numpy(feat_np).float().unsqueeze(0).unsqueeze(0).to(device)
            pred = model(feature)
            pred_th = torch.from_numpy(pred.cpu().numpy())
            pred_th = pred_th.squeeze().reshape(2, -1)
            test_info.loc[test_info['product_pid']==pid, 'apply_amt_pred'] = pred_th[0].squeeze().numpy()
            test_info.loc[test_info['product_pid']==pid, 'redeem_amt_pred'] = pred_th[1].squeeze().numpy()
            test_info.loc[test_info['product_pid']==pid, 'net_in_amt_pred'] = pred_th[0].squeeze().numpy() - pred_th[1].squeeze().numpy()

    test_info.to_csv('result.csv', index=False)   

def main():
    args = get_args()
    print(args)
    train(args)
    predict(args)

if __name__ == '__main__':
    main()