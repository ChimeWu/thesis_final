import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime, timedelta

"""
|`pv_production`|production from solar panels in kWh/h|
|`wind_production`|production from wind turbine in kWh/h|
|`consumption`|consumption in kWh/h|
|`spot_market_price`|energy price per NOK/kWh|
|`precip_1h:mm`|amount of rainfall in millimeters that has fallen during the indicated interval|
|`precip_type:idx`|integer indicating the precipitation type (0 - none, 1 - rain, 2 - rain and snow mix, 3 - snow, 4 - sleet, 5 - freezing rain, 6 - hail)|
|`prob_precip_1h:p`|precipitation probability|
|`clear_sky_rad:W`|instantaneous flux of clear sky radiation in Watts per square meter|
|`clear_sky_energy_1h:J`|accumulated clear sky flux over the given interval in Joules per square meter|
|`(diffuse\|direct\|global)_rad:W`|instantaneous flux of diffuse, direct or global radiation in Watts per square meter|
|`(diffuse\|direct\|global)_rad_1h:Wh`|accumulated energy of diffuse, direct or global radiation in Wh per square meter|
|`sunshine_duration_1h:min`|amount of time the sun was shining within the requested interval|
|`sun_azimuth:d`|solar azimuth angle defines the sun's relative direction along the local horizon|
|`sun_elevation:d`|solar elevation angle (angle between the sun and the horizon) gives the position of the sun above the horizon|
|`(low\|medium\|high\|total\|effective)_cloud_cover:p`|amount of cloud cover in percent at different levels in percent|
|`t_(2\|10\|50\|100):C`|temperature in Celsius at 2, 10, 50 or 100 meters above ground|
|`relative_humidity_(2\|10\|50\|100)m:p`|instantaneous value of the relative humidity in % at 2, 10, 50 or 100 meters above ground|
|`dew_point_(2\|10\|50\|100)m:C`|instantaneous value of the dew point temperature in Celsius at 2, 10, 50 or 100 meters above ground|
|`wind_speed_(2\|10\|50\|100)m:ms`|wind speed in meters per second at 2, 10, 50 or 100 meters above ground|
|`wind_dir_(2\|10\|50\|100)m:d`|wind direction in degrees at 2, 10, 50 or 100 meters above ground|

学习数据在文件`data/train.csv`中，我们将使用这些数据训练一个LSTM模型，用于预测发电量。我们将使用天气数据作为输入，发电量、耗电量、市场价格作为输出。我们将使用PyTorch构建模型，使用MSELoss作为损失函数，使用Adam作为优化器。
测试数据在文件`data/test.csv`中，我们将使用这些数据测试模型的性能。
数据的第一列是时间
时间数据类型为标准库提供的datetime，所有时间数据都是整点数据，即分钟和秒都是0。
我们将以30天，也就是720小时（720条数据）为一个时间窗口，每个时间窗口的数据作为一个样本。
Index(['pv_production', 'wind_production', 'consumption', 'spot_market_price',
       'precip_1h:mm', 'precip_type:idx', 'prob_precip_1h:p',
       'clear_sky_rad:W', 'clear_sky_energy_1h:J', 'diffuse_rad:W',
       'diffuse_rad_1h:Wh', 'direct_rad:W', 'direct_rad_1h:Wh', 'global_rad:W',
       'global_rad_1h:Wh', 'sunshine_duration_1h:min', 'sun_azimuth:d',
       'sun_elevation:d', 'low_cloud_cover:p', 'medium_cloud_cover:p',
       'high_cloud_cover:p', 'total_cloud_cover:p', 'effective_cloud_cover:p',
       'temp', 'relative_humidity_2m:p', 'dew_point_2m:C', 'wind_speed_2m:ms',
       'wind_dir_2m:d', 't_10m:C', 'relative_humidity_10m:p',
       'dew_point_10m:C', 'wind_speed_10m:ms', 'wind_dir_10m:d', 't_50m:C',
       'relative_humidity_50m:p', 'dew_point_50m:C', 'wind_speed_50m:ms',
       'wind_dir_50m:d', 't_100m:C', 'relative_humidity_100m:p',
       'dew_point_100m:C', 'wind_speed_100m:ms', 'wind_dir_100m:d'],
      dtype='object')
"""
# 构建模型
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

# 假设 input_dim 是天气数据的维度，output_dim 是发电数据的维度
input_dim = 39
output_dim = 4
hidden_dim = 64
num_layers = 3

#d读取数据
data = pd.read_csv('data/train.csv', index_col=0, parse_dates=True)
print(len(data.columns))
input = data[['precip_1h:mm', 'precip_type:idx', 'prob_precip_1h:p',
       'clear_sky_rad:W', 'clear_sky_energy_1h:J', 'diffuse_rad:W',
       'diffuse_rad_1h:Wh', 'direct_rad:W', 'direct_rad_1h:Wh', 'global_rad:W',
       'global_rad_1h:Wh', 'sunshine_duration_1h:min', 'sun_azimuth:d',
       'sun_elevation:d', 'low_cloud_cover:p', 'medium_cloud_cover:p',
       'high_cloud_cover:p', 'total_cloud_cover:p', 'effective_cloud_cover:p',
       'temp', 'relative_humidity_2m:p', 'dew_point_2m:C', 'wind_speed_2m:ms',
       'wind_dir_2m:d', 't_10m:C', 'relative_humidity_10m:p',
       'dew_point_10m:C', 'wind_speed_10m:ms', 'wind_dir_10m:d', 't_50m:C',
       'relative_humidity_50m:p', 'dew_point_50m:C', 'wind_speed_50m:ms',
       'wind_dir_50m:d', 't_100m:C', 'relative_humidity_100m:p',
       'dew_point_100m:C', 'wind_speed_100m:ms', 'wind_dir_100m:d']]
output = data[['pv_production', 'wind_production', 'consumption', 'spot_market_price']]


model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 准备数据
input = torch.tensor(input.values).float()
output = torch.tensor(output.values).float()

# 时间窗口大小
window_size = 720

# 训练模型
rolling_window = torch.tensor([input[i:i+window_size] for i in range(len(input)-window_size)])
rolling_output = torch.tensor([output[i+window_size] for i in range(len(output)-window_size)])

# 验证数据
val_data = pd.read_csv('data/test.csv', index_col=0, parse_dates=True)
val_input = val_data[['precip_1h:mm', 'precip_type:idx', 'prob_precip_1h:p',
       'clear_sky_rad:W', 'clear_sky_energy_1h:J', 'diffuse_rad:W',
       'diffuse_rad_1h:Wh', 'direct_rad:W', 'direct_rad_1h:Wh', 'global_rad:W',
       'global_rad_1h:Wh', 'sunshine_duration_1h:min', 'sun_azimuth:d',
       'sun_elevation:d', 'low_cloud_cover:p', 'medium_cloud_cover:p',
       'high_cloud_cover:p', 'total_cloud_cover:p', 'effective_cloud_cover:p',
       'temp', 'relative_humidity_2m:p', 'dew_point_2m:C', 'wind_speed_2m:ms',
       'wind_dir_2m:d', 't_10m:C', 'relative_humidity_10m:p',
       'dew_point_10m:C', 'wind_speed_10m:ms', 'wind_dir_10m:d', 't_50m:C',
       'relative_humidity_50m:p', 'dew_point_50m:C', 'wind_speed_50m:ms',
       'wind_dir_50m:d', 't_100m:C', 'relative_humidity_100m:p',
       'dew_point_100m:C', 'wind_speed_100m:ms', 'wind_dir_100m:d']]
val_output = val_data[['pv_production', 'wind_production', 'consumption', 'spot_market_price']]
val_input = torch.tensor(val_input.values).float()
val_output = torch.tensor(val_output.values).float()

# 训练模型
for i in range(100):
    # 训练阶段
    for j in range(len(rolling_window)-1):
        model.train()
        optimizer.zero_grad()
        y_pred = model(rolling_window[j].unsqueeze(0))
        loss = criterion(y_pred, rolling_output[j+1].unsqueeze(0))
        loss.backward()
        optimizer.step()

    # 验证阶段
    model.eval()
    with torch.no_grad():
        # 计算验证损失
        val_pred = model(val_input[:-1].unsqueeze(0))
        val_loss = criterion(val_pred, val_output[1:].unsqueeze(0))
        print(f'Epoch {i}, Validation Loss: {val_loss.item()}')

# 保存模型
torch.save(model.state_dict(), 'model.pth')

# 测试模型
def test_model():
    data = pd.read_csv('data/test.csv', index_col=0, parse_dates=True)
    input = data[['precip_1h:mm', 'precip_type:idx', 'prob_precip_1h:p', 'clear_sky_rad:W', 'clear_sky_energy_1h:J', 'diffuse_rad:W', 'diffuse_rad_1h:Wh', 'sunshine_duration_1h:min', 'sun_azimuth:d', 'sun_elevation:d', 'low_cloud_cover:p', 't_2:C', 'relative_humidity_2m:p', 'dew_point_2m:C', 'wind_speed_2m:ms', 'wind_dir_2m:d']]
    output = data[['pv_production', 'wind_production', 'consumption', 'spot_market_price']]
    input = torch.tensor(input.values).float()
    output = torch.tensor(output.values).float()
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    with torch.no_grad():
        outputs = model(input)
        loss = criterion(outputs, output)
        print(f'Test Loss: {loss.item()}')
        print(outputs)
        print(output)

    # 保存预测结果
    pd.DataFrame(outputs.numpy(), columns=['pv_production', 'wind_production', 'consumption', 'spot_market_price']).to_csv('predictions.csv')

    # 保存损失
    pd.DataFrame([loss.item()], columns=['loss']).to_csv('loss.csv')

