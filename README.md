# AI Hackathon Challenge - Optimal Control of Microgrid 

## Introduction

In this challenge you will develop a control system that minimizes the cost for the microgrid owners.

> Microgrid is a small network of electricity users with a local source of supply 
> that is usually attached to a centralized grid but is able to function independently.

Microgrids are important for creating sustainable and cost-efficient energy systems based on renewable sources.
AI and optimization methods can be used to improve operational efficiency of microgrids. Good control algorithms ensures reliability and cost efficiency.

To develop and test your system you will use a microgrid simulator included in this repository.

The introduction presentation can be found [here](docs/AI-brAIn-hackaton-2021.pdf).

在这个挑战中，你将开发一个控制系统，以最小化微电网所有者的成本。

微电网是一个拥有本地供应源的小型电力用户网络，通常连接到一个中央电网，但能够独立运行。

微电网对于创建基于可再生资源的可持续和成本效益高的能源系统非常重要。

AI和优化方法可以用来提高微电网的运行效率。好的控制算法确保了可靠性和成本效益。

为了开发和测试你的系统，你将使用这个仓库中包含的一个微电网模拟器。

介绍性的演示文稿可以在这里找到

## Background

The Rye microgrid is a pilot within the EU research project REMOTE. 
It is a small microgrid placed at Langørgen, in the outskirts of Trondheim, 
and is a small energy system designed to supply electricity to a modern farm and three households. 
The REMOTE projects goal for Rye Microgrid is to run the system in islanded mode.

> A microgrid is said to be in islanded mode when it is disconnected from the main grid and 
> it operates independently with micro sources and load

The system has two sources of generation – a wind turbine and a rack of PV panels. 
In addition, the system has two storages – a battery with high charge and discharge response, but with limited storage and losses, 
and a hydrogen energy system, with lower charge and discharge rates, higher losses and storage capacity. 
When you want to charge the hydrogen system, electricity is used to run an electrolyser that makes hydrogen from water and stores the resulting hydrogen in a tank. 
The process can be reversed by producing electricity from hydrogen using a fuel cell. 
For simplicity, minimum charging levels and wear- and tear costs are disregarded in this context.
We also simplify and collect all losses in the conversion process to and from the storages as charge loss. These losses are given as the round trip efficiency in the table below. Thus there are only losses when charging the storages, not when discharging.

Morover, when local production or discharges from storages are not sufficient to cover the demand, the microgrid can draw electricity from
the grid at some costs.

Rye微电网是欧盟研究项目REMOTE的一个试点项目。它位于特隆赫姆的郊区Langørgen，是一个小型能源系统，设计用于为一个现代农场和三个家庭供电。

REMOTE项目的目标是让Rye微电网在孤岛模式下运行。当微电网从主电网断开并独立运行时，称为孤岛模式。

系统有两个发电源 - 一个风力发电机和一组光伏面板。

此外，系统有两个储能设备 - 一个电池，具有高充放电响应，但储能有限且有损耗；还有一个氢能系统，充放电速度较低，损耗较大，储能容量较大。

当你想给氢能系统充电时，电力被用来运行一个电解器，该电解器从水中制造氢，并将产生的氢储存在一个罐子里。这个过程可以通过使用燃料电池从氢中产生电力来逆转。

为了简化，我们在这里忽略了最低充电水平和磨损成本。我们也简化并将所有在储能设备充放电过程中的损耗作为充电损耗。这些损耗在下表中以往返效率给出。因此，只有在给储能设备充电时才有损耗，放电时没有损耗。

此外，当本地生产或储能设备的放电不足以满足需求时，微电网可以从电网中提取电力，但需要一些成本。

![microgrid](docs/microgrid.png)

## Task

The task of this assignment is to make a control system that minimizes the cost of operation of the microgrid, 
given the uncertainty of future consumption and generation from the wind turbine and PV. With clever operation of the microgrid, it should be possible to minimise the cost of grid imports. Your task is to develop a system that optimise the operation of the storages, given limited insight into future PV and wind generation and consumption.

The sole cost element is related to the import of electricity from the grid.
The cost of using electricity from the grid has 3 elements:
- An hourly, variable electricity **spot price** – given as part of the dataset as NOK/kWh 
- An energy part of the **grid tariff**, which is paid per kWh that is imported. In this assignment we use the Tensio winter energy tariff: 0.05 NOK/kWh
- A **peak tariff** that is paid monthly, based on the maximum instantaneous power (measured hourly) imported to the microgrid: 49 NOK/month/kWpeak

你的任务是开发一个控制系统，以最小化微电网的运行成本，考虑到未来消耗和风力涡轮机及光伏发电的不确定性。通过巧妙地操作微电网，应该可以最小化从电网导入的成本。你的任务是开发一个系统，优化储能设备的运行，考虑到对未来光伏和风力发电以及消耗的有限了解。

唯一的成本元素与从电网导入电力有关。

使用电网电力的成本有三个元素：

按小时计算的可变电力现货价格 - 作为数据集的一部分，以NOK/kWh给出
按导入的每kWh支付的电网电价的能源部分。在这个任务中，我们使用Tensio冬季能源电价：0.05 NOK/kWh
基于向微电网导入的最大瞬时功率（按小时测量）支付的峰值电价：49 NOK/月/kWpeak

The system is operated under the following restrictions:

- Consumption, PV and wind generation are all stochastic variables. 
  These can only be observed, not decided. 
  Weather (temperature, wind speed, solar radiation e.t.c. is a main driver behind these stochastic processes). 
- Consumption must be met in all timesteps – either through 
  wind generation, PV generation, discharge from storages or import from the grid. 
- Given that storages are not full, they can be charged by PV generation, wind generation, or import from the grid. 
  For simplicity, we assume that the energy being stored is equal to the charging of the storage units times the round trip efficiency, 
  and that the systems can be discharged without any losses. 
- The microgrid is too large to join any current Norwegian prosumer scheme, 
  thus excess production cannot be fed back into the grid. 
  If production is larger than consumption, and all storages full, 
  then excess production is curtailed (thrown away).
- All losses from transformers and distribution lines can be neglected.

消耗、光伏和风力发电都是随机变量。这些只能被观察，不能被决定。天气（温度、风速、太阳辐射等）是这些随机过程的主要驱动因素。

在所有时间步中，消耗必须得到满足 - 通过风力发电、光伏发电、储能设备的放电或从电网导入。

假设储能设备没有满，它们可以通过光伏发电、风力发电或从电网导入来充电。为了简化，我们假设被储存的能量等于储能设备的充电量乘以往返效率，且系统可以无损耗地放电。

微电网太大，无法加入任何现有的挪威产能者计划，因此，过量生产不能反馈到电网。如果生产大于消耗，且所有储能设备都满了，那么过量生产就会被削减（丢弃）。

可以忽略所有变压器和配电线的损失

The mathematical description can be found in this [document](docs/rye_simulator.pdf).

Technical data is found in the table below:

|System|Attribute|Value|
|---|---|---|
|Microgrid|Latitude|63°24'47.0"N|
| |Longitude|10°06'46.0"E|
|Wind turbine|Brand|VESTAS V27|
| |Max power|225 kW|
| |Hub Height|31.5 m|
| |Cut-in wind speed|3.5 m/s|
| |Rated wind speed|14 m/s|
|Photo Voltaic system|Brand|REC TwinPeak2|
| |Rated output power|86.4kWp|
|Battery Energy Storage System|Brand|Nidec/ LG Chem|
| |Storage capacity|500 kWh|
| |Charge/discharge capacity|400 kWh/h|
| |Round trip efficiency|85%|
|Hydrogen system|Storage capacity|1670 kWh|
| |Electrolyser (charge) capacity|55 kW|
| |Fuel cell (discharge) capacity|100 KW|
| |Round trip efficiency|32.5%|

微电网的地理位置是北纬63°24'47.0"，东经10°06'46.0"。

风力发电机的品牌是VESTAS V27，最大功率为225 kW，枢轴高度为31.5 m，切入风速为3.5 m/s，额定风速为14 m/s。

光伏系统的品牌是REC TwinPeak2，额定输出功率为86.4kWp。

电池储能系统的品牌是Nidec/ LG Chem，储能容量为500 kWh，充放电容量为400 kWh/h，往返效率为85%。

氢能系统的储能容量为1670 kWh，电解器（充电）容量为55 kW，燃料电池（放电）容量为100 KW，往返效率为32.5%

## Data description
Data is stored in `data/train.csv`.
It contains production and consumption measures and weather parameters for every hour of the training period.
This data you can use in the development of the controller.

Later in the event you will get `data/test.csv` for the test period used in final evaluation.
It has the same parameters but for another period. 
You should only use it in for running the evaluation script to get the final score.
数据存储在data/train.csv文件中。它包含了训练期间每小时的生产和消耗测量以及天气参数。你可以在开发控制器时使用这些数据。

在活动的后期，你将获得data/test.csv文件，该文件用于最终评估的测试期。它包含了相同的参数，但是对应于另一个时期。

你只应该在运行评估脚本以获得最终分数时使用data/test.csv文件。

Both files contain the following parameters (columns in the file):
|Paramemter|Description|
|---|---|
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

All timestamps in the data are stored in UTC (Coordinated Universal Time).
pv_production：太阳能板的产量，单位为kWh/h。
wind_production：风力涡轮机的产量，单位为kWh/h。
consumption：消耗量，单位为kWh/h。
spot_market_price：能源价格，单位为NOK/kWh。
所有数据中的时间戳都以UTC（协调世界时）存储。

## Guidelines for using the data

- Consumption, PV production and wind production are only known the hour after it takes place. Future values of consumption and production **can't be used** in the control strategy.
- Weather data are seen as weather forecast, thus you can include future values in your control strategy
- Spot prices becomes available in vectors of 24 hours once a day. The forecasts for next day is available at 13:00. Thus, future prices can be included 12-25 hours ahead in time, depending on the hour of the day.

For **development purposes only** you might want to assume that consumption and production observations are forecasts. This way part of the team can work on control policy and part on the forecasts. At the end replace observations with forecasts

消耗、光伏生产和风力生产只在发生后的一个小时内知道。未来的消耗和生产值不能用于控制策略。

天气数据被视为天气预报，因此你可以在你的控制策略中包含未来的值。

现货价格每天一次以24小时的向量形式变得可用。第二天的预测在13:00可用。因此，根据一天中的时间，未来的价格可以包含在12-25小时之后。

仅用于开发目的，你可能想假设消耗和生产观察值是预测。这样，团队的一部分可以在控制策略上工作，一部分在预测上工作。最后，用预测替换观察值。

## Development environment setup

This setup should work in Windows, Max and Linux.

1. Install miniconda for Python 3.8 64-bit from the [official page](https://docs.conda.io/en/latest/miniconda.html).
2. Setup conda environment defined in `environment.yaml`:
```sh
conda env create -f environment.yaml
```
3. Activate environment in the terminal:
```sh
conda activate rye-flex-env
```
4. Select `rye-flex-env` as your interpreter in your IDE:
- [VS Code](https://code.visualstudio.com/docs/python/environments)
- [PyCharm](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html)

5. To install addition packages use `pip install` or `conda install` commands in 
the terminal when the environment is activated. 
The environment includes Python packages necessary to run the simulator and load the data.
Here are some packages you might find useful:
- [scikit-learn](https://scikit-learn.org/stable) - machine learning 
- [pyomo](http://www.pyomo.org) - optimization modeling language
- [pytorch](https://pytorch.org) - machine and deep learning library
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) - reinforcement learning
- [deap](https://github.com/DEAP/deap) - evolutionary algorithms

## Simulator

This repository includes a microgrid simulator that you will be using for developing and testing your system.
The simulator is implemented as [OpenAI Gym](https://gym.openai.com/).
It is recommended to read the [official docs](https://gym.openai.com/docs/#observations) from OpenAI to get familiar with the basic concepts.
Even though gyms were introduced for training deep reinforcement learning agents, they provide a generic interface for any control system.

You can see example of a random agent using the microgrid environment gym in `scripts/random_action.py`.
Your task is to do better (ideally much better) than taking random actions.
Action and state (observation) variables are documented in `src/rye_flex_env/states.py`.
The environment itself is implemented in `src/rye_flex_env/env.py`.
It can be an advantage to understand what is going on in the code.
该存储库包含一个微电网模拟器，你将使用它来开发和测试你的系统。

模拟器实现为OpenAI Gym。建议阅读OpenAI的官方文档以熟悉基本概念。

尽管Gym最初是为训练深度强化学习代理而引入的，但它们为任何控制系统提供了一个通用接口。

你可以在scripts/random_action.py中看到一个使用微电网环境Gym的随机代理的例子。你的任务是做得比随机行动更好（理想情况下，要好得多）。

行动和状态（观察）变量在src/rye_flex_env/states.py中有文档记录。

环境本身在src/rye_flex_env/env.py中实现。理解代码中发生的事情可能会有优势。

## Evaluation

Evaluation will be done with the evaluation script `scripts/evaluation.py`.
It uses `data/test.csv` data file that will be shared (pushed to the repo) with the participants on **Saturday at 18:00**.
The participants are expected to add the code for using their system agent in this script 
and then run it once to get the score. 
Don't modify the reward function or do it in a way to preserve the original cumulative reward.
The score should be included in your presentation.
Also the code to reproduce both training and testing of your system should be pushed to your team GitHub repo
and shared with the judges.

Evaluation criteria:
- Score = cumulative reward for test period​
  - Reproducibility - we can run your code to get the same/similar score​
- Methodology​
  - Method selection​
  - System design​
  - System implementation​
- Presentation​
  - Show your score​
  - Plots with actions and environment state​
  - Explanation and justification of choices made

评估将使用scripts/evaluation.py评估脚本进行。它使用将在周六18:00与参与者共享（推送到仓库）的data/test.csv数据文件。

参与者需要在此脚本中添加使用他们的系统代理的代码，然后运行一次以获得分数。不要修改奖励函数，或者以保留原始累积奖励的方式进行修改。

分数应包含在你的演示中。

用于复现你的系统训练和测试的代码应推送到你的团队GitHub仓库，并与评委共享。

评估标准：

分数 = 测试期间的累积奖励
可复现性 - 我们可以运行你的代码以获得相同/类似的分数
方法论
方法选择
系统设计
系统实现
演示
展示你的分数
带有动作和环境状态的图表
对所做选择的解释和理
**Happy hacking!**


• 可以从电网中无限制地获取电力 
• 没有电池磨损和撕裂成本 
• 没有氢气磨损和撕裂成本，没有最低充电或放电水平
• 电网费用由每单位能源成本和峰值费用组成，峰值费用必须支付当月的最大峰值 
• 可以从电网中获取电力，但不能注入电网，因为微电网没有生产许可证

决策变量使用小写字母，参数使用大写字母 
参数： 
• Pt - 第t小时的现货市场价格 - NOK/kWh 
• Genergy - 每单位能源的电网费用 - NOK/kWh 
• Gpeak - 峰值电网费用 - NOK/kW/月 
• Ct - 每小时消耗量 - kWh/h 

决策变量： 
• sbattery

电池的充电水平 - kWh t 
• shydrogen
氢气的充电水平 - kWh


目标是最小化微电网的能源成本 minTotalCost = sumt (Pt + Genergy) * gridimportt + Gpeak * maxt (gridimportt)

翻译为：

最小总成本 = Σt (Pt + Genergy) * gridimportt + Gpeak * maxt (gridimportt)

其中：

Σt 表示对所有时间t的求和
Pt 是第t小时的现货市场价格
Genergy 是每单位能源的电网费用
gridimportt 是第t时间的电网输入
Gpeak 是峰值电网费用
maxt (gridimportt) 是所有时间t的电网输入的最大值


消耗和电力供应之间必须立即平衡： W indt + P Vt + dischargebattery

dischargehydrogen
gridimportt = t t Ct + chargebattery
chargehydrogen
curtailmentt t t (2) 
在电能转化为化学能并再转化回电能的过程中会有能量损失。损失为15%。 sbattery = 0.85 ∗ chargebattery − dischargebattery t t t+1 (3) 
在氢能系统中，由于电解和燃料电池过程会有能量损失。损失为32.5%。 shydrogen = 0.325 ∗ chargehydrogen − dischargehydrogen t t t+1 (4) 
充放电和状态必须在电池限制内 0 ≤ sbattery ≤ sbattery,MAX t(5) 
0 ≤ dischargebattery ≤ dischargebattery,MAX t(6) 
0 ≤ chargebattery ≤ chargebattery,MAX t(7) 
同样的限制也适用于氢能系统： 0 ≤ shydrogen ≤ shydrogen,MAX t(8) 
≤ dischargehydrogen,MAX 0 ≤ dischargehydrogen t(9) 
≤ chargehydrogen,MAX 0 ≤ chargehydrogen t