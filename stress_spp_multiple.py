from unicodedata import ucd_3_2_0
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

from BayesianOptimization_stress import selected_indices

matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import json
import scipy.io as sio

# 指定要导入的索引
# selected_indices = [60, 195, 322, 511, 639, 767]
selected_indices = [60, 570]
# 全局变量
U = []
V = []
U0 = None
V0 = None
Ny = 1
Nx = 1
# X = [200, 350, 150, -50, -300, -350]
# Y = [100, -50, 300, -200, -250, 100]
X = [200, -200]
Y = [100, -100]

# 添加矩阵
def read_mat_rows(file_path, variable_name, row_indices):
    try:
        # 加载.mat文件
        mat_data = sio.loadmat(file_path)
        # 获取指定变量的数据
        data = mat_data.get(variable_name)
        if data is None:
            print(f"未找到变量 {variable_name}。")
            return None
        # 检查数据是否为二维数组
        if len(data.shape) != 2:
            print("数据不是二维数组，无法按行索引。")
            return None
        # 提取指定行的数据
        selected_rows = data[row_indices, :]
        return selected_rows
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到。")
    except Exception as e:
        print(f"发生未知错误: {e}")
    return None


def load_global_matrices():
    global U, V, U0, V0
    for index in selected_indices:
        U.append(read_mat_rows('for_dngo/x_FX/' + f'{index}.mat', 'u', 0))
        V.append(read_mat_rows('for_dngo/x_FX/' + f'{index}.mat', 'u', 1))
    for index in selected_indices:
        U.append(read_mat_rows('for_dngo/y_FY/' + f'{index}.mat', 'u', 0))
        V.append(read_mat_rows('for_dngo/y_FY/' + f'{index}.mat', 'u', 1))
    # 加载U0矩阵
    U0 = read_mat_rows('for_dngo/sim_data_dist.mat', 'u_sim', 0)
    V0 = read_mat_rows('for_dngo/sim_data_dist.mat', 'u_sim', 1)

    # _U0与_V0为自拟得到的位移矩阵，未加噪声，真实数据为U0与V0

    _U0 = np.zeros_like(U0)
    _V0 = np.zeros_like(V0)
    for i in range(len(X)+len(Y)):
        if i<len(X):
            _U0 += X[i] * U[i]
            _V0 += X[i] * V[i]
        else:
            _U0 += Y[i-len(X)] * U[i]
            _V0 += Y[i-len(X)] * V[i]

    U0 = _U0
    V0 = _V0
    '''
    # 添加噪声
    noise_u = read_mat_rows('for_dngo/noise1th_0.001max_force0.027595.mat','eu',0)
    noise_v = read_mat_rows('for_dngo/noise1th_0.001max_force0.027595.mat','eu',1)
    U0 += noise_u
    V0 += noise_v
    '''

def best(stress):  # 计算y值，U0需已知，U需已知
    global U, V, U0, V0
    # 初始化加权和矩阵
    weighted_sum_u = np.zeros_like(U0)
    weighted_sum_v = np.zeros_like(V0)
    # 计算加权和
    for i in range(len(stress)):
        if i<len(selected_indices):
            weighted_sum_u += stress[i] * U[i] * Nx
            weighted_sum_v += stress[i] * V[i] * Nx
        else:
            weighted_sum_u += stress[i] * U[i] * Ny
            weighted_sum_v += stress[i] * V[i] * Ny
    # print(weighted_sum_u)
    # print(weighted_sum_v)
    y = np.sum((weighted_sum_u - U0) ** 2)
    y += np.sum((weighted_sum_v - V0) ** 2)
    if y==0:
        return np.inf
    return 1 / y  # 由于DNGO算法要求最大化，所以这里返回1/y



class Spp:
    def __init__(self, layers):
        self.layers = layers
        # 为每一个力元设置大小限制
        self.pbounds = {}
        for i, _ in enumerate(layers):
            self.pbounds[f'x{i + 1}'] = (N_low[i], N_high[i])  # 对各力元大小的默认限制（-1,1）

    def evaluate(self, **params):
        stress = [params[f'x{i + 1}'] for i in range(len(self.layers))]
        result = best(stress)
        print(f"Result: {result}, Params: {params}")

        with open("Xy.json", "a") as f:
            json.dump((float(result), params), f)
            f.write('\n')
        return result


# DNGO模型定义
class DNGO(nn.Module):
    def __init__(self, input_dim, hidden_dim=50):
        super(DNGO, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        output = self.fc3(x)
        return output


# DNGO优化器
class DNGOptimizer:
    def __init__(self, f, pbounds, input_dim, hidden_dim=50, learning_rate=1e-3):
        self.f = f
        self.pbounds = pbounds
        self.input_dim = input_dim
        self.model = DNGO(input_dim, hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.samples = []
        self.targets = []

    def maximize(self, init_points=5, n_iter=25):
        # 初始化样本点
        for _ in range(init_points):
            params = {k: int(np.random.uniform(*v)) for k, v in self.pbounds.items()}  # 随机选取init_points个参数并保留整数
            result = self.f(**params)
            self.samples.append(list(params.values()))
            self.targets.append(result)

        # 开始训练和优化过程
        for _ in range(n_iter):
            self.train_model()
            next_params = self.suggest_next_params()
            next_result = self.f(**next_params)
            self.samples.append(list(next_params.values()))
            self.targets.append(next_result)
            print(f"Iteration {_ + 1}, Params: {next_params}, Result: {next_result}")

    def train_model(self, epochs=1000):
        x_train = torch.tensor(self.samples, dtype=torch.float32)
        y_train = torch.tensor(self.targets, dtype=torch.float32).view(-1, 1)
        self.model.train()

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            predictions = self.model(x_train)
            loss = self.loss_fn(predictions, y_train)
            loss.backward()
            self.optimizer.step()

    def suggest_next_params(self):
        # 随机搜索最优参数
        best_score = float('-inf')
        best_params = None
        for _ in range(5000):  # 随机搜索5000次，找到最优参数
            params = {k: int(np.random.uniform(*v)) for k, v in self.pbounds.items()}  # 随机选取参数并保留整数
            params_list = torch.tensor(list(params.values()), dtype=torch.float32)
            with torch.no_grad():
                pred = self.model(params_list).item()
            if pred > best_score:
                best_score = pred
                best_params = params
        return best_params


if __name__ == '__main__':
    # 加载全局矩阵
    load_global_matrices()
    global N_low, N_high
    # 定义力元结构
    N_low = [-400, -400, -400, -400]
    N_high = [400, 400, 400, 400]
    l_low = [-300,-200,-150,-100,-100,-50,-50]
    l_high = [300,200,150,100,100,50,50]

    max_eff_list = []  # 最大效率历史
    max_param_list = []  # 对应参数配置历史
    max_eff_indices = []  # 对应迭代索引
    current_max = -np.inf  # 当前已知最大效率
    num = 6

    for numbers in range(0,num):
        layers = []
        for _ in range(len(selected_indices) * 2):
            layers.append(0.1)  # 共len(selected_indices) * 2个外力，可根据U矩阵数量进行调整

        sim = Spp(layers)

        # 定义目标函数
        def objective(**params):
            return sim.evaluate(**params)

        # 初始化 DNGO 优化器
        optimizer = DNGOptimizer(
            f=objective,
            pbounds=sim.pbounds,
            input_dim=len(sim.pbounds)
        )

        init_=100
        n_=250

        optimizer.maximize(init_points=init_, n_iter=n_)  # 初始点和迭代次数可以根据需要调整

        max_index = np.argmax(optimizer.targets)
        max_efficiency = optimizer.targets[max_index]
        max_params = []
        new_low=[]
        new_high=[]
        s = optimizer.samples[max_index]
        for i, x in enumerate(s):
            if i < len(selected_indices):  # 前len(selected_indices)个元素
                new_low.append(max(x * Nx + l_low[numbers],-400))
                new_high.append(min(x * Ny + l_high[numbers],400))
                max_params.append(x * Nx)
            else:  # 其余元素
                new_low.append(max(x * Nx + l_low[numbers], -400))
                new_high.append(min(x * Ny + l_high[numbers], 400))
                max_params.append(x * Ny)
        N_low = new_low
        N_high = new_high
        print(N_low, N_high)

        # 动态跟踪最大效率及其对应参数
        for i, res in enumerate(optimizer.targets):
            if res > current_max:  # 当发现更高效率时
                current_max = res
                max_eff_indices.append(i+numbers*(init_+n_))  # 记录刷新最大效率的迭代索引

                # 获取当前迭代的参数配置（考虑缩放）
                params = optimizer.samples[i]
                current_best_params = []
                for j, x in enumerate(params):
                    if j < len(selected_indices):  # X方向参数
                        current_best_params.append(x * Nx)
                    else:  # Y方向参数
                        current_best_params.append(x * Ny)
                max_param_list.append(current_best_params)  # 记录最佳参数
            max_eff_list.append(current_max)  # 记录当前已知的最大效率

    final_eff=max_eff_list[-1]
    final_params=max_param_list[-1]

    # 保存最大效率到文件
    with open("max_efficiency.json", "w") as f:
        json.dump({
            "max_efficiency": float(final_eff),
            "thicknesses": final_params,
            "positions": selected_indices
        }, f, indent=4)

    # 绘制迭代次数-最大效率图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(max_eff_list) + 1), max_eff_list, marker='o')
    plt.xlabel('Number of iterations')
    plt.ylabel('Best objective value')
    plt.title('Optimization History')
    plt.grid(True)
    plt.savefig('optimization_history.png')
    plt.close()

    true_params = X + Y  # 合并X和Y为完整的真实参数列表

    # 绘制最大效率对应的参数配置变化
    if max_param_list:
            plt.figure(figsize=(14, 8))
            param_history_array = np.array(max_param_list)
            num_params = param_history_array.shape[1]

            # 颜色循环，用于区分不同参数
            colors = plt.cm.tab20(np.linspace(0, 1, min(num_params, 20)))

            # 为每个参数绘制单独的线条
            for i in range(num_params):
                direction = 'X' if i < len(selected_indices) else 'Y'
                index = i % len(selected_indices) + 1

                plt.plot(range(1, len(max_param_list) + 1), param_history_array[:, i],
                         marker='o', linewidth=2, color=colors[i % 20],
                         label=f'{direction}{index} (Optimized)')
                # 绘制真实值参考线（虚线）
                plt.axhline(y=true_params[i], color=colors[i % 20], linestyle='--',
                        linewidth=1.5, alpha=0.7, label=f'{direction}{index} (True)')
            plt.legend(
                bbox_to_anchor=(1.05, 1),  # 图例放在右上角外部
                loc='upper left',  # 图例左上角对齐
                frameon=True,  # 显示图例边框
                framealpha=0.8,  # 边框透明度
                ncol=2,  # 分两列显示，减少高度
                fontsize=9,  # 适当缩小字体
                borderpad=0.5,  # 内边距
                labelspacing=0.3  # 标签间距
            )
            # 在x轴上标注对应的迭代次数
            plt.xticks(range(1, len(max_param_list) + 1),
                       [f'Iter {idx + 1}' for idx in max_eff_indices],
                       rotation=45)

            plt.xlabel('Iterations where best value was updated')
            plt.ylabel('Parameter values')
            plt.title('Parameter Evolution When Best Value Was Updated')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig('parameter_evolution.png', dpi=300)
            plt.close()

    ans = X + Y
    print(ans)
    print(U0,V0)
    print(best(ans))
    print("Optimization complete. Results saved as follows:")
    print("计算次数-目标函数值图片(优化历史): optimization_history.png")
    print("计算次数-参数变化图片(优化历史): parameter_evolution.png")