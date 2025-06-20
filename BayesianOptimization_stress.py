from bayes_opt import BayesianOptimization
import numpy as np
import scipy.io as sio
import json
import matplotlib.pyplot as plt

# 指定要导入的索引
selected_indices = [60, 570]
# 全局变量
U = []
V = []
U0 = None
V0 = None
Nx = 1
Ny = 1
X = [200, -200]
Y = [-100, 100]
# weight_force = 0.00005 #函数返回值中受力平衡所占权重

# 添加矩阵
def read_mat_rows(file_path, variable_name, row_indices):
    try:
        mat_data = sio.loadmat(file_path)
        data = mat_data.get(variable_name)
        if data is None:
            print(f"未找到变量 {variable_name}。")
            return None
        if len(data.shape) != 2:
            print("数据不是二维数组，无法按行索引。")
            return None
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
    U0 = read_mat_rows('for_dngo/sim_data_dist.mat', 'u_sim', 0)
    V0 = read_mat_rows('for_dngo/sim_data_dist.mat', 'u_sim', 1)

    _U0 = np.zeros_like(U0)
    _V0 = np.zeros_like(V0)
    for i in range(len(X) + len(Y)):
        if i < len(X):
            _U0 += X[i] * U[i]
            _V0 += X[i] * V[i]
        else:
            _U0 += Y[i - len(X)] * U[i]
            _V0 += Y[i - len(X)] * V[i]

    U0 = _U0
    V0 = _V0
    # 123

    # 添加噪声
    noise_u = read_mat_rows('for_dngo/noise1th_0.003max_force0.027595.mat', 'eu', 0)
    noise_v = read_mat_rows('for_dngo/noise1th_0.003max_force0.027595.mat', 'eu', 1)
    U0 += noise_u
    V0 += noise_v

def best(stress):  # 计算y值，U0需已知，U需已知
    global U, V, U0, V0
    # 初始化加权和矩阵
    weighted_sum_u = np.zeros_like(U0)
    weighted_sum_v = np.zeros_like(V0)
    sumx = 0
    sumy = 0
    # 计算加权和
    for i in range(len(stress)):
        if i<len(selected_indices):
            weighted_sum_u += stress[i] * U[i] * Nx
            weighted_sum_v += stress[i] * V[i] * Nx
            sumx += stress[i]
        else:
            weighted_sum_u += stress[i] * U[i] * Ny
            weighted_sum_v += stress[i] * V[i] * Ny
            sumy += stress[i]
    # print(weighted_sum_u)
    # print(weighted_sum_v)
    y = np.sum((weighted_sum_u - U0) ** 2)
    y += np.sum((weighted_sum_v - V0) ** 2)
    if y==0:
        return np.inf
    return 1 / y  # 由于DNGO算法要求最大化，所以这里返回1/y
    # return (1 / y - weight_force * (sumx ** 2 + sumy ** 2))


def objective(**params):
    stress = [params[f'x{i + 1}'] for i in range(len(selected_indices) * 2)]
    result = best(stress)
    print(f"Result: {result}, Params: {params}")

    with open("Xy.json", "a") as f:
        json.dump((float(result), params), f)
        f.write('\n')
    return result


if __name__ == '__main__':
    load_global_matrices()

    N_low = [0, -400, -400, 0]
    N_high = [400, 0, 0, 400]

    pbounds = {f'x{i + 1}': (N_low[i], N_high[i]) for i in range(len(selected_indices) * 2)}

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        verbose=2,
        random_state=1
    )

    # 开始优化
    optimizer.maximize(init_points=100, n_iter=1000)

    # 手动处理优化历史
    max_eff_list = []  # 最大效率历史
    param_history = []  # 参数优化历史
    max_eff_indices = []
    current_max = -np.inf  # 当前已知最大效率
    i=0

    # 遍历所有迭代结果，手动记录历史
    for res in optimizer.res:
        current_result = res['target']
        if current_result > current_max:
            current_max = current_result
            max_eff_indices.append(i)
            # 保存当前最优参数
            params = list(res['params'].values())
            optimized_params = [
                p * Nx if i < len(selected_indices) else p * Ny
                for i, p in enumerate(params)
            ]
            param_history.append(optimized_params)
        max_eff_list.append(current_max)
        i+=1

    # 输出最优结果
    max_result = optimizer.max
    max_efficiency = max_result['target']
    max_params = [
        x * Nx if i < len(selected_indices) else x * Ny
        for i, x in enumerate(list(max_result['params'].values()))
    ]

    with open("max_efficiency.json", "w") as f:
        json.dump({
            "max_efficiency": float(max_efficiency),
            "thicknesses": max_params,
            "positions": selected_indices
        }, f, indent=4)

    # 绘制迭代次数-最大效率图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(max_eff_list) + 1), max_eff_list, marker='o')
    plt.xlabel('number of iterations')
    plt.ylabel('f(x)')
    plt.title('Bayesian optimization history')
    plt.grid(True)
    plt.savefig('Bayesian_optimization_history.png')
    plt.close()

    true_params = X + Y  # 合并X和Y为完整的真实参数列表

    # 绘制参数优化历史
    if param_history:  # 确保有记录的参数历史
        param_history_array = np.array(param_history)
        num_params = param_history_array.shape[1]
        colors = plt.cm.tab20(np.linspace(0, 1, min(num_params, 20)))

        plt.figure(figsize=(14, 8))
        for i in range(num_params):
            direction = 'X' if i < len(selected_indices) else 'Y'
            index = i % len(selected_indices) + 1

            # 绘制优化参数历史
            plt.plot(range(1, len(param_history) + 1), param_history_array[:, i],
                     marker='o', linewidth=2, color=colors[i % 20],
                     label=f'{direction}{index} (Optimized)')
            # 绘制真实值参考线
            plt.axhline(y=true_params[i], color=colors[i % 20], linestyle='--',
                        linewidth=1.5, alpha=0.7, label=f'{direction}{index} (True)')

        plt.legend(
            bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True,
            framealpha=0.8, ncol=2, fontsize=9, borderpad=0.5, labelspacing=0.3
        )
        # x轴标注为迭代次数
        plt.xticks(range(1, len(param_history) + 1),
                   [f'Iter {idx + 1}' for idx in max_eff_indices],
                   rotation=45)

        plt.xlabel('Iterations where best value was updated')
        plt.ylabel('Parameter values')
        plt.title('Parameter Evolution When Best Value Was Updated')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('bayesian_parameter_evolution.png', dpi=300)
        plt.close()
    else:
        print("警告：未记录到参数优化历史，可能没有更新的最优值。")

    print(true_params)
    print(best(true_params))
    print("Optimization complete. Results saved as follows:")
    print("计算次数-目标函数值图片(优化历史): Bayesianoptimization_history.png")
    print("计算次数-参数变化图片(优化历史): bayesian_parameter_evolution.png")
