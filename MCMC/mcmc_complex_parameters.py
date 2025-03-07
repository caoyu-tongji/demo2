# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm, gaussian_kde
from matplotlib.colors import LinearSegmentedColormap

# 设置随机种子，确保结果可重现
np.random.seed(42)

# 设置绘图风格
sns.set_style('whitegrid')
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'sans-serif']  # 设置多个备选字体
plt.rcParams['axes.unicode_minus'] = False  # 用于显示负号


# 1. 定义参数的先验分布
def prior_a(a):
    """
    参数a的先验分布 (正态分布)
    """
    return stats.norm.pdf(a, loc=0, scale=2)


def prior_b(b):
    """
    参数b的先验分布 (双峰分布，使用两个正态分布的混合)
    """
    # 双峰分布：两个正态分布的混合
    peak1 = stats.norm.pdf(b, loc=-2, scale=0.5)
    peak2 = stats.norm.pdf(b, loc=2, scale=0.5)
    return 0.5 * peak1 + 0.5 * peak2


# 2. 定义参数a和b如何影响参数c的模型
def model_c(a, b):
    """
    根据参数a和b计算参数c的理论值
    """
    # 这里定义a和b如何影响c的关系
    # 例如：c = a^2 + b^2 + a*b
    return a + b**2 + a*b


# 3. 定义似然函数
def likelihood(c_observed, a, b, sigma_c=0.1):
    """
    计算给定参数a和b下观测到c_observed的似然
    
    参数:
        c_observed: 观测到的c值
        a, b: 参数值
        sigma_c: 观测误差的标准差
    """
    # 计算理论上的c值
    c_theory = model_c(a, b)
    
    # 假设观测误差服从正态分布
    return stats.norm.pdf(c_observed, loc=c_theory, scale=sigma_c)


# 4. 定义后验分布（未归一化）
def posterior_unnormalized(c_observed, a, b):
    """
    计算未归一化的后验概率
    后验 ∝ 似然 × 先验
    """
    return likelihood(c_observed, a, b) * prior_a(a) * prior_b(b)


# 5. 实现Metropolis-Hastings算法
def metropolis_hastings(c_observed, n_iterations=20000, burn_in=5000):
    """
    使用Metropolis-Hastings算法对参数a和b进行MCMC采样
    
    参数:
        c_observed: 观测到的c值
        n_iterations: 迭代次数
        burn_in: 预热期迭代次数
    
    返回:
        采样得到的参数链
    """
    # 初始化参数
    current_a = 0.0  # 初始猜测值
    current_b = 0.0  # 初始猜测值
    
    # 初始化参数链
    chain = np.zeros((n_iterations, 2))  # 存储a和b
    
    # 计算初始后验概率
    current_posterior = posterior_unnormalized(c_observed, current_a, current_b)
    
    # 设置建议分布的标准差
    proposal_width_a = 0.3
    proposal_width_b = 0.3
    
    # 接受率统计
    accepts = 0
    
    # 使用tqdm创建进度条
    for i in tqdm(range(n_iterations), desc="MCMC采样进度"):
        # 从建议分布中抽取新的参数值
        proposed_a = current_a + np.random.normal(0, proposal_width_a)
        proposed_b = current_b + np.random.normal(0, proposal_width_b)
        
        # 计算新参数的后验概率
        proposed_posterior = posterior_unnormalized(c_observed, proposed_a, proposed_b)
        
        # 计算接受概率
        acceptance_ratio = proposed_posterior / current_posterior if current_posterior > 0 else 1
        
        # 决定是否接受新参数
        if np.random.rand() < acceptance_ratio:
            current_a = proposed_a
            current_b = proposed_b
            current_posterior = proposed_posterior
            accepts += 1
        
        # 存储当前参数
        chain[i, 0] = current_a
        chain[i, 1] = current_b
    
    # 计算接受率
    acceptance_rate = accepts / n_iterations
    print(f"接受率: {acceptance_rate:.2f}")
    
    # 返回burn-in后的链
    return chain[burn_in:]


# 6. 生成真实参数和观测数据
def generate_true_parameters_and_data():
    """
    生成真实的参数a、b和对应的c值
    
    返回:
        true_a, true_b, observed_c
    """
    # 生成真实参数
    true_a = np.random.normal(0, 1)  # 从正态分布生成a
    
    # 从双峰分布生成b
    if np.random.rand() < 0.5:
        true_b = np.random.normal(-2, 0.5)  # 第一个峰
    else:
        true_b = np.random.normal(2, 0.5)   # 第二个峰
    
    # 计算真实的c值
    true_c = model_c(true_a, true_b)
    
    # 添加观测噪声
    observed_c = true_c + np.random.normal(0, 0.1)
    
    return true_a, true_b, observed_c


# 7. 分析和可视化结果
def analyze_results(chain, c_observed, true_a=None, true_b=None):
    """
    分析MCMC采样结果并可视化
    
    参数:
        chain: MCMC采样链
        c_observed: 观测到的c值
        true_a, true_b: 真实参数值（如果已知）
    """
    # 提取参数估计值
    a_chain = chain[:, 0]
    b_chain = chain[:, 1]
    
    # 计算后验均值和95%可信区间
    a_estimate = np.mean(a_chain)
    b_estimate = np.mean(b_chain)
    a_ci = np.percentile(a_chain, [2.5, 97.5])
    b_ci = np.percentile(b_chain, [2.5, 97.5])
    
    print(f"\n参数估计结果:")
    print(f"参数a: {a_estimate:.2f}, 95%可信区间: [{a_ci[0]:.2f}, {a_ci[1]:.2f}]")
    print(f"参数b: {b_estimate:.2f}, 95%可信区间: [{b_ci[0]:.2f}, {b_ci[1]:.2f}]")
    
    if true_a is not None and true_b is not None:
        print(f"\n真实参数值:")
        print(f"真实参数a: {true_a:.2f}")
        print(f"真实参数b: {true_b:.2f}")
        print(f"真实参数c: {model_c(true_a, true_b):.2f}")
    
    print(f"观测参数c: {c_observed:.2f}")
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 参数轨迹图
    axes[0, 0].plot(a_chain, label='a')
    if true_a is not None:
        axes[0, 0].axhline(true_a, color='r', linestyle='--', label='真实a')
    axes[0, 0].set_title('参数a轨迹')
    axes[0, 0].set_xlabel('迭代次数')
    axes[0, 0].set_ylabel('a值')
    axes[0, 0].legend()
    
    axes[0, 1].plot(b_chain, label='b')
    if true_b is not None:
        axes[0, 1].axhline(true_b, color='r', linestyle='--', label='真实b')
    axes[0, 1].set_title('参数b轨迹')
    axes[0, 1].set_xlabel('迭代次数')
    axes[0, 1].set_ylabel('b值')
    axes[0, 1].legend()
    
    # 2. 参数后验分布
    sns.histplot(a_chain, kde=True, ax=axes[1, 0])
    if true_a is not None:
        axes[1, 0].axvline(true_a, color='r', linestyle='--', label='真实值')
    axes[1, 0].axvline(a_estimate, color='g', linestyle='-', label='后验均值')
    axes[1, 0].axvline(a_ci[0], color='g', linestyle=':', label='95%可信区间')
    axes[1, 0].axvline(a_ci[1], color='g', linestyle=':')
    axes[1, 0].set_title('参数a后验分布')
    axes[1, 0].set_xlabel('a')
    axes[1, 0].legend()
    
    sns.histplot(b_chain, kde=True, ax=axes[1, 1])
    if true_b is not None:
        axes[1, 1].axvline(true_b, color='r', linestyle='--', label='真实值')
    axes[1, 1].axvline(b_estimate, color='g', linestyle='-', label='后验均值')
    axes[1, 1].axvline(b_ci[0], color='g', linestyle=':', label='95%可信区间')
    axes[1, 1].axvline(b_ci[1], color='g', linestyle=':')
    axes[1, 1].set_title('参数b后验分布')
    axes[1, 1].set_xlabel('b')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('MCMC/complex_results_posterior.png', dpi=300)
    
    # 3. 参数联合分布
    plt.figure(figsize=(10, 8))
    
    # 使用KDE绘制联合分布
    sns.kdeplot(x=a_chain, y=b_chain, cmap="viridis", fill=True)
    
    # 标记真实值和估计值
    if true_a is not None and true_b is not None:
        plt.plot(true_a, true_b, 'r*', markersize=15, label='真实参数')
    plt.plot(a_estimate, b_estimate, 'go', markersize=10, label='后验均值')
    
    # 添加等高线，显示c的理论值
    a_range = np.linspace(min(a_chain) - 1, max(a_chain) + 1, 100)
    b_range = np.linspace(min(b_chain) - 1, max(b_chain) + 1, 100)
    A, B = np.meshgrid(a_range, b_range)
    C = model_c(A, B)
    
    # 绘制等高线
    contour = plt.contour(A, B, C, colors='black', alpha=0.5, levels=10)
    plt.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
    
    plt.title('参数a和b的联合后验分布与c的等值线')
    plt.xlabel('a')
    plt.ylabel('b')
    plt.legend()
    plt.savefig('MCMC/complex_results_joint.png', dpi=300)
    
    # 4. 先验分布与后验分布比较
    plt.figure(figsize=(12, 10))
    
    # 绘制参数a的先验和后验分布
    plt.subplot(2, 1, 1)
    a_grid = np.linspace(-6, 6, 1000)
    prior_a_values = [prior_a(a_val) for a_val in a_grid]
    
    plt.plot(a_grid, prior_a_values, 'b--', label='先验分布')
    sns.kdeplot(a_chain, label='后验分布', color='g')
    if true_a is not None:
        plt.axvline(true_a, color='r', linestyle='-', label='真实值')
    plt.title('参数a的先验与后验分布比较')
    plt.xlabel('a')
    plt.ylabel('密度')
    plt.legend()
    
    # 绘制参数b的先验和后验分布
    plt.subplot(2, 1, 2)
    b_grid = np.linspace(-6, 6, 1000)
    prior_b_values = [prior_b(b_val) for b_val in b_grid]
    
    plt.plot(b_grid, prior_b_values, 'b--', label='先验分布')
    sns.kdeplot(b_chain, label='后验分布', color='g')
    if true_b is not None:
        plt.axvline(true_b, color='r', linestyle='-', label='真实值')
    plt.title('参数b的先验与后验分布比较')
    plt.xlabel('b')
    plt.ylabel('密度')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('MCMC/complex_results_prior_posterior.png', dpi=300)
    
    plt.show()


# 8. 主函数
def main():
    # 场景1：已知真实参数，生成观测数据
    print("场景1：已知真实参数，生成观测数据")
    true_a, true_b, observed_c = generate_true_parameters_and_data()
    print(f"生成的真实参数: a={true_a:.2f}, b={true_b:.2f}")
    print(f"观测到的参数c: {observed_c:.2f}")
    
    # 运行MCMC采样
    print("\n开始MCMC采样...")
    chain = metropolis_hastings(observed_c, n_iterations=20000, burn_in=5000)
    
    # 分析结果
    print("\n分析采样结果...")
    analyze_results(chain, observed_c, true_a, true_b)
    
    # 场景2：只知道观测值c，不知道真实参数a和b
    print("\n\n场景2：只知道观测值c，推断参数a和b")
    # 假设我们只观测到c值
    c_only = 10.0  # 这里可以设置一个具体的c值
    print(f"只观测到参数c: {c_only:.2f}")
    
    # 运行MCMC采样
    print("\n开始MCMC采样...")
    chain_unknown = metropolis_hastings(c_only, n_iterations=20000, burn_in=5000)
    
    # 分析结果
    print("\n分析采样结果...")
    analyze_results(chain_unknown, c_only)  # 不提供真实参数


# 执行主函数
if __name__ == "__main__":
    main()