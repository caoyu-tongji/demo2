# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from tqdm import tqdm

# 设置随机种子，确保结果可重现
np.random.seed(42)

# 设置绘图风格
sns.set_style('whitegrid')
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'sans-serif']  # 设置多个备选字体
plt.rcParams['axes.unicode_minus'] = False  # 用于显示负号


# 1. 定义真实模型和生成模拟数据
def generate_data(n_samples=100, true_mu=5.0, true_sigma=2.0):
    """
    生成来自正态分布的模拟数据
    
    参数:
        n_samples: 样本数量
        true_mu: 真实均值
        true_sigma: 真实标准差
    
    返回:
        模拟数据数组
    """
    return np.random.normal(true_mu, true_sigma, n_samples)


# 2. 定义先验分布
def prior_mu(mu):
    """
    均值参数的先验分布 (正态分布)
    """
    return stats.norm.pdf(mu, loc=0, scale=10)


def prior_sigma(sigma):
    """
    标准差参数的先验分布 (半正态分布)
    """
    if sigma <= 0:
        return 0
    return stats.halfnorm.pdf(sigma, loc=0, scale=10)


# 3. 定义似然函数
def likelihood(data, mu, sigma):
    """
    计算给定参数下观测数据的似然
    """
    if sigma <= 0:
        return 0
    return np.prod(stats.norm.pdf(data, loc=mu, scale=sigma))


# 4. 定义后验分布（未归一化）
def posterior_unnormalized(data, mu, sigma):
    """
    计算未归一化的后验概率
    后验 ∝ 似然 × 先验
    """
    return likelihood(data, mu, sigma) * prior_mu(mu) * prior_sigma(sigma)


# 5. 实现Metropolis-Hastings算法
def metropolis_hastings(data, n_iterations=10000, burn_in=1000):
    """
    使用Metropolis-Hastings算法对参数进行MCMC采样
    
    参数:
        data: 观测数据
        n_iterations: 迭代次数
        burn_in: 预热期迭代次数
    
    返回:
        采样得到的参数链
    """
    # 初始化参数
    current_mu = np.mean(data)  # 以数据均值作为初始值
    current_sigma = np.std(data)  # 以数据标准差作为初始值
    
    # 初始化参数链
    chain = np.zeros((n_iterations, 2))  # 存储mu和sigma
    
    # 计算初始后验概率
    current_posterior = posterior_unnormalized(data, current_mu, current_sigma)
    
    # 设置建议分布的标准差
    proposal_width_mu = 0.5
    proposal_width_sigma = 0.5
    
    # 接受率统计
    accepts = 0
    
    # 使用tqdm创建进度条
    for i in tqdm(range(n_iterations), desc="MCMC采样进度"):
        # 从建议分布中抽取新的参数值
        proposed_mu = current_mu + np.random.normal(0, proposal_width_mu)
        proposed_sigma = current_sigma + np.random.normal(0, proposal_width_sigma)
        
        # 计算新参数的后验概率
        proposed_posterior = posterior_unnormalized(data, proposed_mu, proposed_sigma)
        
        # 计算接受概率
        acceptance_ratio = proposed_posterior / current_posterior if current_posterior > 0 else 1
        
        # 决定是否接受新参数
        if np.random.rand() < acceptance_ratio:
            current_mu = proposed_mu
            current_sigma = proposed_sigma
            current_posterior = proposed_posterior
            accepts += 1
        
        # 存储当前参数
        chain[i, 0] = current_mu
        chain[i, 1] = current_sigma
    
    # 计算接受率
    acceptance_rate = accepts / n_iterations
    print(f"接受率: {acceptance_rate:.2f}")
    
    # 返回burn-in后的链
    return chain[burn_in:]


# 6. 分析和可视化结果
def analyze_results(chain, data, true_mu, true_sigma):
    """
    分析MCMC采样结果并可视化
    """
    # 提取参数估计值
    mu_chain = chain[:, 0]
    sigma_chain = chain[:, 1]
    
    # 计算后验均值和95%可信区间
    mu_estimate = np.mean(mu_chain)
    sigma_estimate = np.mean(sigma_chain)
    mu_ci = np.percentile(mu_chain, [2.5, 97.5])
    sigma_ci = np.percentile(sigma_chain, [2.5, 97.5])
    
    print(f"\n参数估计结果:")
    print(f"均值 (μ): {mu_estimate:.2f}, 95%可信区间: [{mu_ci[0]:.2f}, {mu_ci[1]:.2f}]")
    print(f"标准差 (σ): {sigma_estimate:.2f}, 95%可信区间: [{sigma_ci[0]:.2f}, {sigma_ci[1]:.2f}]")
    print(f"\n真实参数值:")
    print(f"真实均值 (μ): {true_mu:.2f}")
    print(f"真实标准差 (σ): {true_sigma:.2f}")
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 参数轨迹图
    axes[0, 0].plot(mu_chain, label='μ')
    axes[0, 0].axhline(true_mu, color='r', linestyle='--', label='真实μ')
    axes[0, 0].set_title('均值参数轨迹')
    axes[0, 0].set_xlabel('迭代次数')
    axes[0, 0].set_ylabel('μ值')
    axes[0, 0].legend()
    
    axes[0, 1].plot(sigma_chain, label='σ')
    axes[0, 1].axhline(true_sigma, color='r', linestyle='--', label='真实σ')
    axes[0, 1].set_title('标准差参数轨迹')
    axes[0, 1].set_xlabel('迭代次数')
    axes[0, 1].set_ylabel('σ值')
    axes[0, 1].legend()
    
    # 2. 参数后验分布
    sns.histplot(mu_chain, kde=True, ax=axes[1, 0])
    axes[1, 0].axvline(true_mu, color='r', linestyle='--', label='真实值')
    axes[1, 0].axvline(mu_estimate, color='g', linestyle='-', label='后验均值')
    axes[1, 0].axvline(mu_ci[0], color='g', linestyle=':', label='95%可信区间')
    axes[1, 0].axvline(mu_ci[1], color='g', linestyle=':')
    axes[1, 0].set_title('均值参数后验分布')
    axes[1, 0].set_xlabel('μ')
    axes[1, 0].legend()
    
    sns.histplot(sigma_chain, kde=True, ax=axes[1, 1])
    axes[1, 1].axvline(true_sigma, color='r', linestyle='--', label='真实值')
    axes[1, 1].axvline(sigma_estimate, color='g', linestyle='-', label='后验均值')
    axes[1, 1].axvline(sigma_ci[0], color='g', linestyle=':', label='95%可信区间')
    axes[1, 1].axvline(sigma_ci[1], color='g', linestyle=':')
    axes[1, 1].set_title('标准差参数后验分布')
    axes[1, 1].set_xlabel('σ')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('MCMC/results_posterior.png', dpi=300)
    
    # 3. 参数联合分布
    plt.figure(figsize=(10, 8))
    sns.kdeplot(x=mu_chain, y=sigma_chain, cmap="viridis", fill=True)
    plt.plot(true_mu, true_sigma, 'r*', markersize=15, label='真实参数')
    plt.plot(mu_estimate, sigma_estimate, 'go', markersize=10, label='后验均值')
    plt.title('参数联合后验分布')
    plt.xlabel('μ')
    plt.ylabel('σ')
    plt.legend()
    plt.savefig('MCMC/results_joint.png', dpi=300)
    
    # 4. 数据与后验预测比较
    plt.figure(figsize=(12, 6))
    
    # 绘制原始数据直方图
    sns.histplot(data, bins=20, kde=True, label='观测数据', alpha=0.6)
    
    # 绘制后验预测分布
    x = np.linspace(min(data) - 2, max(data) + 2, 1000)
    y_true = stats.norm.pdf(x, true_mu, true_sigma)
    y_estimated = stats.norm.pdf(x, mu_estimate, sigma_estimate)
    
    plt.plot(x, y_true, 'r--', linewidth=2, label='真实分布')
    plt.plot(x, y_estimated, 'g-', linewidth=2, label='后验估计分布')
    
    plt.title('数据与模型分布比较')
    plt.xlabel('值')
    plt.ylabel('密度')
    plt.legend()
    plt.savefig('MCMC/results_fit.png', dpi=300)
    
    # 5. 收敛诊断 - 自相关图
    plt.figure(figsize=(12, 6))
    
    lags = 50
    acf_mu = [1.] + [np.corrcoef(mu_chain[:-i], mu_chain[i:])[0, 1] for i in range(1, lags)]
    acf_sigma = [1.] + [np.corrcoef(sigma_chain[:-i], sigma_chain[i:])[0, 1] for i in range(1, lags)]
    
    plt.plot(range(lags), acf_mu, 'o-', label='μ自相关')
    plt.plot(range(lags), acf_sigma, 'o-', label='σ自相关')
    plt.axhline(0, color='black', linestyle='--')
    plt.title('参数链的自相关函数')
    plt.xlabel('滞后')
    plt.ylabel('自相关')
    plt.legend()
    plt.savefig('MCMC/results_autocorr.png', dpi=300)
    
    plt.show()


# 7. 主函数
def main():
    # 设置真实参数
    true_mu = 5.0
    true_sigma = 2.0
    
    # 生成模拟数据
    print("生成模拟数据...")
    data = generate_data(n_samples=100, true_mu=true_mu, true_sigma=true_sigma)
    
    # 运行MCMC采样
    print("\n开始MCMC采样...")
    chain = metropolis_hastings(data, n_iterations=20000, burn_in=5000)
    
    # 分析结果
    print("\n分析采样结果...")
    analyze_results(chain, data, true_mu, true_sigma)


# 执行主函数
if __name__ == "__main__":
    main()