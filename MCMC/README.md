# MCMC贝叶斯参数估计

本项目实现了使用马尔可夫链蒙特卡洛（Markov Chain Monte Carlo, MCMC）方法进行贝叶斯参数估计的完整流程。代码通过Metropolis-Hastings算法对正态分布的均值和标准差参数进行采样，并提供了详细的可视化分析。

## 目录

1. [MCMC方法简介](#1-mcmc方法简介)
2. [代码实现步骤详解](#2-代码实现步骤详解)
   - [数据生成](#21-数据生成)
   - [先验分布定义](#22-先验分布定义)
   - [似然函数计算](#23-似然函数计算)
   - [后验分布定义](#24-后验分布定义)
   - [Metropolis-Hastings算法实现](#25-metropolis-hastings算法实现)
   - [结果分析与可视化](#26-结果分析与可视化)
3. [运行结果说明](#3-运行结果说明)
4. [参考资料](#4-参考资料)

## 1. MCMC方法简介

MCMC是一类用于从复杂概率分布中抽取样本的算法。在贝叶斯统计中，我们通常需要计算后验分布，但对于复杂模型，后验分布往往没有解析解或难以直接采样。MCMC方法通过构建一个马尔可夫链，使其平稳分布等于目标后验分布，从而间接地从后验分布中获取样本。

Metropolis-Hastings算法是MCMC方法的一种经典实现，其基本步骤如下：

1. 从当前状态生成一个候选状态（建议状态）
2. 计算接受概率
3. 以一定概率接受或拒绝候选状态
4. 重复上述步骤多次，得到一个状态序列（链）

经过足够长的迭代后，链中的状态将近似服从目标分布。

## 2. 代码实现步骤详解

### 2.1 数据生成

```python
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
```

这个函数使用NumPy的`random.normal`方法生成服从正态分布的随机数据。在本例中，我们生成了均值为5.0，标准差为2.0的100个样本点。这些数据将作为我们的"观测数据"，用于后续的参数估计。

### 2.2 先验分布定义

在贝叶斯统计中，先验分布表示在观测数据之前，我们对参数的信念。代码定义了两个先验分布：

```python
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
```

- `prior_mu`: 均值参数μ的先验分布被设定为均值为0，标准差为10的正态分布。这是一个相对宽松的先验，表示我们对μ的真实值没有很强的先验信念。
- `prior_sigma`: 标准差参数σ的先验分布被设定为尺度参数为10的半正态分布。由于标准差必须为正，所以使用半正态分布而非普通正态分布。

### 2.3 似然函数计算

似然函数表示在给定参数值的情况下，观测到当前数据的概率：

```python
def likelihood(data, mu, sigma):
    """
    计算给定参数下观测数据的似然
    """
    if sigma <= 0:
        return 0
    return np.prod(stats.norm.pdf(data, loc=mu, scale=sigma))
```

这个函数计算了在给定均值μ和标准差σ的情况下，观测到数据集`data`的概率。由于我们假设数据点之间相互独立，总的似然等于各个数据点似然的乘积（使用`np.prod`）。

### 2.4 后验分布定义

根据贝叶斯定理，后验分布正比于似然函数与先验分布的乘积：

```python
def posterior_unnormalized(data, mu, sigma):
    """
    计算未归一化的后验概率
    后验 ∝ 似然 × 先验
    """
    return likelihood(data, mu, sigma) * prior_mu(mu) * prior_sigma(sigma)
```

这里计算的是未归一化的后验概率，因为在MCMC方法中，我们只需要知道后验概率的比值，而不需要知道其绝对值。

### 2.5 Metropolis-Hastings算法实现

Metropolis-Hastings算法是本代码的核心部分：

```python
def metropolis_hastings(data, n_iterations=10000, burn_in=1000):
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
```

算法步骤解析：

1. **初始化**：
   - 使用数据的样本均值和标准差作为参数的初始值
   - 创建一个数组来存储参数链（每次迭代的参数值）
   - 计算初始参数下的后验概率

2. **迭代过程**：
   - 从建议分布中抽取新的参数值（这里使用以当前值为中心的正态分布）
   - 计算新参数下的后验概率
   - 计算接受概率（新后验概率/当前后验概率）
   - 以接受概率决定是否接受新参数
   - 无论是否接受新参数，都将当前参数存入链中

3. **返回结果**：
   - 丢弃预热期（burn-in）的样本，返回剩余的参数链

预热期是为了让马尔可夫链有足够的时间收敛到目标分布，通常会丢弃前面一部分样本。

### 2.6 结果分析与可视化

```python
def analyze_results(chain, data, true_mu, true_sigma):
```

这个函数对MCMC采样得到的参数链进行分析和可视化，主要包括：

1. **参数估计**：
   - 计算参数的后验均值
   - 计算参数的95%可信区间

2. **可视化**：
   - 参数轨迹图：展示参数在迭代过程中的变化
   - 参数后验分布：展示参数的边缘后验分布
   - 参数联合分布：展示两个参数的联合后验分布
   - 数据与模型比较：比较原始数据与估计模型的拟合情况
   - 自相关图：用于诊断MCMC采样的收敛性

## 3. 运行结果说明

代码运行后会生成以下几个图像文件：

- `results_posterior.png`：参数的轨迹图和边缘后验分布
- `results_joint.png`：参数的联合后验分布
- `results_fit.png`：数据与模型分布的比较
- `results_autocorr.png`：参数链的自相关函数，用于诊断收敛性

同时，控制台会输出参数的估计结果，包括后验均值和95%可信区间，以及与真实参数的比较。

## 4. 参考资料

- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). Bayesian Data Analysis (3rd ed.). CRC Press.
- Robert, C. P., & Casella, G. (2004). Monte Carlo Statistical Methods (2nd ed.). Springer.
- Brooks, S., Gelman, A., Jones, G. L., & Meng, X.-L. (Eds.). (2011). Handbook of Markov Chain Monte Carlo. CRC Press.