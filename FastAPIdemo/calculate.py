import numpy as np

def generate_sample_curves():
    """
    生成两条带有随机噪声和随机初始相位的示例曲线
    
    返回:
    tuple: 包含两条曲线数据的元组
    """
    # 生成基础数据点
    x = np.linspace(0, 2*np.pi, 100)
    
    # 生成随机噪声和随机初始相位
    noise1 = np.random.normal(0, 0.1, 100)  # 均值为0，标准差为0.1的高斯噪声
    noise2 = np.random.normal(0, 0.1, 100)
    phase1 = np.random.uniform(0, 2*np.pi)  # 随机初始相位 [0, 2π]
    phase2 = np.random.uniform(0, 2*np.pi)
    
    # 生成第一条曲线 - 带噪声和随机相位的正弦波
    curve1 = np.sin(x + phase1) + noise1
    
    # 生成第二条曲线 - 带噪声和随机相位的余弦波
    curve2 = np.cos(x + phase2) + noise2
    
    return curve1.tolist(), curve2.tolist()
def calculate_cosine_similarity(curve1, curve2):
    """
    计算两条曲线的余弦相似度
    
    参数:
    curve1: 第一条曲线的数据点列表
    curve2: 第二条曲线的数据点列表
    
    返回:
    float: 两条曲线的余弦相似度，范围在[-1, 1]之间
    """
    # 确保输入数据是numpy数组
    curve1 = np.array(curve1)
    curve2 = np.array(curve2)
    
    # 检查输入数据的维度是否匹配
    if len(curve1) != len(curve2):
        raise ValueError("两条曲线的数据点数量必须相同")
    
    # 计算余弦相似度
    # cos_sim = (A·B)/(||A||·||B||)
    numerator = np.dot(curve1, curve2)
    denominator = np.linalg.norm(curve1) * np.linalg.norm(curve2)
    
    # 避免除以零
    if denominator == 0:
        return 0.0
    
    cosine_similarity = numerator / denominator
    return float(cosine_similarity)