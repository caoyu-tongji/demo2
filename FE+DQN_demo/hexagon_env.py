# 导入必要的库
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于图形绘制
from matplotlib.patches import RegularPolygon  # 用于绘制正多边形
import gym  # 强化学习环境库
from gym import spaces  # gym环境空间定义

class HexagonEnv(gym.Env):
    """
    空心六边形优化环境
    这是一个自定义的强化学习环境，继承自gym.Env基类
    目标是优化空心六边形结构，在满足应力条件的前提下最小化面积
    
    状态空间（3维）：
    - 内部六边形的边长：决定内部结构尺寸
    - 外部六边形的边长：决定外部结构尺寸
    - 当前应力状态：结构承受的应力水平
    
    动作空间（离散，4个动作）：
    - 增大/减小内部六边形边长：调整内部结构
    - 增大/减小外部六边形边长：调整外部结构
    
    奖励设计：
    - 主要奖励：负的空心六边形面积（面积越小奖励越大）
    - 约束惩罚：如果不满足受力条件，给予大的负奖励
    """
    
    def __init__(self, max_size=10.0, min_size=1.0, load=100.0, max_steps=100):
        """初始化环境参数
        Args:
            max_size (float): 最大允许的边长
            min_size (float): 最小允许的边长
            load (float): 外部施加的荷载大小
            max_steps (int): 每个episode的最大步数
        """
        super(HexagonEnv, self).__init__()
        
        # 环境参数初始化
        self.max_size = max_size  # 最大边长约束
        self.min_size = min_size  # 最小边长约束
        self.load = load  # 外部荷载大小
        self.max_steps = max_steps  # 每个episode的最大步数限制
        self.step_count = 0  # 当前步数计数器
        
        # 定义动作空间：4个离散动作
        # 0-减小内边长，1-增大内边长，2-减小外边长，3-增大外边长
        self.action_space = spaces.Discrete(4)
        
        # 定义状态空间：3维连续空间 [内边长, 外边长, 应力状态]
        self.observation_space = spaces.Box(
            low=np.array([min_size, min_size, 0]),  # 最小值
            high=np.array([max_size, max_size, 1000]),  # 最大值
            dtype=np.float32
        )
        
        # 初始化环境状态
        self.reset()
    
    def reset(self):
        """重置环境到初始状态
        Returns:
            ndarray: 初始状态观测值
        """
        # 随机初始化内外六边形边长
        self.outer_length = np.random.uniform(self.max_size/2, self.max_size)
        self.inner_length = np.random.uniform(self.min_size, self.outer_length * 0.8)
        
        # 计算初始应力状态
        self.stress = self._calculate_stress()
        
        # 重置步数计数器
        self.step_count = 0
        
        # 返回初始观测
        return self._get_observation()
    
    def _get_observation(self):
        """获取当前状态的观测值
        Returns:
            ndarray: 包含内边长、外边长和应力的状态向量
        """
        return np.array([self.inner_length, self.outer_length, self.stress])
    
    def _calculate_area(self):
        """计算空心六边形的面积
        Returns:
            float: 空心部分的面积（外六边形面积减去内六边形面积）
        """
        # 正六边形面积计算公式：(3√3/2) * a^2，其中a为边长
        outer_area = (3 * np.sqrt(3) / 2) * self.outer_length**2
        inner_area = (3 * np.sqrt(3) / 2) * self.inner_length**2
        return outer_area - inner_area
    
    def _calculate_stress(self):
        """计算当前结构的应力状态
        Returns:
            float: 计算得到的应力值，如果结构无效则返回无穷大
        """
        # 计算结构面积
        area = self._calculate_area()
        if area <= 0:
            return float('inf')  # 无效结构返回无穷大应力
        
        # 简化的应力计算模型
        # 实际工程中应该使用有限元分析等更精确的方法
        thickness = (self.outer_length - self.inner_length) / 2
        if thickness <= 0:
            return float('inf')
        
        # 应力 = 荷载 / (厚度 * 6)，6代表六边形的边数
        stress = self.load / (thickness * 6)
        return stress
    
    def _is_valid_structure(self):
        """检查当前结构是否满足约束条件
        Returns:
            bool: 结构是否有效
        """
        # 结构有效性检查：
        # 1. 几何约束：内边长必须小于外边长
        # 2. 强度约束：应力必须小于阈值
        stress_threshold = 200  # 应力阈值
        return (self.inner_length < self.outer_length) and (self.stress < stress_threshold)
    
    def step(self, action):
        """执行一步环境交互
        Args:
            action (int): 动作编号（0-3）
        Returns:
            tuple: (observation, reward, done, info) 新的观测、奖励、是否结束、额外信息
        """
        self.step_count += 1
        
        # 根据动作调整结构参数
        delta = 0.1  # 每步调整的步长
        
        # 根据动作调整内外边长
        if action == 0:  # 减小内边长
            self.inner_length = max(self.min_size, self.inner_length - delta)
        elif action == 1:  # 增大内边长
            self.inner_length = min(self.outer_length * 0.95, self.inner_length + delta)
        elif action == 2:  # 减小外边长
            self.outer_length = max(self.inner_length * 1.05, self.outer_length - delta)
        elif action == 3:  # 增大外边长
            self.outer_length = min(self.max_size, self.outer_length + delta)
        
        # 更新应力状态
        self.stress = self._calculate_stress()
        
        # 计算奖励
        area = self._calculate_area()
        is_valid = self._is_valid_structure()
        
        # 奖励设计：
        # - 有效结构：负的面积（面积越小奖励越大）+ 应力利用率奖励
        # - 无效结构：固定的惩罚值
        stress_threshold = 200  # 应力阈值
        if is_valid:
            # 计算应力利用率奖励
            # 当应力接近阈值但不超过时，给予额外奖励
            stress_utilization = self.stress / stress_threshold
            stress_reward = 50 * (1 - (1 - stress_utilization) ** 2)  # 二次函数形式的奖励
            reward = -area + stress_reward
        else:
            reward = -1000
        
        # 判断是否结束：
        # - 达到最大步数
        # - 结构无效
        done = (self.step_count >= self.max_steps) or not is_valid
        
        # 返回结果
        return self._get_observation(), reward, done, {"area": area, "stress": self.stress, "valid": is_valid}
    
    def render(self, mode='human'):
        """渲染当前环境状态
        Args:
            mode (str): 渲染模式，'human'用于显示，'rgb_array'用于返回图像数据
        Returns:
            ndarray or None: 如果mode为'rgb_array'则返回图像数据
        """
        # 创建新的图形
        plt.close('all')  # 清除之前的图形
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        
        # 绘制外部六边形
        outer_hex = RegularPolygon((0, 0), numVertices=6, radius=self.outer_length,
                                 orientation=np.pi/6, alpha=0.5, edgecolor='blue', facecolor='skyblue')
        ax.add_patch(outer_hex)
        
        # 绘制内部六边形
        inner_hex = RegularPolygon((0, 0), numVertices=6, radius=self.inner_length,
                                 orientation=np.pi/6, alpha=0.5, edgecolor='red', facecolor='white')
        ax.add_patch(inner_hex)
        
        # 设置图形显示范围和比例
        limit = self.max_size * 1.2
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        
        # 添加结构信息文本
        area = self._calculate_area()
        stress = self.stress
        is_valid = self._is_valid_structure()
        status = "Valid" if is_valid else "Invalid"
        
        info_text = f"Inner Length: {self.inner_length:.2f}\nOuter Length: {self.outer_length:.2f}\nArea: {area:.2f}\nStress: {stress:.2f}\nStatus: {status}"
        ax.text(limit * 0.5, limit * 0.8, info_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        
        # 设置标题和网格
        plt.title("Hollow Hexagon Optimization")
        plt.grid(True)
        plt.tight_layout()
        
        # 根据渲染模式返回结果
        if mode == 'human':
            plt.show()
        elif mode == 'rgb_array':
            # 将图形转换为RGB数组
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return img
        
        plt.close(fig)