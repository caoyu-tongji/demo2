import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# 定义DQN网络结构
class DQN(nn.Module):
    """深度Q网络模型
    
    这是一个简单的全连接神经网络，用于近似Q函数
    输入为状态，输出为每个动作的Q值
    """
    def __init__(self, state_size, action_size):
        """初始化网络结构
        Args:
            state_size (int): 状态空间维度
            action_size (int): 动作空间维度
        """
        super(DQN, self).__init__()
        # 定义三层全连接网络
        self.fc1 = nn.Linear(state_size, 64)  # 第一层：输入层 -> 64个神经元
        self.fc2 = nn.Linear(64, 64)          # 第二层：64 -> 64个神经元
        self.fc3 = nn.Linear(64, action_size)  # 第三层：64 -> 动作数量个输出
        
    def forward(self, x):
        """前向传播
        Args:
            x (Tensor): 输入状态
        Returns:
            Tensor: 每个动作的Q值
        """
        x = F.relu(self.fc1(x))  # 第一层使用ReLU激活函数
        x = F.relu(self.fc2(x))  # 第二层使用ReLU激活函数
        return self.fc3(x)       # 输出层不使用激活函数

# 定义经验回放缓冲区
class ReplayBuffer:
    """经验回放缓冲区
    
    存储和采样智能体的经验(状态、动作、奖励、下一状态、是否结束)
    用于打破样本间的相关性，提高训练稳定性
    """
    def __init__(self, capacity=10000):
        """初始化经验回放缓冲区
        Args:
            capacity (int): 缓冲区容量，默认10000
        """
        # 使用双端队列，当超过容量时自动移除最早的经验
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """添加经验到缓冲区
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """从缓冲区随机采样一批经验
        Args:
            batch_size (int): 批量大小
        Returns:
            tuple: (states, actions, rewards, next_states, dones)批量经验
        """
        # 随机采样batch_size个经验
        batch = random.sample(self.buffer, batch_size)
        # 解包采样的经验
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones
    
    def __len__(self):
        """返回缓冲区当前大小
        Returns:
            int: 缓冲区中的经验数量
        """
        return len(self.buffer)

# 定义DQN智能体
class DQNAgent:
    """DQN强化学习智能体
    
    实现了深度Q学习算法，包括双网络架构、经验回放和ε-贪婪策略
    """
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """初始化DQN智能体
        Args:
            state_size (int): 状态空间维度
            action_size (int): 动作空间维度
            learning_rate (float): 学习率
            gamma (float): 折扣因子，决定未来奖励的重要性
            epsilon (float): 初始探索率
            epsilon_min (float): 最小探索率
            epsilon_decay (float): 探索率衰减系数
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer()  # 经验回放缓冲区
        self.gamma = gamma    # 折扣因子，决定未来奖励的重要性
        self.epsilon = epsilon  # 探索率，控制探索与利用的平衡
        self.epsilon_min = epsilon_min  # 最小探索率
        self.epsilon_decay = epsilon_decay  # 探索率衰减系数
        self.learning_rate = learning_rate  # 学习率
        
        # 创建主网络和目标网络
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用GPU或CPU
        self.model = DQN(state_size, action_size).to(self.device)  # 主网络，用于选择动作和更新
        self.target_model = DQN(state_size, action_size).to(self.device)  # 目标网络，用于计算目标Q值
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)  # Adam优化器
        
        # 初始化目标网络权重，使其与主网络相同
        self.update_target_model()
    
    def update_target_model(self):
        """更新目标网络
        将主网络的权复制到目标网络
        """
        self.target_model.load_state_dict(self.model.state_dict())
    
    def act(self, state):
        """根据当前状态选择动作
        使用ε-贪婪策略平衡探索与利用
        
        Args:
            state: 当前状态
        Returns:
            int: 选择的动作
        """
        # 探索：以epsilon的概率随机选择动作
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # 利用：选择Q值最大的动作
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # 转换为张量并添加批次维度
        with torch.no_grad():  # 不计算梯度
            action_values = self.model(state)  # 获取所有动作的Q值
        return torch.argmax(action_values).item()  # 返回Q值最大的动作
    
    def remember(self, state, action, reward, next_state, done):
        """将经验存储到回放缓冲区
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.memory.add(state, action, reward, next_state, done)
    
    def replay(self, batch_size):
        """从经验回放缓冲区中学习
        Args:
            batch_size (int): 批量大小
        Returns:
            float: 损失值
        """
        # 如果缓冲区中的经验不足，则不进行学习
        if len(self.memory) < batch_size:
            return
        
        # 从经验回放缓冲区中采样一批经验
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # 转换为PyTorch张量
        states = torch.FloatTensor(states).to(self.device)  # 状态
        actions = torch.LongTensor(actions).to(self.device)  # 动作
        rewards = torch.FloatTensor(rewards).to(self.device)  # 奖励
        next_states = torch.FloatTensor(next_states).to(self.device)  # 下一状态
        dones = torch.FloatTensor(dones).to(self.device)  # 是否结束
        
        # 计算当前Q值：Q(s,a)
        # gather操作选择实际执行的动作对应的Q值
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值：r + γ * max_a' Q'(s',a')
        # 使用目标网络计算下一状态的最大Q值
        next_q_values = self.target_model(next_states).max(1)[0]
        # 如果episode结束，则只有即时奖励，没有未来奖励
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算均方误差损失并更新网络
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()  # 清除梯度
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新参数
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()  # 返回损失值
    
    def save(self, filepath):
        """保存模型权重
        Args:
            filepath (str): 保存路径
        """
        torch.save(self.model.state_dict(), filepath)
    
    def load(self, filepath):
        """加载模型权重
        Args:
            filepath (str): 加载路径
        """
        self.model.load_state_dict(torch.load(filepath))
        self.update_target_model()  # 同步更新目标网络