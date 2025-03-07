import numpy as np  # 导入NumPy库，用于数值计算
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
import torch  # 导入PyTorch库，用于深度学习
import os  # 导入os库，用于文件和目录操作
import time  # 导入time库，用于时间相关操作
from tqdm import tqdm  # 导入tqdm库，用于显示进度条
from hexagon_env import HexagonEnv  # 导入自定义的六边形环境
from dqn_model import DQNAgent  # 导入自定义的DQN智能体

# 设置matplotlib后端为Agg，避免显示图形界面
import matplotlib
matplotlib.use('Agg')

# 创建结果目录
results_dir = 'results'  # 定义结果保存目录
os.makedirs(results_dir, exist_ok=True)  # 创建目录，如果已存在则不报错

# 创建图片保存目录
pictures_dir = os.path.join(results_dir, 'pictures')  # 定义图片保存子目录
os.makedirs(pictures_dir, exist_ok=True)  # 创建图片目录

# 训练参数
EPISODES = 1000  # 总训练回合数
BATCH_SIZE = 64  # 经验回放的批量大小
UPDATE_TARGET_EVERY = 5  # 每5个episode更新一次目标网络
SAVE_MODEL_EVERY = 100    # 每100个episode保存一次模型
RENDER_EVERY = 50         # 每50个episode渲染一次

# 初始化环境和智能体
env = HexagonEnv(max_size=10.0, min_size=1.0, load=100.0, max_steps=100)  # 创建六边形环境实例
state_size = env.observation_space.shape[0]  # 获取状态空间维度
action_size = env.action_space.n  # 获取动作空间维度
agent = DQNAgent(state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995)  # 创建DQN智能体

# 训练记录
episode_rewards = []  # 用于记录每个episode的总奖励
best_reward = -float('inf')  # 初始化最佳奖励为负无穷
best_area = float('inf')  # 初始化最佳面积为正无穷（我们要最小化面积）
best_state = None  # 用于记录最佳状态参数

# 训练循环
pbar = tqdm(range(1, EPISODES + 1), desc='Training')  # 创建进度条
for episode in pbar:
    state = env.reset()  # 重置环境，获取初始状态
    total_reward = 0  # 初始化当前episode的总奖励
    done = False  # 初始化结束标志
    step = 0  # 初始化步数计数器
    
    # 单个episode循环
    while not done:
        # 智能体选择动作
        action = agent.act(state)  # 根据当前状态选择动作
        
        # 执行动作
        next_state, reward, done, info = env.step(action)  # 在环境中执行动作，获取下一状态、奖励等信息
        
        # 记忆经验
        agent.remember(state, action, reward, next_state, done)  # 将经验存入回放缓冲区
        
        # 更新状态和奖励
        state = next_state  # 更新当前状态
        total_reward += reward  # 累加奖励
        step += 1  # 步数加1
        
        # 如果有足够的经验，进行批量学习
        if len(agent.memory) > BATCH_SIZE:
            agent.replay(BATCH_SIZE)  # 从经验回放缓冲区中随机采样并学习
    
    # 记录本次episode的总奖励
    episode_rewards.append(total_reward)  # 将当前episode的总奖励添加到记录列表
    
    # 更新目标网络
    if episode % UPDATE_TARGET_EVERY == 0:
        agent.update_target_model()  # 定期更新目标网络，提高训练稳定性
    
    # 保存模型
    if episode % SAVE_MODEL_EVERY == 0:
        agent.save(os.path.join(results_dir, f'dqn_model_episode_{episode}.pth'))  # 定期保存模型
    
    # 记录最佳结果
    if info['valid'] and -total_reward < best_area:  # 注意：奖励是负的面积
        best_reward = total_reward  # 更新最佳奖励
        best_area = -total_reward  # 转换回正的面积值
        best_state = (env.inner_length, env.outer_length, env.stress)  # 记录最佳状态参数
        
        # 保存最佳模型
        agent.save(os.path.join(results_dir, 'best_dqn_model.pth'))  # 保存当前最佳模型
        
        # 渲染并保存最佳结果图
        fig, ax = plt.subplots(figsize=(10, 8))  # 创建图形
        img = env.render(mode='rgb_array')  # 使用rgb_array模式并获取图像数据
        plt.imshow(img)  # 显示图像
        plt.axis('off')  # 关闭坐标轴
        plt.tight_layout()  # 调整布局
        plt.savefig(os.path.join(results_dir, 'best_hexagon.png'), bbox_inches='tight', dpi=100)  # 保存图像
        plt.close(fig)  # 关闭图形
    
    # 定期渲染和更新进度条
    if episode % RENDER_EVERY == 0:
        # 更新进度条描述
        pbar.set_postfix({  # 在进度条上显示当前训练状态
            'Reward': f"{total_reward:.2f}", 
            'Epsilon': f"{agent.epsilon:.4f}",
            'Area': f"{info['area']:.2f}", 
            'Valid': info['valid'],
            'Best': f"{best_area:.2f}"
        })
        
        # 渲染当前状态并保存到pictures文件夹
        fig, ax = plt.subplots(figsize=(10, 8))  # 创建图形
        img = env.render(mode='rgb_array')  # 使用rgb_array模式并获取图像数据
        plt.imshow(img)  # 显示图像
        plt.axis('off')  # 关闭坐标轴
        plt.tight_layout()  # 调整布局
        plt.savefig(os.path.join(pictures_dir, f'hexagon_episode_{episode}.png'), bbox_inches='tight', dpi=100)  # 保存图像
        plt.close(fig)  # 关闭图形

# 训练完成后，绘制奖励曲线
plt.figure(figsize=(12, 6))  # 创建图形
plt.plot(episode_rewards)  # 绘制奖励曲线
plt.title('DQN Training Rewards')  # 设置标题
plt.xlabel('Episode')  # 设置x轴标签
plt.ylabel('Total Reward')  # 设置y轴标签
plt.grid(True)  # 显示网格
plt.savefig(os.path.join(results_dir, 'training_rewards.png'))  # 保存奖励曲线图

# 输出最佳结果
print("\n训练完成！")  # 打印训练完成信息
print(f"最佳结构参数：")  # 打印最佳结构参数标题
print(f"内边长: {best_state[0]:.4f}")  # 打印最佳内边长
print(f"外边长: {best_state[1]:.4f}")  # 打印最佳外边长
print(f"应力: {best_state[2]:.4f}")  # 打印最佳应力
print(f"面积: {best_area:.4f}")  # 打印最佳面积

# 加载最佳模型并渲染最终结果
agent.load(os.path.join(results_dir, 'best_dqn_model.pth'))  # 加载最佳模型
env.inner_length = best_state[0]  # 设置环境内边长为最佳值
env.outer_length = best_state[1]  # 设置环境外边长为最佳值
env.stress = best_state[2]  # 设置环境应力为最佳值

# 创建图形并渲染
fig, ax = plt.subplots(figsize=(10, 8))  # 创建图形
img = env.render(mode='rgb_array')  # 使用rgb_array模式并获取图像数据
plt.imshow(img)  # 显示图像
plt.axis('off')  # 关闭坐标轴
plt.tight_layout()  # 调整布局
plt.savefig(os.path.join(results_dir, 'final_best_hexagon.png'), bbox_inches='tight', dpi=100)  # 保存最终最佳图像
plt.close(fig)  # 关闭图形而不是显示