#导入OpenAI Gym库，用于创建和管理强化学习环境
import gym
#导入随机数生成库，用于策略中的随即决策
import random
#导入NumPy库，用于高效数值计算和数组操作
import numpy as np
 # 导入强化学习智能体
from RL_QG_agent import RL_QG_agent 

# 创建初始环境并重置（8*8棋盘）
env = gym.make('Reversi8x8-v0')#使用openAI Gym接口创建黑白棋环境
env.reset()#重置环境到初始状态

# 初始化智能体并加载预训练模型
agent = RL_QG_agent()#创建智能体实例
agent.load_model()#加载预训练模型参数

# 设置最大训练轮数/局数（epochs）
max_epochs = 100  # 总共进行100局训练，每局包含完整的游戏过程

# 以下为胜负统计变量
# black_wins = 0    # 统计黑方获胜次数（适用于围棋等双人博弈游戏）
# white_wins = 0    # 统计白方获胜次数  
# draws = 0         # 统计平局次数

# 主训练循环（控制训练的总轮数）
# max_epochs：决定智能体与环境交互的总次数
for i_episode in range(max_epochs):
    # 初始化棋局，返回初始 observation（3x8x8 的状态表示）
    #3个通道分别表示：黑棋位置、白棋位置、当前玩家
    observation = env.reset()
    
    # 每局最多进行 100 步操作（黑白双方交替下棋）
    for t in range(100):
        action = [1, 2]  # 初始化 action，占位，稍后会被赋真实值
        # action[0]: 表示棋盘上的位置（0~63），或特殊值表示“跳过”
        # action[1]: 表示棋子的颜色，0 表示黑棋，1 表示白棋

        ################### 黑棋（随机策略） ###################
        env.render()  # 打印当前棋盘状态
        enables = env.possible_actions  # 获取当前黑棋可落子的位置列表

        if len(enables) == 0:# 判断是否有合法落子的位置
            # 没有可落子位置，则选择“跳过”操作，编码为 board_size^2 + 1
            action_ = env.board_size**2 + 1
        else:
            # 随机选择一个合法位置
            action_ = random.choice(enables)
        action[0] = action_# 设置落子位置
        action[1] = 0  # 设置为黑棋
        # 黑棋落子并更新环境状态，返回新状态、奖励、是否结束等信息
        # 执行动作并获取新状态
        # observation: 更新后的游戏状态(3x8x8张量)
        # reward: 执行动作后的即时奖励(黑白棋中通常为0，结束时才有奖励)
        # done: 游戏是否结束
        # info: 包含额外信息的字典(如获胜方)
        observation, reward, done, info = env.step(action)

        ################### 白棋（智能体策略） ###################
        # 可视化当前棋盘状态，便于调试或展示
        env.render()  # 打印当前棋盘状态
        enables = env.possible_actions  # 获取当前白棋可落子的位置列表
        if not enables:
            # 无法落子，执行“跳过”操作
            action_ = env.board_size ** 2 + 1
        else:  
            # 使用训练好的智能体模型选择最佳落子位置
            # observation: 当前环境观测（如棋盘状态）
            # enables: 合法动作列表
            # 返回值: 选中的动作编号（对应enables中的索引）
            action_ = agent.place(observation, enables)
            # 构建动作向量 [动作编号, 玩家编号]
        action[0] = action_ # 落子位置或特殊动作编号
        action[1] = 1  # 设置为白棋
        # 白棋落子并更新环境状态
        # observation: 更新后的环境观测
        # reward: 执行动作后获得的奖励（如胜利+1，失败-1）
        # done: 游戏是否结束
        # info: 额外信息（如获胜方、结束原因）
        observation, reward, done, info = env.step(action)

        # 如果对局结束
        # 检查游戏是否结束
        if done:
            # 打印游戏结束信息（总步数 = t+1，因为索引从0开始）
            print(f"第 {i_episode+1} 局游戏在 {t+1} 步后结束")
            
            # 计算黑棋得分（棋盘上黑棋的数量）
            black_score = len(np.where(env.state[0, :, :] == 1)[0])
            total_tiles = env.board_size ** 2  # 棋盘总格子数（8x8=64）
            
            # 判断游戏结果
            if black_score > total_tiles / 2:  # 黑棋数量超过一半
                print("黑棋获胜！")
            elif black_score < total_tiles / 2:  # 白棋数量超过一半
                print("白棋获胜！")
            else:  # 双方棋子数量相等
                print("平局！")
            
            # 打印详细得分
            white_score = total_tiles - black_score
            print(f"比分: 黑棋 {black_score} - 白棋 {white_score}")
            
            break  # 结束当前游戏，开始下一局
         # 关闭环境（资源清理）
         env.close()
         print("训练完成！共进行了 {max_epochs} 局游戏")
