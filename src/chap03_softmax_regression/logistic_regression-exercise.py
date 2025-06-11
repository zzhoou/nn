#!/usr/bin/env python
# coding: utf-8
# # Logistic Regression Example
# ### 生成数据集，看明白即可无需填写代码
# #### '<font color="blue">+</font>' 从高斯分布采样 (X, Y) ~ N(3, 6, 1, 1, 0).<br>
# #### '<font color="green">o</font>' 从高斯分布采样 (X, Y) ~ N(6, 3, 1, 1, 0)<br>

# 导入 TensorFlow 深度学习框架
import tensorflow as tf
# 导入 matplotlib 的 pyplot 模块，用于数据可视化
import matplotlib.pyplot as plt

# 从 matplotlib 导入 animation 和 rc 模块
# animation：用于创建动态动画
# rc：运行时配置(runtime configuration)，用于设置图形默认参数
from matplotlib import animation, rc
# 导入 IPython 的 HTML 显示功能，用于在 Notebook 中嵌入动画
from IPython.display import HTML
# 导入 matplotlib 的 colormap 模块，用于颜色映射
import matplotlib.cm as cm
# 导入 NumPy 数值计算库
import numpy as np

# 设置随机种子（确保结果可复现）
# NumPy的随机种子
np.random.seed(42)
# TensorFlow的随机种子
tf.random.set_seed(42)

# 确保在 Jupyter Notebook 中内联显示图形
get_ipython().run_line_magic('matplotlib', 'inline')

# 设置数据点数量
dot_num = 100
# 从均值为3，标准差为1的高斯分布中采样x坐标，用于正样本
x_p = np.random.normal(3., 1, dot_num)
# 从均值为6，标准差为1的高斯分布中采样y坐标，用于正样本
y_p = np.random.normal(6., 1, dot_num)
# 正样本的标签设为1
y = np.ones(dot_num)
# 将正样本的x、y坐标和标签组合成一个数组，形状为 (dot_num, 3)
C1 = np.array([x_p, y_p, y]).T
# random函数为伪随机数生成，并非真随机

# 从均值为6，标准差为1的高斯分布中采样x坐标，用于负样本
x_n = np.random.normal(6., 1, dot_num)
# 从均值为3，标准差为1的高斯分布中采样y坐标，用于负样本
y_n = np.random.normal(3., 1, dot_num)
# 负样本的标签设为0
y = np.zeros(dot_num)
# 将负样本的x、y坐标和标签组合成一个数组，形状为 (dot_num, 3)
C2 = np.array([x_n, y_n, y]).T

# 绘制正样本，用蓝色加号表示
plt.scatter(C1[:, 0], C1[:, 1], c = 'b', marker = '+')
# 绘制负样本，用绿色圆圈表示
plt.scatter(C2[:, 0], C2[:, 1], c = 'g', marker = 'o')

# 将正样本和负样本连接成一个数据集
data_set = np.concatenate((C1, C2), axis = 0)
# 随机打乱数据集的顺序
np.random.shuffle(data_set)

# ## 建立模型
# 建立模型类，定义loss函数，定义一步梯度下降过程函数
# 填空一：实现sigmoid的交叉熵损失函数(不使用tf内置的loss 函数)
# 防止对数运算出现数值不稳定问题，添加一个极小值
epsilon = 1e-12


class LogisticRegression():
    def __init__(self):
        # 正则化常见目的：防止过拟合、提高稳定性、降低方差、提高泛化能力
        # L2正则化，防止过拟合，正则化系数为0.01
        # L2正则化通过对权重施加平方惩罚项来防止权重过大
        l2_reg = tf.keras.regularizers.l2(0.01)
        # 初始化权重变量W，形状为[2, 1]，初始值在-0.1到0.1之间均匀分布，并应用L2正则化
        self.W = tf.Variable(
            initial_value = tf.random.uniform(
                shape = [2, 1], minval = -0.1, maxval = 0.1
            ),
            regularizer = l2_reg
        )
        # 初始化偏置变量b，形状为[1]，初始值为0
        self.b = tf.Variable(
            shape=[1],
            dtype=tf.float32,
            initial_value=tf.zeros(shape=[1])
        )
        # 定义模型的可训练变量，即权重W和偏置b
        self.trainable_variables = [self.W, self.b]

    @tf.function
    def __call__(self, inp):
        """
           计算神经网络模型的前向传播过程，包括输入数据与权重的矩阵乘法、加偏置、然后应用sigmoid激活函数。
    
           参数:
               inp (tf.Tensor): 输入数据，形状通常为(N, D)，其中N是样本数，D是特征维度。
    
           返回:
                tf.Tensor: 预测的概率值，形状为(N, 1)，值在[0, 1]之间。
        """
        # 计算输入数据与权重的矩阵乘法，再加上偏置，得到logits，形状为(N, 1)
        logits = tf.matmul(inp, self.W) + self.b
        # 对logits应用sigmoid函数，得到预测概率
        pred = tf.nn.sigmoid(logits)
        return pred


# 使用tf.function将该方法编译为静态图，提高执行效率
# 以下代码计算了二分类问题里的交叉熵损失以及准确率，并且进行了一系列的数据处理和数值稳定性方面的优化
@tf.function
def compute_loss(pred, label):
    """
        计算二分类交叉熵损失函数（手动实现，不使用tf内置loss）
        
        参数:
            pred: 输入特征，形状为(N, 2)的Tensor
            label: 真实标签，形状为(N, 1)的Tensor，取值为0或1
            
        返回:
            loss: 平均交叉熵损失 + L2正则化项
        """
    if not isinstance(label, tf.Tensor):
        # 如果标签不是Tensor类型，将其转换为Tensor类型，数据类型为float32
        label = tf.constant(label, dtype = tf.float32)
    # 压缩预测结果的维度，从形状(N, 1)变为(N,)
    pred = tf.squeeze(pred, axis = 1)
    '''============================='''
    # 输入label shape(N,), pred shape(N,)
    # 输出 losses shape(N,) 每一个样本一个loss
    # todo 填空一，实现sigmoid的交叉熵损失函数(不使用tf内置的loss 函数)
    # 计算每个样本的sigmoid交叉熵损失，防止log(0)的情况，加上一个小的epsilon
    losses = -label * tf.math.log(pred + epsilon) - (1 - label) * tf.math.log(1 - pred + epsilon)
    # 交叉熵损失公式（二分类问题）：
    # L = -Σ [y * log(p) + (1 - y) * log(1 - p)]
    # 其中 y 是真实标签，p 是预测概率
    '''============================='''
    # 计算所有样本损失的平均值
    loss = tf.reduce_mean(losses)
    
    # 将预测概率大于0.5的设置为1，小于等于0.5的设置为0，得到预测标签
    pred = tf.where(pred > 0.5, tf.ones_like(pred), tf.zeros_like(pred))
    # 计算预测标签与真实标签相等的比例，即准确率
    accuracy = tf.reduce_mean(tf.cast(tf.equal(label, pred), dtype=tf.float32))
    return loss, accuracy# 返回计算得到的损失值和准确率


@tf.function
def train_one_step(model, optimizer, x, y):
    # 使用GradientTape记录计算图，以便计算梯度
    with tf.GradientTape() as tape:
        # 使用模型对输入数据x进行预测
        pred = model(x)
        # 计算预测结果的损失和准确率
        loss, accuracy = compute_loss(pred, y)
        
    # 计算损失对可训练变量的梯度
    grads = tape.gradient(loss, model.trainable_variables)
    # 使用优化器更新模型的可训练变量
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, accuracy, model.W, model.b


if __name__ == '__main__':
   # 实例化逻辑回归模型
   # LogisticRegression 是一个用于二分类问题的线性模型
   model = LogisticRegression()

   # 使用自适应优化器 Adam ，学习率为0.01
   # Adam 是一种自适应学习率的优化算法，结合了 Momentum 和 RMSProp 的优点
   # learning_rate=0.01 设置了初始学习率为 0.01
   opt = tf.keras.optimizers.Adam(learning_rate=0.01)  # 或Nadam/RMSprop

   # 从数据集中解包出x1, x2坐标和标签y
   # data_set 是一个包含多个样本的列表，每个样本格式为 (x1, x2, y)
   # 使用 zip(*data_set) 对数据进行转置，然后转换为列表
   # 这样可以将所有x1、x2和y分别分组
   x1, x2, y = list(zip(*data_set))

   # 将x1和x2组合成输入数据 x
   # 使用 zip(x1, x2) 将每个样本的x1和x2特征重新组合成特征对
   # 最终 x 的形式是 [(x1_1, x2_1), (x1_2, x2_2), ...]
   x = list(zip(x1, x2))

   # 用于存储训练过程中每一步的模型参数和损失值，便于动画可视化
   # animation_frames 列表将记录训练过程中每个步骤或epoch的:
   #   - 模型参数(如权重和偏置)
   #   - 当前损失值
   # 这些信息可以用于后续创建训练过程的动画演示
   animation_frames = []

    for i in range(200):
        # 执行一次训练步骤，返回损失、准确率、当前的权重 W 和偏置 b
        loss, accuracy, W_opt, b_opt = train_one_step(model, opt, x, y)
        # 将当前的权重W的第一个元素、第二个元素、偏置b和损失值添加到animation_frames中
        animation_frames.append(
            (W_opt.numpy()[0, 0], W_opt.numpy()[1, 0], b_opt.numpy(), loss.numpy())
        )
        if i % 20 == 0:
            print(f'loss: {loss.numpy():.4}\t accuracy: {accuracy.numpy():.4}')


    f, ax = plt.subplots(figsize=(6, 4))  # 创建一个图形和坐标轴
    f.suptitle('Logistic Regression Example', fontsize=15)  # 设置图形的标题
    plt.ylabel('Y') 
    plt.xlabel('X')  
    ax.set_xlim(0, 10)  
    ax.set_ylim(0, 10) 
    
    line_d, = ax.plot([], [], label = 'fit_line')
    C1_dots, = ax.plot([], [], '+', c = 'b', label = 'actual_dots')
    C2_dots, = ax.plot([], [], 'o', c = 'g', label = 'actual_dots')

    # 创建用于显示动态文本的文本对象（位于左上角）
    frame_text = ax.text(
        0.02, 0.95, '',
        horizontalalignment='left',
        verticalalignment='top', 
        transform=ax.transAxes
    )

    def init():
        """
        初始化动画所需的图形元素
        该函数将所有动态绘图对象的数据清空，为动画初始化做准备
        通常用于 Matplotlib 的 FuncAnimation 初始化函数
        """
        # 清空线条对象的数据（x, y 坐标）
        line_d.set_data([], [])
        # 清空类别1的散点数据（C1）
        C1_dots.set_data([], [])
        # 清空类别2的散点数据（C2）
        C2_dots.set_data([], [])
        # 返回所有需要动画更新的图形对象（打包为元组）
        # Matplotlib 动画要求返回值为 Artist 对象的集合
        return (line_d,) + (C1_dots,) + (C2_dots,)

    def animate(i):
        """
        动画的每一帧更新函数
        参数:
        i: 当前帧的索引
        返回:
        更新后的图形对象
        """
        # 具体实现 
        xx = np.arange(10, step=0.1)# 生成x轴数据点，范围0-9.9，步长0.1
        a = animation_frames[i][0]  # 从帧数据中提取当前帧的参数，假设animation_frames是一个列表，每个元素包含[a, b, c, loss]四个值
        b = animation_frames[i][1]  # 从帧数据中提取当前帧的参数b（通常表示偏移量或截距）
        c = animation_frames[i][2]  # 从帧数据中提取当前帧的参数c
        yy = a/-b * xx + c/-b       # 计算直线方程 y = (-a/b)x + (-c/b)
        line_d.set_data(xx, yy)     # 更新直线数据
        C1_dots.set_data(C1[:, 0], C1[:, 1]) # 更新C1和C2散点数据
        C2_dots.set_data(C2[:, 0], C2[:, 1]) # 更新C2散点数据
        frame_text.set_text(        # 更新帧文本信息，显示当前时间步和损失值
            'Timestep = %.1d/%.1d\nLoss = %.3f' % 
            (i, len(animation_frames), animation_frames[i][3])
        )
        return (line_d,) + (C1_dots,) + (C2_dots,)  # 返回需要更新的对象元组，用于blitting优化
    # 创建FuncAnimation对象
    anim = animation.FuncAnimation(
        f, animate, init_func=init, # 要绘制的图形对象，动画更新函数，初始化函数，设置动画初始状态
        frames=len(animation_frames), interval=50, blit=True, repeat=False # 帧间隔(毫秒)，是否使用blitting优化，# 是否循环播放
    )
    HTML(anim.to_html5_video())# 将动画转换为HTML5视频并显示
