#!/usr/bin/env python
# coding: utf-8

# # Softmax Regression Example

# ### 生成数据集， 看明白即可无需填写代码
# #### '<font color="blue">+</font>' 从高斯分布采样 (X, Y) ~ N(3, 6, 1, 1, 0).<br>
# #### '<font color="green">o</font>' 从高斯分布采样  (X, Y) ~ N(6, 3, 1, 1, 0)<br>
# #### '<font color="red">*</font>' 从高斯分布采样  (X, Y) ~ N(7, 7, 1, 1, 0)<br>

# In[1]:

# 导入运行所需模块
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import matplotlib.cm as cm
import numpy as np

# get_ipython().run_line_magic('matplotlib', 'inline')  # 仅在Jupyter环境下需要

# 设置数据点数量
dot_num = 100  

# 生成类别1的数据：均值为(3,6)，标准差为1
x_p = np.random.normal(3.0, 1, dot_num) 
y_p = np.random.normal(6.0, 1, dot_num)
y = np.ones(dot_num)  # 标签为1
C1 = np.array([x_p, y_p, y]).T  # 组合成(x, y, label)格式

# 生成类别2的数据：均值为(6,3)，标准差为1
x_n = np.random.normal(6.0, 1, dot_num)
y_n = np.random.normal(3.0, 1, dot_num)
y = np.zeros(dot_num)  # 标签为0
C2 = np.array([x_n, y_n, y]).T

# 生成类别3的数据：均值为(7,7)，标准差为1
x_b = np.random.normal(7.0, 1, dot_num)
y_b = np.random.normal(7.0, 1, dot_num)
y = np.ones(dot_num) * 2  # 标签为2
C3 = np.array([x_b, y_b, y]).T

# 绘制三类样本的散点图
plt.scatter(C1[:, 0], C1[:, 1], c="b", marker="+")  # 类别1：蓝色加号
plt.scatter(C2[:, 0], C2[:, 1], c="g", marker="o")  # 类别2：绿色圆圈
plt.scatter(C3[:, 0], C3[:, 1], c="r", marker="*")  # 类别3：红色星号

# 合并所有类别的数据，形成完整数据集
data_set = np.concatenate((C1, C2, C3), axis=0)
np.random.shuffle(data_set)  # 随机打乱数据集顺序

# ## 建立模型
# 建立模型类，定义loss函数，定义一步梯度下降过程函数
#
# 填空一：在`__init__`构造函数中建立模型所需的参数
#
# 填空二：实现softmax的交叉熵损失函数(不使用tf内置的loss 函数)

# In[1]:

epsilon = 1e-12  # 防止 log(0)，处理数值稳定性问题

class SoftmaxRegression(tf.Module):
    def __init__(self, input_dim=2, num_classes=3):
        """
        初始化 Softmax 回归模型参数
        :param input_dim: 输入特征维度
        :param num_classes: 类别数量
        """
        super().__init__()
        # 初始化权重 W 和偏置 b
        # 使用均匀分布随机初始化权重，偏置初始化为0
        self.W = tf.Variable(
            tf.random.uniform([input_dim, num_classes], minval=-0.1, maxval=0.1),
            name = "W",
        )
        self.b = tf.Variable(tf.zeros([num_classes]), name="b") # 全0初始化，形状为[类别数]，变量名称为b

    @tf.function
    def __call__(self, x):
        """
        模型前向传播：计算线性变换并应用softmax函数得到概率分布
        :param x: 输入数据，shape = (N, input_dim)
        :return: softmax 概率分布，shape = (N, num_classes)
        """
        # 计算线性变换 logits
        logits = tf.matmul(x, self.W) + self.b
        # 应用softmax函数，将logits转换为概率分布
        return tf.nn.softmax(logits)

@tf.function
def compute_loss(pred, labels, num_classes=3):
    """
    计算交叉熵损失和准确率
    :param pred: 模型输出，shape = (N, num_classes)
    :param labels: 实际标签，shape = (N,)
    :param num_classes: 类别数
    :return: 平均损失值和准确率
    """
    # 将真实标签转换为one-hot编码形式
    one_hot_labels = tf.one_hot(
        tf.cast(labels, tf.int32), depth=num_classes, dtype=tf.float32
    )
    
    # 防止log(0)的情况，将预测概率限制在[epsilon, 1.0]范围内
    pred = tf.clip_by_value(pred, epsilon, 1.0)
    
    # 计算每个样本的交叉熵损失，对于每个样本，计算其真实类别的概率的负对数
    # 公式：L_i = -Σ_{c=1}^C y_{i,c} · log(p_{i,c})
    # 其中：
    #   y_{i,c} 是样本i的真实类别c的one-hot值（0或1）
    #   p_{i,c} 是模型预测样本i属于类别c的概率
    #   当y_{i,c}=1时（即样本i的真实类别为c），对应的log(p_{i,c})才会被计入损失
    sample_losses = -tf.reduce_sum(one_hot_labels * tf.math.log(pred), axis=1)
    
    # 计算所有样本的平均损失
    loss = tf.reduce_mean(sample_losses)
    
    # 计算准确率，比较模型预测的类别和真实类别是否一致
    # tf.argmax(pred, axis=1) 获取预测的类别索引（概率最大的位置）
    # tf.argmax(one_hot_labels, axis=1) 获取真实类别索引
    acc = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.argmax(pred, axis=1),
                tf.argmax(one_hot_labels, axis=1)
            ),
            dtype=tf.float32,
        )
    )
    # 返回损失值和准确率
    # loss: 预先计算好的损失值（如交叉熵损失）
    # acc: 当前批次的分类准确率（0-1 标量）
    return loss, acc

@tf.function
def train_one_step(model, optimizer, x_batch, y_batch):
    # 单步训练：计算梯度并更新参数
    """
    一步梯度下降优化
    :param model: SoftmaxRegression 实例
    :param optimizer: 优化器（如 Adam, SGD）
    :param x_batch: 输入特征
    :param y_batch: 标签
    :return: 当前批次的损失与准确率
    """
    with tf.GradientTape() as tape:
        predictions = model(x_batch)  # 前向传播：计算模型对输入批次的预测
        loss, accuracy = compute_loss(predictions, y_batch)  # 计算损失和准确率

    grads = tape.gradient(loss, model.trainable_variables)  # 自动计算损失函数对模型参数的梯度
    optimizer.apply_gradients(zip(grads, model.trainable_variables))  # 优化步骤：使用优化器将计算出的梯度应用到模型参数上
    return loss, accuracy  # 返回当前批次的损失和准确率

# ### 实例化一个模型，进行训练，提取所需的数据

# In[12]:

model = SoftmaxRegression()
# 创建一个 SoftmaxRegression 模型实例 model
opt = tf.keras.optimizers.SGD(learning_rate=0.01)
# 创建随机梯度下降（SGD）优化器实例 opt，设置学习率为 0.01

x1, x2, y = list(zip(*data_set))
# 转换为 float32
x = np.array(list(zip(x1, x2)), dtype=np.float32)  
# 转换为 int32
y = np.array(y, dtype=np.int32)  
# 从混合数据集 data_set 中提取特征和标签，并转换为所需的数据类型

for i in range(1000):
    loss, accuracy = train_one_step(model, opt, x, y)
    if i % 50 == 49:
        print(f"loss: {loss.numpy():.4}\t accuracy: {accuracy.numpy():.4}")
# 执行 1000 次迭代的模型训练，并每隔 50 步打印损失和准确率

# # 结果展示，无需填写代码

# In[13]:

# 绘制三种不同类别的散点图
# C1[:, 0] 和 C1[:, 1] 分别表示 C1 的第一列和第二列数据（通常是特征）
plt.scatter(C1[:, 0], C1[:, 1], c="b", marker="+") # c="b" 设置颜色为蓝色，marker="+" 设置标记为加号
plt.scatter(C2[:, 0], C2[:, 1], c="g", marker="o") # c="g" 设置颜色为绿色，marker="o" 设置标记为圆形
plt.scatter(C3[:, 0], C3[:, 1], c="r", marker="*") # c="r" 设置颜色为红色，marker="*" 设置标记为星号

# 创建网格点用于绘制决策边界
x = np.arange(0.0, 10.0, 0.1)
y = np.arange(0.0, 10.0, 0.1)

# 生成网格坐标矩阵
# 将网格坐标展平并组合为输入特征矩阵
X, Y = np.meshgrid(x, y)
# 将X和Y数组重塑为一维数组后进行配对组合
inp = np.array(list(zip(X.reshape(-1), Y.reshape(-1))), dtype=np.float32)
print(inp.shape)
# 模型预测
Z = model(inp)
# 获取预测的类别
Z = np.argmax(Z, axis=1)
# 重塑为网络形状
Z = Z.reshape(X.shape)
# 绘制决策边界
plt.contour(X, Y, Z, alpha=0.5)
plt.show()

# 保存模型权重
model.save_weights('softmax_regression_weights')

# 加载模型权重
model.load_weights('softmax_regression_weights')

# In[ ]:
