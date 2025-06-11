#!/usr/bin/env python
# coding: utf-8

# 设计基函数(basis function) 以及数据读取
# 导入NumPy库 - 用于高性能科学计算和多维数组处理
import numpy as np
# 导入Matplotlib的pyplot模块 - 用于数据可视化和绘图
import matplotlib.pyplot as plt
# 导入Matplotlib的pyplot模块 - 用于数据可视化和绘图
import tensorflow as tf
# 从Keras导入常用模块
from tensorflow.keras import optimizers, layers, Model


def identity_basis(x):
    """恒等基函数：直接返回输入本身，适用于线性回归"""
    return np.expand_dims(x, axis=1)

    # 生成多项式基函数
def multinomial_basis(x, feature_num=10):
    """多项式基函数：将输入x映射为多项式特征
    feature_num: 多项式的最高次数"""
    x = np.expand_dims(x, axis=1)  # shape(N, 1)
    # 初始化特征列表
    feat = [x]
    # 生成从 x^2 到 x^feature_num 的多项式特征
    for i in range(2, feature_num + 1):
        feat.append(x**i)
    # 将所有特征沿着第二维（axis=1）拼接起来
    ret = np.concatenate(feat, axis=1)
    return ret # 返回一个二维数组，其中每一行是输入样本的多项式特征向量，列数为 feature_num


def gaussian_basis(x, feature_num=10):
    """高斯基函数：将输入x映射为一组高斯分布特征
    用于提升模型对非线性关系的拟合能力"""
    # 使用np.linspace在区间[0, 25]上均匀生成feature_num个中心点
    centers = np.linspace(0, 25, feature_num)
    # 计算高斯函数的宽度(标准差)
    width = 1.0 * (centers[1] - centers[0])
    # 使用np.expand_dims在x的第1维度(axis=1)上增加一个维度
    x = np.expand_dims(x, axis=1)
    # 将x沿着第1维度(axis=1)复制feature_num次并连接
    x = np.concatenate([x] * feature_num, axis=1)
    
    out = (x - centers) / width  # 计算每个样本点到每个中心点的标准化距离
    ret = np.exp(-0.5 * out ** 2)  # 对标准化距离应用高斯函数
    return ret


def load_data(filename, basis_func=gaussian_basis):
    """载入数据并进行基函数变换
    返回：(特征, 标签), (原始x, 原始y)"""
    xys = []
    with open(filename, "r") as f:
        for line in f:
            # 读取每一行数据，并将其转换为列表
            # 改进: 转换为list
            xys.append(list(map(float, line.strip().split()))) # 读取每行数据
        xs, ys = zip(*xys) # 解压为特征和标签
        xs, ys = np.asarray(xs), np.asarray(ys) # 转换为numpy数组
        o_x, o_y = xs, ys # 保存原始数据
        phi0 = np.expand_dims(np.ones_like(xs), axis=1) # 添加偏置项（全1列）
        phi1 = basis_func(xs) # 应用基函数变换
        xs = np.concatenate([phi0, phi1], axis=1) 
        # 拼接偏置和变换后的特征
        return (np.float32(xs), np.float32(ys)), (o_x, o_y)# 返回处理好的训练数据和原始数据


#定义模型
class linearModel(Model):
    """线性回归模型，实现 y = w·x + b"""
    
    def __init__(self, ndim):
        """
        初始化线性模型
        
        参数:
        ndim: 输入特征的维度
        """
        # 调用父类(Model)的构造函数
        super(linearModel, self).__init__()
        
        # 定义模型参数：权重矩阵 w
        # 形状为 [ndim, 1]，表示从 ndim 维输入到 1 维输出的线性变换
        # 初始值从均匀分布 [-0.1, 0.1) 中随机生成
        # trainable=True 表示该变量需要在训练过程中被优化
        self.w = tf.Variable(
            shape=[ndim, 1],    # 权重矩阵形状：ndim×1
            initial_value=tf.random.uniform(
                [ndim, 1], minval=-0.1, maxval=0.1, dtype=tf.float32
            ),
            trainable=True,
            name="weight"
        )
        
        # 注意：代码中缺少偏置项 b，完整的线性模型通常需要包含偏置
        # 可补充：
        # self.b = tf.Variable(
        #     initial_value=tf.zeros([1]),
        #     trainable=True,
        #     name="bias"
        # )
        
    @tf.function
    def call(self, x):
        """模型前向传播
        
        参数:
            x: 输入特征，形状为(batch_size, ndim)
            
        返回:
            预测值，形状为(batch_size,)
        """
        y = tf.squeeze(tf.matmul(x, self.w), axis=1)  # 矩阵乘法后压缩维度
        return y


    (xs, ys), (o_x, o_y) = load_data("train.txt")    # 调用load_data函数      
    ndim = xs.shape[1]  # 获取特征维度

    model = linearModel(ndim=ndim)  # 实例化线性模型


#训练以及评估
optimizer = optimizers.Adam(0.1)


@tf.function
def train_one_step(model, xs, ys):
    # 在梯度带(GradientTape)上下文中记录前向计算过程
    with tf.GradientTape() as tape:
        y_preds = model(xs)    # 模型前向传播计算预测值
        loss = tf.keras.losses.MSE(ys, y_preds)   #计算损失函数
    grads = tape.gradient(loss, model.w)    # 计算损失函数对模型参数w的梯度
    optimizer.apply_gradients([(grads, model.w)])    # 更新模型参数
    return loss


# 使用@tf.function装饰器将Python函数转换为TensorFlow图，以提高执行效率
@tf.function
def predict(model, xs):
    # 使用模型对输入xs进行预测（前向传播）
    y_preds = model(xs)  # 模型前向传播
    
    # 返回模型的预测结果
    return y_preds


def evaluate(ys, ys_pred):
    """评估模型。"""
    std = np.std(ys - ys_pred) # 计算预测误差的标准差
    return std


# 评估指标的计算
for i in range(1000): # 进行1000次训练迭代
    loss = train_one_step(model, xs, ys) # 执行单步训练并获取当前损失值
    if i % 100 == 1: # 每100步打印一次损失值（从第1步开始：1, 101, 201, ...）
        print(f"loss is {loss:.4}")  # `:.4` 表示保留4位有效数字

# 使用模型对训练集数据进行预测
y_preds = predict(model, xs)
# 打印测试集预测值与真实值的标准差
std = evaluate(ys, y_preds)
# 打印训练集预测值与真实值的标准差
print("训练集预测值与真实值的标准差：{:.1f}".format(std)) # 格式化输出标准差，保留一位小数

# 加载测试集数据
(xs_test, ys_test), (o_x_test, o_y_test) = load_data("test.txt")

# 使用模型对测试集数据进行预测
y_test_preds = predict(model, xs_test)
# 计算测试集预测值与真实值的标准差
std = evaluate(ys_test, y_test_preds)
# 打印测试集预测值与真实值的标准差
print("训练集预测值与真实值的标准差：{:.1f}".format(std))

# 绘制原始数据点：红色圆点标记，大小3
# o_x: 原始数据X坐标
# o_y: 原始数据Y坐标
# "ro": 红色(r)圆形(o)标记
plt.plot(o_x, o_y, "ro", markersize=3) 
# 绘制模型预测曲线：黑色实线
# o_x_test: 测试集X坐标
# y_test_preds: 模型在测试集上的预测结果
# "k": 黑色(k)实线（默认线型）
plt.plot(o_x_test, y_test_preds, "k")
# 设置x、y轴标签
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression") # 图表标题
# 虚线网格，半透明灰色
plt.grid(True, linestyle="--", alpha=0.7, color="gray")
plt.legend(["train", "test", "pred"]) # 添加图例，元素依次对应
plt.tight_layout()  # 自动调整布局
plt.show()
