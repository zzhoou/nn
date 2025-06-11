#!/usr/bin/env python
# coding: utf-8

# # 加法进位实验
# 这个实验展示了如何使用RNN学习大整数加法的进位机制

# <img src="https://github.com/JerrikEph/jerrikeph.github.io/raw/master/Learn2Carry.png" width=650>

# In[1]:


# 导入必要的库和模块
import numpy as np  # 数值计算库
import tensorflow as tf  # 深度学习框架
import collections  # 提供有用的数据结构
from tensorflow import keras  # TensorFlow的高级API
from tensorflow.keras import layers  # Keras的层模块
from tensorflow.keras import layers, optimizers, datasets  # 从Keras导入层、优化器和数据集
import os  # 操作系统接口
import sys  # 系统相关参数和函数
import tqdm  # 进度条工具


# 数据生成
# 我们随机在 `start->end`之间采样除整数对`(num1, num2)`，计算结果`num1+num2`作为监督信号。
# * 首先将数字转换成数字位列表 `convertNum2Digits`
# * 将数字位列表反向
# * 将数字位列表填充到同样的长度 `pad2len`

# In[2]:


def gen_data_batch(batch_size: int, start: int, end: int) -> tuple:
    '''在(start, end)区间采样生成一个batch的整型的数据
    Args :
        batch_size: batch_size
        start: 开始数值
        end: 结束数值
    '''
    # 生成随机数
    numbers_1 = np.random.randint(start, end, batch_size)
    numbers_2 = np.random.randint(start, end, batch_size)
    results = numbers_1 + numbers_2
    return numbers_1, numbers_2, results

def convertNum2Digits(Num):
    '''将一个整数转换成一个数字位的列表,例如 133412 ==> [1, 3, 3, 4, 1, 2]
    '''
    strNum = str(Num)
    chNums = list(strNum)
    digitNums = [int(o) for o in strNum]
    return digitNums

def convertDigits2Num(Digits):
    '''将数字位列表反向， 例如 [1, 3, 3, 4, 1, 2] ==> [2, 1, 4, 3, 3, 1]
    '''# 便于RNN按低位到高位处理
    digitStrs = [str(o) for o in Digits]
    numStr = ''.join(digitStrs)
    Num = int(numStr)
    return Num

def pad2len(lst, length, pad=0):
    '''将一个列表用`pad`填充到`length`的长度 例如 pad2len([1, 3, 2, 3], 6, pad=0) ==> [1, 3, 2, 3, 0, 0]
    '''#用0填充数位列表至固定长度，适配批量训练。
    lst+=[pad]*(length - len(lst))
    return lst

def results_converter(res_lst):
    '''将预测好的数字位列表批量转换成为原始整数
    Args:
        res_lst: shape(b_sz, len(digits))
    '''
    # 反转每个数字位列表，因为我们在输入时反转了数字
    res = [reversed(digits) for digits in res_lst]
    return [convertDigits2Num(digits) for digits in res]

def prepare_batch(Nums1, Nums2, results, maxlen):
    '''准备一个batch的数据，将数值转换成反转的数位列表并且填充到固定长度
    #1. 将整数转换为数字位列表
    #2. 反转数字位列表(低位在前，高位在后)
    #3. 填充到固定长度
    
    Args:
        Nums1: shape(batch_size,)
        Nums2: shape(batch_size,)
        results: shape(batch_size,)
        maxlen:  type(int)
    Returns:
        Nums1: shape(batch_size, maxlen)
        Nums2: shape(batch_size, maxlen)
        results: shape(batch_size, maxlen)
    '''
     # 将整数转换为数字位列表
    Nums1 = [convertNum2Digits(o) for o in Nums1]
    Nums2 = [convertNum2Digits(o) for o in Nums2]
    results = [convertNum2Digits(o) for o in results]
    # 反转数字位列表，使低位在前，高位在后
    # 这有助于RNN学习进位机制，因为低位的计算影响高位
    Nums1 = [list(reversed(o)) for o in Nums1]
    Nums2 = [list(reversed(o)) for o in Nums2]
    results = [list(reversed(o)) for o in results]
    # 填充所有列表到相同长度
    Nums1 = [pad2len(o, maxlen) for o in Nums1]
    Nums2 = [pad2len(o, maxlen) for o in Nums2]
    results = [pad2len(o, maxlen) for o in results]
    
    return Nums1, Nums2, results


# # 建模过程， 按照图示完成建模

# In[3]:


class myRNNModel(keras.Model):
    def __init__(self):
        super(myRNNModel, self).__init__()
        # 嵌入层：将数字0-9转换为32维向量
        # 输入的数字范围是0-9，嵌入维度为32，batch_input_shape=[None, None]表示输入的批次大小和序列长度是动态的
        self.embed_layer = tf.keras.layers.Embedding(10, 32, 
                                                    batch_input_shape=[None, None])
       
        # 基础RNN单元和RNN层
        # 定义一个基础的RNN单元，隐藏层大小为64
        self.rnncell = tf.keras.layers.SimpleRNNCell(64)
        # 构建RNN层，使用定义的RNN单元，并返回整个序列的输出
        self.rnn_layer = tf.keras.layers.RNN(self.rnncell, return_sequences=True)
        # 分类层：预测每个时间步的数字（0-9）
        # 使用一个全连接层，输出维度为10，对应于数字0-9的概率分布
        self.dense = tf.keras.layers.Dense(10) 
        
    @tf.function
    def call(self, num1, num2):
        
         #模型前向传播过程：
        #1. 将两个输入数字的每个位进行嵌入
        #2. 将嵌入后的向量相加
        #3. 通过RNN处理相加后的向量序列
        #4. 通过全连接层预测每个位的数字
      Args:
            num1: 第一个输入数字，shape为(batch_size, maxlen)
            num2: 第二个输入数字，shape为(batch_size, maxlen)
            
        Returns:
            logits: 预测结果，shape为(batch_size, maxlen, 10)
        # 嵌入处理
        embed1 = self.embed_layer(num1)  # [batch_size, maxlen, embed_dim]
        embed2 = self.embed_layer(num2)  # [batch_size, maxlen, embed_dim]
        
        # 将两个输入的嵌入向量相加
        inputs = tf.concat([emb1, emb2], axis=-1)  # [batch_size, maxlen, embed_dim]
        
        # 通过RNN层处理
        rnn_out = self.rnn_layer(inputs)  # [batch_size, maxlen, rnn_units]
        
        # 通过全连接层得到每个位的预测结果
        logits = self.dense(rnn_out)  # [batch_size, maxlen, 10]
        
        return logits
    
# In[4]:


@tf.function
def compute_loss(logits, labels):# 使用 sparse_softmax_cross_entropy_with_logits 计算每个样本的交叉熵损失
    # 输入是 logits 和对应的 labels（真实类别索引）
    # 输出是一个形状为 (B,) 的损失张量
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
    return tf.reduce_mean(losses)# 对所有样本的损失求平均，得到一个标量值作为最终的 loss

@tf.function
def train_one_step(model, optimizer, x, y, label):
    with tf.GradientTape() as tape: #使用 TensorFlow 的梯度磁带（GradientTape）上下文管理器，自动追踪该作用域内的所有可训练变量操作
        logits = model(x, y)
        loss = compute_loss(logits, label) #对比模型的预测值 logits 和真实标签 label，输出当前损失值

    # 计算梯度
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables)) #将梯度与对应的模型参数配对，使用优化器按梯度方向更新模型参数
    return loss


def train(steps, model, optimizer):
    loss = 0.0
    accuracy = 0.0
    for step in range(steps):
        # 生成训练数据（数值范围0~555,555,554）
        datas = gen_data_batch(batch_size=200, start=0, end=555555555)
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)
        # 单步训练：计算损失、更新参数
        loss = train_one_step(model, optimizer, tf.constant(Nums1, dtype=tf.int32), 
                              tf.constant(Nums2, dtype=tf.int32),
                              tf.constant(results, dtype=tf.int32))
        if step % 50 == 0:
            print('step', step, ': loss', loss.numpy())

    return loss

def evaluate(model):
    # 评估模型在大数加法（555,555,555~999,999,998）上的准确率
    datas = gen_data_batch(batch_size=2000, start=555555555, end=999999999)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)
    logits = model(tf.constant(Nums1, dtype=tf.int32), tf.constant(Nums2, dtype=tf.int32))
    logits = logits.numpy() # 将模型输出的tensor转换为numpy数组，便于后续处理
    pred = np.argmax(logits, axis=-1) # 预测数位列表
    res = results_converter(pred)
    for o in list(zip(datas[2], res))[:20]:
        print(o[0], o[1], o[0]==o[1])

    print('accuracy is: %g' % np.mean([o[0]==o[1] for o in zip(datas[2], res)]))


# In[5]:


optimizer = optimizers.Adam(0.001)
model = myRNNModel()


# In[6]:


train(3000, model, optimizer)
evaluate(model)


# In[11]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




