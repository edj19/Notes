# Pytorch教程



## 1 安装

![image-20200926092751488](C:\Users\edj\AppData\Roaming\Typora\typora-user-images\image-20200926092751488.png)

CUDA（Compute Unified Device Architecture），是显卡厂商[NVIDIA](https://baike.baidu.com/item/NVIDIA)推出的运算平台。 CUDA™是一种由NVIDIA推出的通用[并行计算](https://baike.baidu.com/item/并行计算/113443)架构，该架构使[GPU](https://baike.baidu.com/item/GPU)能够解决复杂的计算问题。 它包含了CUDA[指令集架构](https://baike.baidu.com/item/指令集架构)（[ISA](https://baike.baidu.com/item/ISA)）以及GPU内部的并行计算引擎。 （以上解释来自百度词条）

了解了CUDA是什么，那么我的CUDA版本号是多少呢？（以下列出两种方式，我只是成功了第一种方式，可以直接看第一种就好了）

![image-20200926093035492](C:\Users\edj\AppData\Roaming\Typora\typora-user-images\image-20200926093035492.png)



安装完成进行验证：

```python
from __future__ import print_function
import torch
x = torch.rand(5,3)
print(x)
# tensor([[0.6406, 0.0381, 0.1101],
#        [0.4794, 0.0882, 0.3598],
 #       [0.3074, 0.6615, 0.6951],
  #      [0.8943, 0.1857, 0.8353],
   #     [0.5862, 0.0350, 0.1165]])
```

此外，要检查您的 GPU 驱动程序和 CUDA 是否已启用并可被 PyTorch 访问，请运行以下命令以返回 CUDA 驱动程序是否已启用。

```python
import torch
torch.cuda.is_available()

#True
```



















## 2 人工神经网络

在人工神经网络里, 没有凭空产生新联结这回事. 人工神经网络典型的一种学习方式就是, 我已经知道吃到糖果时, 手会如何动, 但是我想让神经网络学着帮我做这件动动手的事情. 所以我预先准备好非常多吃糖的学习数据, 然后将这些数据一次次放入这套人工神经网络系统中, 糖的信号会通过这套系统传递到手. 然后通过对比这次信号传递后, 手的动作是不是”讨糖”动作, 来修改人工神经网络当中的神经元强度. 这种修改在专业术语中叫做”误差反向传递”, 也可以看作是再一次将传过来的信号传回去, 看看这个负责传递信号神经元对于”讨糖”的动作到底有没有贡献, 让它好好反思与改正, 争取下次做出更好的贡献. 这样看来, 人工神经网络和生物神经网络的确不是一回事.



人工神经网络靠的是正向和反向传播来更新神经元, 从而形成一个好的神经系统, 本质上, 这是一个能让计算机处理和优化的数学模型. 而生物神经网络是通过刺激, 产生新的联结, 让信号能够通过新的联结传递而形成反馈. 虽然现在的计算机技术越来越高超, 不过我们身体里的神经系统经过了数千万年的进化, 还是独一无二的, 迄今为止, 再复杂, 再庞大的人工神经网络系统也不能替代我们的小脑袋. 我们应该感到自豪, 也应该珍惜上天的这份礼物.



神经网络是当今最流行的一种深度学习机制，它的基本原理也很简单，就是一种梯度下降机制。

**Optimization**





## 3 Pytorch教程

[PyTorch](http://pytorch.org/) 是 [PyTorch](http://pytorch.org/) 在 Python 上的衍生. 因为 [PyTorch](http://pytorch.org/) 是一个使用 [PyTorch](http://pytorch.org/) 语言的神经网络库, Torch 很好用, 但是 Lua 又不是特别流行, 所有开发团队将 Lua 的 Torch 移植到了更流行的语言 Python 上. 是的 PyTorch 一出生就引来了剧烈的反响. 

## 3.1 Deep Learning with Pytorch: A 60 minute blitz

* 高水平地理解PyTorch的Tensor库和神经网络。
* 训练一个小型神经网络来对图像进行分类。

**首先我们需要了解什么是Pytorch?**

这是一个基于Python的科学计算包，针对两类受众。

* NumPy的替代者，利用GPU的强大功能
* 深度学习研究平台，提供最大的灵活性和速度。

现在开始学习，首先需要张量的含义：

Tensors:Tensors类似于NumPy的ndarrays，增加的是Tensors也可以在GPU上使用，以加速计算

构建一个未初始化的5×3矩阵：

注意：一个未初始化的矩阵被声明，但在使用前并不包含明确的已知值。当创建一个未初始化的矩阵时，当时在分配的内存中的任何值都将作为初始值出现。

```python
from __future__ import print_function
import torch

x = torch.empty(5,3)

print(x)

tensor([[4.2246e-39, 1.0286e-38, 1.0653e-38],
        [1.0194e-38, 8.4490e-39, 1.0469e-38],
        [9.3674e-39, 9.9184e-39, 8.7245e-39],
        [9.2755e-39, 8.9082e-39, 9.9184e-39],
        [8.4490e-39, 9.6429e-39, 1.0653e-38]])
```

构建一个随机的初始化矩阵：

```python
y = torch.rand(5,3)

print(y)

tensor([[0.1145, 0.9702, 0.7530],
        [0.7579, 0.9220, 0.0424],
        [0.8180, 0.7831, 0.9887],
        [0.3857, 0.7545, 0.5047],
        [0.7996, 0.0504, 0.5500]])
```

创建一个由0组成的long类型的矩阵：

```python
x = torch.zeros(5,3,dtype=torch.long)
print(x)

tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
```

从数据直接创建张量：

```python
x = torch.tensor([[5.5,3],[7,8.9]])

tensor([[5.5000, 3.0000],
        [7.0000, 8.9000]])
```

或基于现有张量创建一个张量。这些方法将重用输入张量的属性，例如dtype，除非用户提供新的值。

```python
x = x.new_ones(5,3,dtype=torch.double)
print(x)

tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
        
        
        
x = torch.randn_like(x,dtype=torch.float)
print(x)

tensor([[-0.5572,  0.0364, -0.5343],
        [ 1.2312,  1.3603,  0.7347],
        [-1.5149, -0.5823,  1.4924],
        [ 1.6598,  0.3817, -2.3977],
        [ 0.1060, -0.7908,  1.4388]])
```

获得

























































































