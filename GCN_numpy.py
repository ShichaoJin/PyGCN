#!coding:utf-8
#https://www.jiqizhixin.com/articles/2019-02-20-12
#learn GCN with numpy 
from IPython import embed
import numpy as np
#一个简单的有向图
#使用 numpy 编写的上述有向图的邻接矩阵表征如下：
A = np.matrix([
    [0, 1, 0, 0],
    [0, 0, 1, 1], 
    [0, 1, 0, 0],
    [1, 0, 1, 0]],
    dtype=float
)
#接下来，我们需要抽取出特征！我们基于每个节点的索引为其生成两个整数特征，这简化了本文后面手动验证矩阵运算的过程

X = np.matrix([
            [i, -i]
            for i in range(A.shape[0])
        ], dtype=float)
#应用传播规则
#我们现在已经建立了一个图，其邻接矩阵为 A，输入特征的集合为 X。下面让我们来看看，当我们对其应用传播规则后会发生什么：
AX = np.dot(A,X) #每个节点的表征（每一行）现在是其相邻节点特征的和！换句话说，图卷积层将每个节点表示为其相邻节点的聚合。

#问题
'''
节点的聚合表征不包含它自己的特征！该表征是相邻节点的特征聚合，因此只有具有自环（self-loop）的节点才会在该聚合中包含自己的特征 [1]。

度大的节点在其特征表征中将具有较大的值，度小的节点将具有较小的值。这可能会导致梯度消失或梯度爆炸 [1, 2]，也会影响随机梯度下降算法（随机梯度下降算法通常被用于训练这类网络，且对每个输入特征的规模（或值的范围）都很敏感）。
'''

#增加自环
I = np.matrix(np.eye(A.shape[0]))
A_hat = A + I
AhX = np.dot(A_hat,X) #现在，由于每个节点都是自己的邻居，每个节点在对相邻节点的特征求和过程中也会囊括自己的特征！

#对特征表征进行归一化处理
'''通过将邻接矩阵 A 与度矩阵 D 的逆相乘，对其进行变换，从而通过节点的度对特征表征进行归一化。因此，我们简化后的传播规则如下：
f(X, A) = D⁻¹AX'''
Dh = np.array(np.sum(A_hat, axis=0))[0]
Dh = np.matrix(np.diag(Dh))
print(A_hat) #变换之前
DhAh = np.dot(np.linalg.inv(Dh),A_hat) #可以观察到，邻接矩阵中每一行的权重（值）都除以该行对应节点的度。我们接下来对变换后的邻接矩阵

#应用传播规则：
DhAhX = np.dot(DhAh,X) #得到与相邻节点的特征均值对应的节点表征。这是因为（变换后）邻接矩阵的权重对应于相邻节点特征加权和的权重。

#添加权重
#首先要做的是应用权重。请注意，这里的 D_hat 是 A_hat = A + I 对应的度矩阵，即具有强制自环的矩阵 A 的度矩阵。
W = np.matrix([
             [1, -1],
             [-1, 1]
         ])
DhAhXW = np.dot(DhAhX, W)
#如果我们想要减小输出特征表征的维度，我们可以减小权重矩阵 W 的规模：
W = np.matrix([
             [1],
             [-1]
         ])
DhAhXW = np.dot(DhAhX, W)

#添加激活函数
#relu(DhAhXW)

if __name__ == "__main__":
    
    embed()
