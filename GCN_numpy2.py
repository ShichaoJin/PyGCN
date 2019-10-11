#!coding:utf-8
#https://www.jiqizhixin.com/articles/2019-02-20-12
#learn GCN with numpy 
from IPython import embed
import numpy as np

def relu(inX): #https://blog.csdn.net/tintinetmilou/article/details/78186896
    return np.maximum(0,inX)

from networkx import to_numpy_matrix
import networkx as nx
zkc = nx.karate_club_graph()
order = sorted(list(zkc.nodes()))
A = to_numpy_matrix(zkc, nodelist=order)
I = np.eye(zkc.number_of_nodes())
A_hat = A + I
D_hat = np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.matrix(np.diag(D_hat))
#接下来，我们将随机初始化权重。


W_1 = np.random.normal(loc=0, scale=1, size=(zkc.number_of_nodes(), 4))
W_2 = np.random.normal(loc=0, size=(W_1.shape[1], 2))

#接着，我们会堆叠 GCN 层。这里，我们只使用单位矩阵作为特征表征，即每个节点被表示为一个 one-hot 编码的类别变量。
def gcn_layer(A_hat, D_hat, X, W):
    DhAh = np.dot(np.linalg.inv(D_hat),A_hat)
    DhAhX = np.dot(DhAh,X)
    DhAhXW = np.dot(DhAhX,W)
    return relu(DhAhXW)

H_1 = gcn_layer(A_hat, D_hat, I, W_1)
H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)
output = H_2


#我们进一步抽取出特征表征。
feature_representations = {
    node: np.array(output)[node] 
    for node in zkc.nodes()}
print(feature_representations)

labels =[]
for i in range(len(zkc.nodes())):
    if zkc.node[i]['club'] == 'Mr. Hi':
        labels.append('r')
    else:
        labels.append('b')

#labels = np.asarray(labels)
xy = np.asarray(output)
from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111) #, projection = '3d')
for i in range(xy.shape[0]):
    #embed()
    ax.scatter(xy[:,0][i], xy[:,1][i],  color=labels[i])
plt.show()

embed()


if __name__ == "__main__":
    
    embed()

