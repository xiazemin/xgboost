# 决策树的数学原理

通俗来讲，决策树的构建过程就是将数据根据其特征分布划分到不同的区域，使得同一个区域的样本有尽可能一致的类别标签。在决策树构建的过程中，我们需要一个衡量标准来确定每次数据划分所带来的收益，这个标准就是信息熵，以0-1二分类问题为例，衡量一个节点的信息熵公式如下：

其中p为当前节点中正样本的比例，Entropy越大，说明节点的样本越杂，因此Entropy越小越好。假设我们每次对数据划分都是将数据一分为二，分别为left和right， 分裂的收益就是分裂前节点的Entropy减去这两个节点的Entropy的加权和。即：Entropy\\(parent\\) - Prob\\(left\\) \\* Entropy\\(left\\) + Prob\\(right\\) \\* Entropy\\(right\\)，这个值越大越好。这个收益，学术上我们称作“信息增益”。其中Prob\\(left\\)为左节点的样比例，Prob\\(right\\)为右节点的样本比例。 由于单纯使用信息增益作为标准来构建决策树，容易导致过拟合的问题。因此前辈们又引入了“信息增益率”，以及对树进行剪枝等方式来优化树的创建过程。  



