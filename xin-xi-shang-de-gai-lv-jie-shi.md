以二分类问题为例，即样本中只有两类样本：正例\(标记为“1”\)，负例\(标记为“0”\)。为了对二分类问题建模，我们需要一个假设，即假设第_i_个样本的模型结果pi的靠谱程度由如下公式衡量：

![](/assets/li.png)

其中yi表示第_i_个样本的类标签，

![](http://dl2.iteye.com/upload/attachment/0116/5151/bf77f456-edb6-38f5-b56b-534f8cc5ca3e.png)

，通俗来讲就是假设样本类别判断正确的概率服从给定参数

![](http://dl2.iteye.com/upload/attachment/0116/5153/9104422a-f26e-3def-8bb5-3381fb1a6799.png)

的二项分布，也叫伯努利分布，每个样本都服从相同的分布，且相互之间独立。这样，我们可以写出整个数据集的似然函数，也就是该二分类问题的目标函数：

![](/assets/bernuli.png)

模型优化的目的，或者说构建一棵决策树的目的，就是使得该公式的值尽可能的大。那这个跟我们要讨论的信息增益\(熵的降低\)有什么关系？不急，我们先对目标函数两边分别取对数并取反得到：

![](http://dl2.iteye.com/upload/attachment/0116/4630/1b314445-71bb-38ca-8809-465aec268422.png)求原函数的最大值等价于求该函数的最小值。

由于该函数对参数 ![](http://dl2.iteye.com/upload/attachment/0116/5155/efae5ffd-c634-3f53-8932-405109d60ce7.png) 的二阶导大于0恒成立，如下：

  


![](http://dl2.iteye.com/upload/attachment/0116/4632/515c29c5-f75a-32f7-933b-33fce0c1f6b8.png)

        因此，这是一个有且仅有一个最小值的凸函数，为求极值对应的函数参数，只需求其一阶导等于0即可，即求 pi 使得：

![](http://dl2.iteye.com/upload/attachment/0116/4634/af0a16eb-bcc4-3eaf-8eb7-2c980d5fd093.png)

