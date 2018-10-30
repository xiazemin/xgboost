GBDT是以决策树（CART）为基学习器的GB算法，xgboost扩展和改进了GDBT，xgboost算法更快，准确率也相对高一些。    

## 1. Gradient boosting\(GB\)

      机器学习中的学习算法的目标是为了优化或者说最小化loss Function， Gradient boosting的思想是迭代生多个（M个）弱的模型，然后将每个弱模型的预测结果相加，后面的模型Fm+1\(x\)基于前面学习模型的Fm\(x\)的效果生成的，关系如下：

![](https://upload.wikimedia.org/math/e/4/7/e4746d8b9e912c1875cd9e5f612ab2ae.png "1 \le m \le M")![](https://upload.wikimedia.org/math/7/5/7/7579bdfb140e56a86da109ec96b87421.png "F\_{m+1}\(x\) = F\_m\(x\) + h\(x\)")

      GB算法的思想很简单，关键是怎么生成h\(x\)?

      如果目标函数是回归问题的均方误差，很容易想到最理想的h\(x\)应该是能够完全拟合![](https://upload.wikimedia.org/math/9/7/d/97dfe316cb29bf7289a75ade5be7de1c.png "y - F\_m\(x\)") ，这就是常说基于残差的学习。残差学习在回归问题中可以很好的使用，但是为了一般性（分类，排序问题），实际中往往是基于loss Function 在函数空间的的负梯度学习，对于回归问题![](https://upload.wikimedia.org/math/d/6/9/d69638d1c777cf19b3b3b04bc81d101e.png "\frac{1}{2}\(y - F\(x\)\)^2")残差和负梯度也是相同的。![](https://upload.wikimedia.org/math/6/1/3/613b882386356b2a890c03969c349759.png "L\(y, f\)")中的f，不要理解为传统意义上的函数，而是一个函数向量![](https://upload.wikimedia.org/math/a/9/6/a9631135ced6265452825a49f4c7badb.png "\! f\(x\_1\), \ldots, f\(x\_n\)")，向量中元素的个数与训练样本的个数相同，因此基于Loss Function函数空间的负梯度的学习也称为“伪残差”。

**GB算法的步骤：**

　　1.初始化模型为常数值：

![](https://upload.wikimedia.org/math/7/f/7/7f7ee5504c0d54de6d2510bdae7f723a.png "F\_0\(x\) = \underset{\gamma}{\arg\min} \sum\_{i=1}^n L\(y\_i, \gamma\).")

　　2.迭代生成M个基学习器

　　　　1.计算伪残差

![](https://upload.wikimedia.org/math/0/b/e/0bebe45631e9a1c4ed693590d60829c0.png "r\_{im} = -\left\[\frac{\partial L\(y\_i, F\(x\_i\)\)}{\partial F\(x\_i\)}\right\]\_{F\(x\)=F\_{m-1}\(x\)} \quad \mbox{for } i=1,\ldots,n.")

　　　　2.基于![](https://upload.wikimedia.org/math/2/b/b/2bbe4b0725baed85eae8dbbb20360ea6.png "\{\(x\_i, r\_{im}\)\}\_{i=1}^n")生成基学习器![](https://upload.wikimedia.org/math/7/f/d/7fd32efb7a21cc484be23d1015ee074e.png "\! h\_m\(x\)")

　　　　3.计算最优的![](https://upload.wikimedia.org/math/8/7/2/8721621535a98cea1b0e38459f594057.png "\! \gamma\_m")

![](https://upload.wikimedia.org/math/9/2/e/92e9576607b45540b71e23c14c737d0a.png "\gamma\_m = \underset{\gamma}{\operatorname{arg\,min}} \sum\_{i=1}^n L\left\(y\_i, F\_{m-1}\(x\_i\) + \gamma h\_m\(x\_i\)\right\).")

      　　4.更新模型

![](https://upload.wikimedia.org/math/3/b/0/3b047653ac126f19f09112f47ddb9f9c.png "F\_m\(x\) = F\_{m-1}\(x\) + \gamma\_m h\_m\(x\).")

##  2. Gradient boosting Decision Tree\(GBDT\)

　　GB算法中最典型的基学习器是决策树，尤其是CART，正如名字的含义，GBDT是GB和DT的结合。要注意的是这里的决策树是回归树，GBDT中的决策树是个弱模型，深度较小一般不会超过5，叶子节点的数量也不会超过10，对于生成的每棵决策树乘上比较小的缩减系数（学习率&lt;0.1），有些GBDT的实现加入了随机抽样（subsample 0.5&lt;=f &lt;=0.8）提高模型的泛化能力。通过交叉验证的方法选择最优的参数。因此GBDT实际的核心问题变成怎么基于![](https://upload.wikimedia.org/math/2/b/b/2bbe4b0725baed85eae8dbbb20360ea6.png "\{\(x\_i, r\_{im}\)\}\_{i=1}^n")使用CART回归树生成![](https://upload.wikimedia.org/math/7/f/d/7fd32efb7a21cc484be23d1015ee074e.png "\! h\_m\(x\)")？

　　CART分类树在很多书籍和资料中介绍比较多，但是再次强调GDBT中使用的是回归树。作为对比，先说分类树，我们知道CART是二叉树，CART分类树在每次分枝时，是穷举每一个feature的每一个阈值，根据GINI系数找到使不纯性降低最大的的feature以及其阀值，然后按照feature&lt;=阈值，和feature&gt;阈值分成的两个分枝，每个分支包含符合分支条件的样本。用同样方法继续分枝直到该分支下的所有样本都属于统一类别，或达到预设的终止条件，若最终叶子节点中的类别不唯一，则以多数人的类别作为该叶子节点的性别。回归树总体流程也是类似，不过在每个节点（不一定是叶子节点）都会得一个预测值，以年龄为例，该预测值等于属于这个节点的所有人年龄的平均值。分枝时穷举每一个feature的每个阈值找最好的分割点，但**衡量最好的标准不再是GINI系数，而是最小化均方差--即（每个人的年龄-预测年龄）^2 的总和 / N**，或者说是每个人的预测误差平方和 除以 N。这很好理解，被预测出错的人数越多，错的越离谱，均方差就越大，通过最小化均方差能够找到最靠谱的分枝依据。分枝直到每个叶子节点上人的年龄都唯一（这太难了）或者达到预设的终止条件（如叶子个数上限），若最终叶子节点上人的年龄不唯一，则以该节点上所有人的平均年龄做为该叶子节点的预测年龄。

## 3. Xgboost

　　Xgboost是GB算法的高效实现，xgboost中的基学习器除了可以是CART（gbtree）也可以是线性分类器（gblinear）。下面所有的内容来自原始paper，包括公式。

　　\(1\). xgboost在目标函数中显示的加上了正则化项，基学习为CART时，正则化项与树的叶子节点的数量T和叶子节点的值有关。

![](https://images2015.cnblogs.com/blog/754644/201605/754644-20160530163025555-653522936.jpg)

　　\(2\). GB中使用Loss Function对f\(x\)的一阶导数计算出伪残差用于学习生成fm\(x\)，xgboost不仅使用到了一阶导数，还使用二阶导数。

　　　　第t次的loss：

![](https://images2015.cnblogs.com/blog/754644/201605/754644-20160530164442602-1288079039.jpg)

　　　　对上式做二阶泰勒展开：g为一阶导数，h为二阶导数

![](https://images2015.cnblogs.com/blog/754644/201605/754644-20160530164744149-143494562.jpg)

　　\(3\). 上面提到CART回归树中寻找最佳分割点的衡量标准是最小化均方差，xgboost寻找分割点的标准是最大化，lamda，gama与正则化项相关

![](https://images2015.cnblogs.com/blog/754644/201605/754644-20160530170902758-1033686275.jpg)

 　　xgboost算法的步骤和GB基本相同，都是首先初始化为一个常数，gb是根据一阶导数ri，xgboost是根据一阶导数gi和二阶导数hi，迭代生成基学习器，相加更新学习器。

xgboost与gdbt除了上述三点的不同，xgboost在实现时还做了许多**优化**：

* 在寻找最佳分割点时，考虑传统的枚举每个特征的所有可能分割点的贪心法效率太低，xgboost实现了一种近似的算法。大致的思想是根据百分位法列举几个可能成为分割点的候选者，然后从候选者中根据上面求分割点的公式计算找出最佳的分割点。
* xgboost考虑了训练数据为稀疏值的情况，可以为缺失值或者指定的值指定分支的默认方向，这能大大提升算法的效率，paper提到50倍。
* 特征列排序后以块的形式存储在内存中，在迭代中可以重复使用；虽然boosting算法迭代必须串行，但是在处理每个特征列时可以做到并行。
* 按照特征列方式存储能优化寻找最佳的分割点，但是当以行计算梯度数据时会导致内存的不连续访问，严重时会导致cache miss，降低算法效率。paper中提到，可先将数据收集到线程内部的buffer，然后再计算，提高算法的效率。
* xgboost 还考虑了当数据量比较大，内存不够时怎么有效的使用磁盘，主要是结合多线程、数据压缩、分片的方法，尽可能的提高算法的效率。



