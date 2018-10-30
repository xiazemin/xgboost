[https://xgboost.readthedocs.io/en/latest/parameter.html\#general-parameters](https://xgboost.readthedocs.io/en/latest/parameter.html#general-parameters)

[https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)

[https://xgboost.readthedocs.io/en/latest/python/python\_api.html\#module-xgboost.sklearn](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn)

https://cran.r-project.org/web/packages/xgboost/index.html

XGBoost风靡Kaggle、天池、DataCastle、Kesci等国内外数据竞赛平台，是比赛夺冠的必备大杀器。

传统GBDT以CART作为基分类器，xgboost还支持线性分类器，这个时候xgboost相当于带L1和L2正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题）。

传统GBDT在优化时只用到一阶导数信息，xgboost则对代价函数进行了二阶泰勒展开，同时用到了一阶和二阶导数。顺便提一下，xgboost工具支持自定义代价函数，只要函数可一阶和二阶求导。

xgboost在代价函数里加入了正则项，用于控制模型的复杂度。正则项里包含了树的叶子节点个数、每个叶子节点上输出的score的L2模的平方和。从Bias-variance tradeoff角度来讲，正则项降低了模型的variance，使学习出来的模型更加简单，防止过拟合，这也是xgboost优于传统GBDT的一个特性。

Shrinkage（缩减），相当于学习速率（xgboost中的eta）。xgboost在进行完一次迭代后，会将叶子节点的权重乘上该系数，主要是为了削弱每棵树的影响，让后面有更大的学习空间。实际应用中，一般把eta设置得小一点，然后迭代次数设置得大一点。（补充：传统GBDT的实现也有学习速率）

列抽样（column subsampling）。xgboost借鉴了随机森林的做法，支持列抽样，不仅能降低过拟合，还能减少计算，这也是xgboost异于传统gbdt的一个特性。

对缺失值的处理。对于特征的值有缺失的样本，xgboost可以自动学习出它的分裂方向。

xgboost工具支持并行。boosting不是一种串行的结构吗?怎么并行的？注意xgboost的并行不是tree粒度的并行，xgboost也是一次迭代完才能进行下一次迭代的（第t次迭代的代价函数里包含了前面t-1次迭代的预测值）。xgboost的并行是在特征粒度上的。我们知道，决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点），xgboost在训练之前，预先对数据进行了排序，然后保存为block结构，后面的迭代中重复地使用这个结构，大大减小计算量。这个block结构也使得并行成为了可能，在进行节点的分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么各个特征的增益计算就可以开多线程进行。

可并行的近似直方图算法。树节点在进行分裂时，我们需要计算每个特征的每个分割点对应的增益，即用贪心法枚举所有可能的分割点。当数据无法一次载入内存或者在分布式情况下，贪心算法效率就会变得很低，所以xgboost还提出了一种可并行的近似直方图算法，用于高效地生成候选的分割点。

，randomForest, gbm和glmnet是三个尤其流行的R包，它们在Kaggle的各大数据挖掘竞赛中的出现频率独占鳌头，被坊间人称为R数据挖掘包中的三驾马车。根据我的个人经验，gbm包比同样是使用树模型的randomForest包占用的内存更少，同时训练速度较快，尤其受到大家的喜爱。在python的机器学习库sklearn里也有GradientBoostingClassifier的存在。

Boosting分类器属于集成学习模型，它基本思想是把成百上千个分类准确率较低的树模型组合起来，成为一个准确率很高的模型。这个模型会不断地迭代，每次迭代就生成一颗新的树。对于如何在每一步生成合理的树，大家提出了很多的方法，我们这里简要介绍由Friedman提出的Gradient Boosting Machine。它在生成每一棵树的时候采用梯度下降的思想，以之前生成的所有树为基础，向着最小化给定目标函数的方向多走一步。在合理的参数设置下，我们往往要生成一定数量的树才能达到令人满意的准确率。在数据集较大较复杂的时候，我们可能需要几千次迭代运算，如果生成一个树模型需要几秒钟，那么这么多迭代的运算耗时，应该能让你专心地想静静……

。xgboost的全称是eXtreme Gradient Boosting。正如其名，它是Gradient Boosting Machine的一个c++实现，作者为正在华盛顿大学研究机器学习的大牛陈天奇。他在研究中深感自己受制于现有库的计算速度和精度，因此在一年前开始着手搭建xgboost项目，并在去年夏天逐渐成型。xgboost最大的特点在于，它能够自动利用CPU的多线程进行并行，同时在算法上加以改进提高了精度。它的处女秀是Kaggle的希格斯子信号识别竞赛，因为出众的效率与较高的预测准确度在比赛论坛中引起了参赛选手的广泛关注，在1700多支队伍的激烈竞争中占有一席之地。随着它在Kaggle社区知名度的提高，最近也有队伍借助xgboost在比赛中夺得第一。

为了方便大家使用，陈天奇将xgboost封装成了python库。

