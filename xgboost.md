xgboost是大规模并行boosted tree的工具，它是目前最快最好的开源boosted tree工具包，比常见的工具包快10倍以上。在数据科学方面，有大量kaggle选手选用它进行数据挖掘比赛，其中包括两个以上kaggle比赛的夺冠方案。在工业界规模方面，xgboost的分布式版本有广泛的可移植性，支持在YARN, MPI, Sungrid Engine等各个平台上面运行，并且保留了单机并行版本的各种优化，使得它可以很好地解决于工业界规模的问题。

微软出了个LightGBM,号称性能更强劲，速度更快。简单实践了一波，发现收敛速度要快一些，不过调参还不6 ，没有权威。看了GitHub上的介绍以及知乎上的一些回答，大致理解了性能提升的原因。

主要是两个：①histogram算法替换了传统的Pre-Sorted，某种意义上是牺牲了精度（但是作者声明实验发现精度影响不大）换取速度，直方图作差构建叶子直方图挺有创造力的。（xgboost的分布式实现也是基于直方图的，利于并行）②带有深度限制的按叶子生长 \(leaf-wise\) 算法代替了传统的\(level-wise\) 决策树生长策略，提升精度，同时避免过拟合危险。

一、xgboost基本原理介绍

1.提升方法是一种非常有效的机器学习方法，在前几篇笔记中介绍了提升树与GBDT基本原理，xgboost（eXtreme Gradient Boosting）可以说是提升方法的完全加强版本。xgboost算法在各大比赛中展现了强大的威力。

2.Regression Tree and Ensemble \(What are we Learning，得到学习目标\)

（1）.Regression Tree \(CART\)回归树

![](/assets/gdbt1.png)（2）.Regression Tree Ensemble 回归树集成

![](/assets/gdbt2.png)在上面的例子中，我们用两棵树来进行预测。我们对于每个样本的预测结果就是每棵树预测分数的和。

（3）.Objective for Tree Ensemble 得到学习目标函数

![](/assets/gbdt3.png)

这里是构造一个目标函数，然后我们要做的就是去尝试优化这个目标函数。读到这里，感觉与gbdt好像没有什么区别，确实如此，不过在后面就能看到他们的不同了（构造（学习）模型参数）。

3.Gradient Boosting \(How do we Learn，如何学习\)

（1）.So How do we Learn?

目标函数

![](/assets/gbdt4.png)

![](/assets/gbdt5.png)这里理解很关键，这里目标函数优化的是整体的模型，yi’是整个累加模型的输出，正则化项是所有树的复杂度之和，这个复杂度组成后面（6）会讲。这种包含树的模型不适合直接用SGD等优化算法直接对整体模型进行优化，因而采用加法学习方法，boosting的学习策略是每次学习当前的树，找到当前最佳的树模型加入到整体模型中，因此关键在于学习第t棵树。

（2）.Additive Training ：定义目标函数，优化，寻找最佳的ft。

![](/assets/gbdt6.png)如图所示，第t轮的模型预测等于前t-1轮的模型预测y\(t-1\)加上ft，因此误差函数项记为l\(yi,y\(t-1\)+ft\),后面一项为正则化项。

在当前步，yi以及y\(t-1\)都是已知值，模型学习的是ft。

（3）.Taylor Expansion Approximation of Loss 对误差函数进行二阶泰勒近似展开

![](/assets/gbdt7.png)把平方损失函数的一二次项带入原目标函数，你会发现与之前那张ppt的损失函数是一致的 。

至于为什么要这样展开呢,这里就是xgboost的特点了，通过这种近似，你可以自定义一些损失函数（只要保证二阶可导），树分裂的打分函数是基于gi,hi（Gj，Hj）计算的。

（4）.Our New Goal 得到了新的目标函数

![](/assets/gbdt8.png)

