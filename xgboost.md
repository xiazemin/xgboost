xgboost是大规模并行boosted tree的工具，它是目前最快最好的开源boosted tree工具包，比常见的工具包快10倍以上。在数据科学方面，有大量kaggle选手选用它进行数据挖掘比赛，其中包括两个以上kaggle比赛的夺冠方案。在工业界规模方面，xgboost的分布式版本有广泛的可移植性，支持在YARN, MPI, Sungrid Engine等各个平台上面运行，并且保留了单机并行版本的各种优化，使得它可以很好地解决于工业界规模的问题。

微软出了个LightGBM,号称性能更强劲，速度更快。简单实践了一波，发现收敛速度要快一些，不过调参还不6 ，没有权威。看了GitHub上的介绍以及知乎上的一些回答，大致理解了性能提升的原因。

主要是两个：①histogram算法替换了传统的Pre-Sorted，某种意义上是牺牲了精度（但是作者声明实验发现精度影响不大）换取速度，直方图作差构建叶子直方图挺有创造力的。（xgboost的分布式实现也是基于直方图的，利于并行）②带有深度限制的按叶子生长 \(leaf-wise\) 算法代替了传统的\(level-wise\) 决策树生长策略，提升精度，同时避免过拟合危险。

一、xgboost基本原理介绍

1.提升方法是一种非常有效的机器学习方法，在前几篇笔记中介绍了提升树与GBDT基本原理，xgboost（eXtreme Gradient Boosting）可以说是提升方法的完全加强版本。xgboost算法在各大比赛中展现了强大的威力。

2.Regression Tree and Ensemble \(What are we Learning，得到学习目标\)

（1）.Regression Tree \(CART\)回归树

![](/assets/gdbt1.png)（2）.Regression Tree Ensemble 回归树集成 

![](/assets/gdbt2.png)

