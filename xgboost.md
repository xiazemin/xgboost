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

![](/assets/gbdt8.png)从这里就可以看出xgboost的不同了，目标函数保留了泰勒展开的二次项。

![](/assets/gbdt9.png)（5）.Refine the definition of tree 重新定义每棵树

![](/assets/gbdt10.png)

（6）.Define the Complexity of Tree 树的复杂度项

![](/assets/import11.png)从图中可以看出，xgboost算法中对树的复杂度项包含了两个部分，一个是叶子节点总数，一个是叶子节点得分L2正则化项，针对每个叶结点的得分增加L2平滑，目的也是为了避免过拟合。

（7）.Revisit the Objectives 更新

![](/assets/gbdt12.png)注意，这里优化目标的n-&gt;T,T是叶子数。???

论文中定义了：Define I j = {i\|q\(x i \) = j} as the instance set of leaf j.这一步是由于xgb加了两个正则项，一个是叶子节点个数\(T\),一个是叶节点分数\(w\). 原文中的等式4，加了正则项的目标函数里就出现了两种累加，一种是i-&gt;n（样本数）,一种是j-&gt;T（叶子节点数）。这步转换是为了统一目标函数中的累加项，Ij是每个叶节点j上的样本集合。

（8）.The Structure Score 这个score是用来评价树结构的。根据目标函数得到（见论文公式\(4\)、\(5\)、\(6\)），用于切分点查找算法。

![](/assets/gbdt13.png)

![](/assets/import14.png)The Structure Score Calculation：

![](/assets/gbdt15.png)（9）切分点查找算法

![](/assets/gbdt16.png)如上图可见Gain还加了一个叶子节点复杂度项，有点类似CART的剪枝。

![](/assets/gbdt18.png)上图中G都是各自区域内的gi总和，根据Gain（max）选择最优分割点。此外，作者针对算法设计对特征进行了排序，分位点划分等，有兴趣的可以阅读原始论文，这里不做详解。

算法步骤：

![](/assets/gbdt19.png)

根据特征划分有无数可能的树结构，因此采用近似算法（特征分位点，候选分割点）

![](/assets/gbdt20.png)

（10）小结一下： Boosted Tree Algorithm

![](/assets/gbdt21.png)

二、xgboost特点（与gbdt对比）

说明一下：这部分内容参考了知乎上的一个问答—机器学习算法中GBDT和XGBOOST的区别有哪些？，答主是wepon大神，根据他的总结我自己做了一理解和补充。



1.传统GBDT以CART作为基分类器，xgboost还支持线性分类器，这个时候xgboost相当于带L1和L2正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题）。 —可以通过booster \[default=gbtree\]设置参数:gbtree: tree-based models/gblinear: linear models



2.传统GBDT在优化时只用到一阶导数信息，xgboost则对代价函数进行了二阶泰勒展开，同时用到了一阶和二阶导数。顺便提一下，xgboost工具支持自定义代价函数，只要函数可一阶和二阶求导。 —对损失函数做了改进（泰勒展开，一阶信息g和二阶信息h,上一章节有做介绍）



3.xgboost在代价函数里加入了正则项，用于控制模型的复杂度。正则项里包含了树的叶子节点个数、每个叶子节点上输出的score的L2模的平方和。从Bias-variance tradeoff角度来讲，正则项降低了模型variance，使学习出来的模型更加简单，防止过拟合，这也是xgboost优于传统GBDT的一个特性 

—正则化包括了两个部分，都是为了防止过拟合，剪枝是都有的，叶子结点输出L2平滑是新增的。



4.shrinkage and column subsampling —还是为了防止过拟合，论文2.3节有介绍，这里答主已概括的非常到位



（1）shrinkage缩减类似于学习速率，在每一步tree boosting之后增加了一个参数n（权重），通过这种方式来减小每棵树的影响力，给后面的树提供空间去优化模型。



（2）column subsampling列\(特征\)抽样，说是从随机森林那边学习来的，防止过拟合的效果比传统的行抽样还好（行抽样功能也有），并且有利于后面提到的并行化处理算法。



5.split finding algorithms\(划分点查找算法\)：—理解的还不够透彻，需要进一步学习 

（1）exact greedy algorithm—贪心算法获取最优切分点 

（2）approximate algorithm— 近似算法，提出了候选分割点概念，先通过直方图算法获得候选分割点的分布情况，然后根据候选分割点将连续的特征信息映射到不同的buckets中，并统计汇总信息。详细见论文3.3节 

（3）Weighted Quantile Sketch—分布式加权直方图算法，论文3.4节 

这里的算法（2）、（3）是为了解决数据无法一次载入内存或者在分布式情况下算法（1）效率低的问题，以下引用的还是wepon大神的总结：



可并行的近似直方图算法。树节点在进行分裂时，我们需要计算每个特征的每个分割点对应的增益，即用贪心法枚举所有可能的分割点。当数据无法一次载入内存或者在分布式情况下，贪心算法效率就会变得很低，所以xgboost还提出了一种可并行的近似直方图算法，用于高效地生成候选的分割点。



6.对缺失值的处理。对于特征的值有缺失的样本，xgboost可以自动学习出它的分裂方向。 —稀疏感知算法，论文3.4节，Algorithm 3: Sparsity-aware Split Finding



7.Built-in Cross-Validation（内置交叉验证\)



XGBoost allows user to run a cross-validation at each iteration of the boosting process and thus it is easy to get the exact optimum number of boosting iterations in a single run. 

This is unlike GBM where we have to run a grid-search and only a limited values can be tested.



8.continue on Existing Model（接着已有模型学习）



User can start training an XGBoost model from its last iteration of previous run. This can be of significant advantage in certain specific applications. 

GBM implementation of sklearn also has this feature so they are even on this point.



9.High Flexibility（高灵活性）



\*\*XGBoost allow users to define custom optimization objectives and evaluation criteria. 

This adds a whole new dimension to the model and there is no limit to what we can do.\*\*



10.并行化处理 —系统设计模块,块结构设计等



xgboost工具支持并行。boosting不是一种串行的结构吗?怎么并行的？注意xgboost的并行不是tree粒度的并行，xgboost也是一次迭代完才能进行下一次迭代的（第t次迭代的代价函数里包含了前面t-1次迭代的预测值）。xgboost的并行是在特征粒度上的。我们知道，决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点），xgboost在训练之前，预先对数据进行了排序，然后保存为block结构，后面的迭代中重复地使用这个结构，大大减小计算量。这个block结构也使得并行成为了可能，在进行节点的分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么各个特征的增益计算就可以开多线程进行。



此外xgboost还设计了高速缓存压缩感知算法，这是系统设计模块的效率提升。 

当梯度统计不适合于处理器高速缓存和高速缓存丢失时，会大大减慢切分点查找算法的速度。 

（1）针对 exact greedy algorithm采用缓存感知预取算法 

（2）针对 approximate algorithms选择合适的块大小



我觉得关于xgboost并行化设计仅仅从论文PPT博客上学习是远远不够的，有时间还要从代码层面去学习分布式 xgboost的设计理念。



三、xgboost参数详解

官方参数介绍看这里： 

Parameters \(official guide\)



General Parameters（常规参数） 

1.booster \[default=gbtree\]：选择基分类器，gbtree: tree-based models/gblinear: linear models 

2.silent \[default=0\]:设置成1则没有运行信息输出，最好是设置为0. 

3.nthread \[default to maximum number of threads available if not set\]：线程数



Booster Parameters（模型参数） 

1.eta \[default=0.3\]:shrinkage参数，用于更新叶子节点权重时，乘以该系数，避免步长过大。参数值越大，越可能无法收敛。把学习率 eta 设置的小一些，小学习率可以使得后面的学习更加仔细。 

2.min\_child\_weight \[default=1\]:这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言，假设 h 在 0.01 附近，min\_child\_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。 

3.max\_depth \[default=6\]: 每颗树的最大深度，树高越深，越容易过拟合。 

4.max\_leaf\_nodes:最大叶结点数，与max\_depth作用有点重合。 

5.gamma \[default=0\]：后剪枝时，用于控制是否后剪枝的参数。 

6.max\_delta\_step \[default=0\]：这个参数在更新步骤中起作用，如果取0表示没有约束，如果取正值则使得更新步骤更加保守。可以防止做太大的更新步子，使更新更加平缓。 

7.subsample \[default=1\]：样本随机采样，较低的值使得算法更加保守，防止过拟合，但是太小的值也会造成欠拟合。 

8.colsample\_bytree \[default=1\]：列采样，对每棵树的生成用的特征进行列采样.一般设置为： 0.5-1 

9.lambda \[default=1\]：控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。 

10.alpha \[default=0\]:控制模型复杂程度的权重值的 L1 正则项参数，参数值越大，模型越不容易过拟合。 

11.scale\_pos\_weight \[default=1\]：如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。



Learning Task Parameters（学习任务参数） 

1.objective \[default=reg:linear\]：定义最小化损失函数类型，常用参数： 

binary:logistic –logistic regression for binary classification, returns predicted probability \(not class\) 

multi:softmax –multiclass classification using the softmax objective, returns predicted class \(not probabilities\) 

you also need to set an additional num\_class \(number of classes\) parameter defining the number of unique classes 

multi:softprob –same as softmax, but returns predicted probability of each data point belonging to each class. 

2.eval\_metric \[ default according to objective \]： 

The metric to be used for validation data. 

The default values are rmse for regression and error for classification. 

Typical values are: 

rmse – root mean square error 

mae – mean absolute error 

logloss – negative log-likelihood 

error – Binary classification error rate \(0.5 threshold\) 

merror – Multiclass classification error rate 

mlogloss – Multiclass logloss 

auc: Area under the curve 

3.seed \[default=0\]： 

The random number seed. 随机种子，用于产生可复现的结果 

Can be used for generating reproducible results and also for parameter tuning.



注意: python sklearn style参数名会有所变化 

eta –&gt; learning\_rate 

lambda –&gt; reg\_lambda 

alpha –&gt; reg\_alpha

