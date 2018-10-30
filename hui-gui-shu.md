一、原理

决策树是一种非参数的监督学习方法，它主要用于分类和回归。决策树的目的是构造一种模型，使之能够从样本数据的特征属性中，通过学习简单的决策规则——IF THEN规则，从而预测目标变量的值。

![](https://img-blog.csdn.net/20160112131643628)  




图1 决策树

例如，在某医院内，对因心脏病发作而入院治疗的患者，在住院的前24小时内，观测记录下来他们的19个特征属性——血压、年龄、以及其他17项可以综合判断病人状况的重要指标，用图1所示的决策树判断病人是否属于高危患者。在图1中，圆形为中间节点，也就是树的分支，它代表IF THEN规则的条件；方形为终端节点（叶节点），也就是树的叶，它代表IF THEN规则的结果。我们也把第一个节点称为根节点。

决策树往往采用的是自上而下的设计方法，每迭代循环一次，就会选择一个特征属性进行分叉，直到不能再分叉为止。因此在构建决策树的过程中，选择最佳（既能够快速分类，又能使决策树的深度小）的分叉特征属性是关键所在。这种“最佳性”可以用非纯度（impurity）进行衡量。如果一个数据集合中只有一种分类结果，则该集合最纯，即一致性好；反之有许多分类，则不纯，即一致性不好。有许多指标可以定量的度量这种非纯度，最常用的有熵，基尼指数（Gini Index）和分类误差，它们的公式分别为：



![](https://img-blog.csdn.net/20160112131832033)（1）



![](https://img-blog.csdn.net/20160112131937162)（2）



![](https://img-blog.csdn.net/20160112132040826)（3）

上述所有公式中，值越大，表示越不纯，这三个度量之间并不存在显著的差别。式中_D_表示样本数据的分类集合，并且该集合共有_J_种分类，_pj_表示第_j_种分类的样本率：



![](https://img-blog.csdn.net/20160112132147226)（4）

式中_N_和_Nj_分别表示集合_D_中样本数据的总数和第_j_个分类的样本数量。把式4带入式2中，得到：



![](https://img-blog.csdn.net/20160112132238074)（5）

目前常用的决策树的算法包括ID3（Iterative Dichotomiser 3，第3代迭戈二叉树）、C4.5和CART（ClassificationAnd Regression Tree，分类和回归树）。前两种算法主要应用的是基于熵的方法，而第三种应用的是基尼指数的方法。下面我们就逐一介绍这些方法。

### ID3

ID3是由Ross Quinlan首先提出，它是基于所谓“Occam'srazor”（奥卡姆剃刀），即越简单越好，也就是越是小型的决策树越优于大型的决策树。如前所述，我们已经有了熵作为衡量样本集合纯度的标准，熵越大，越不纯，因此我们希望在分类以后能够降低熵的大小，使之变纯一些。这种分类后熵变小的判定标准可以用信息增益（Information Gain）来衡量，它的定义为：



![](https://img-blog.csdn.net/20160112132340737)（6）

该式表示在样本集合_D_下特征属性_A_的信息增益，_n_表示针对特征属性_A_，样本集合被划分为_n_个不同部分，即_A_中包含着_n_个不同的值，_Ni_表示第_i_个部分的样本数量，E\(_Di_\)表示特征属性_A_下第_i_个部分的分类集合的熵。信息增益越大，分类后熵下降得越快，则分类效果越好。因此我们在_D_内遍历所有属性，选择信息增益最大的那个特征属性进行分类。在下次迭代循环中，我们只需对上次分类剩下的样本集合计算信息增益，如此循环，直至不能再分类为止。

### C4.5

C4.5算法也是由Quinlan提出，它是ID3算法的扩展。ID3应用的是信息增益的方法，但这种方法存在一个问题，那就是它会更愿意选择那些包括很多种类的特征属性，即哪个_A_中的_n_多，那么这个_A_的信息增益就可能更大。为此，C4.5使用信息增益率这一准则来衡量非纯度，即：



![](https://img-blog.csdn.net/20160112132437914)（7）

式中，SI\(_D_,_A_\)表示分裂信息值，它的定义为\(实际就分类熵\)：



![](https://img-blog.csdn.net/20160112132523769)（8）

该式中的符号含义与式6相同。同样的，我们选择信息增益率最大的那个特征属性作为分类属性。

### CART

CART算法是由Breiman等人首先提出，它包括分类树和回归树两种。我们先来讨论分类树，针对特征属性_A_，分类后的基尼指数为：



![](https://img-blog.csdn.net/20160112132631547)（9）

该式中的符号含义与式6相同。与ID3和C4.5不同，我们选择分类基尼指数最小的那个特征属性作为分类属性。当我们每次只想把样本集合分为两类时，即每个中间节点只产生两个分支，但如果特征属性_A_中有多于2个的值，即_n_&gt; 2，这时我们就需要一个阈值_β_，它把_D_分割成了_D_1和_D_2两个部分，不同的_β_得到不同的_D_1和_D_2，我们重新设_D_1的样本数为_L_，_D_2的样本数为_R_，因此有_L_+_R_=_N_，则式9可简写为：



![](https://img-blog.csdn.net/20160112132723140)（10）

我们把式5带入上式中，得到：







![](https://img-blog.csdn.net/20160112132825026)（11）

式中，∑_Lj_=_L_，∑_Rj_=_R_。式11只是通过不同特征属性_A_的不同阈值_β_来得到样本集_D_的不纯度，由于_D_内的样本数量_N_是一定的，因此对式11求最小值问题就转换为求式12的最大值问题：



![](https://img-blog.csdn.net/20160112132915806)（12）

以上给出的是分类树的计算方法，下面介绍回归树。两者的不同之处是，分类树的样本输出（即响应值）是类的形式，如判断蘑菇是有毒还是无毒，周末去看电影还是不去。而回归树的样本输出是数值的形式，比如给某人发放房屋贷款的数额就是具体的数值，可以是0到120万元之间的任意值。为了得到回归树，我们就需要把适合分类的非纯度度量用适合回归的非纯度度量取代。因此我们将熵计算用均方误差替代：



![](https://img-blog.csdn.net/20160112133010898)（13）

式中_N_表示_D_集合的样本数量，_yi_和_ri_分别为第_i_个样本的输出值和预测值。如果我们把样本的预测值用样本输出值的平均来替代，则式13改写为：



![](https://img-blog.csdn.net/20160112133116130)（14）

上式表示了集合_D_的最小均方误差，如果针对于某种特征属性_A_，我们把集合_D_划分为_s_个部分，则划分后的均方误差为：



![](https://img-blog.csdn.net/20160112133222074)（15）

式中_Ni_表示被划分的第_i_个集合_Di_的样本数量。式15与式14的差值为划分为_s_个部分后的误差减小：



![](https://img-blog.csdn.net/20160112133302275)（16）

与式6所表示的信息增益相似，我们寻求的是最大化的误差减小，此时就得到了最佳的_s_个部分的划分。

同样的，当我们仅考虑二叉树的情况时，即每个中间节点只有两个分支，此时_s_= 2，基于特征属性_A_的值，集合_D_被阈值_β_划分为_D_1和_D_2两个集合，每个集合的样本数分别为_L_和_R_，则：



![](https://img-blog.csdn.net/20160112133342929)（17）

把式14带入上式，得：



![](https://img-blog.csdn.net/20160112133447383)（18）

式中，_yi_是属于集合_D_的样本响应值，_li_和_ri_分别是属于集合_D_1和_D_2的样本响应值。对于某个节点来说，它的样本数量以及样本响应值的和是一个定值，因此式18的结果完全取决于方括号内的部分，即：



![](https://img-blog.csdn.net/20160112133537049)（19）

因此求式18的最大值问题就转变为求式19的最大值问题。

我们按照样本响应值是类的形式，还是数值的形式，把决策树分成了分类树和回归树，它们对应不同的计算公式。那么表示特征属性的形式也会有这两种形式，即类的形式和数值的形式，比如决定是否出去踢球，取决于两个条件：风力和气温。风力的表示形式是：无风、小风、中风、大风，气温的表示形式就是具体的摄氏度，如-10℃～40℃之间。风力这个特征属性就是类的形式，而气温就是数值的形式。又比如决定发放房屋贷款，其金额取决于两个条件：是否有车有房和年薪。有车有房的表示形式是：无车无房、有车无房、无车有房、有车有房，而年薪的表示形式就是具体的钱数，如0～20万。有车有房这个特征属性就是类的形式，年薪就是数值的形式。因此在分析样本的特征属性时，我们要把决策树分为四种情况：特征为类的分类树（如决定是否踢球的风力）、特征为数值的分类树（如决定是否踢球的温度）、特征为类的回归树（如发放贷款的有车有房）和特征为数值的回归树（如发放贷款的年薪）。由于特征形式不同，所以计算方法上有所不同：

Ⅰ、特征为类的分类树：对于两类问题，即样本的分类（响应值）只有两种情况：响应值为0和1，按照特征属性的类别的样本响应值为1的数量的多少进行排序。例如我们采集20个样本来构建是否踢球分类树，设出去踢球的响应值为1，不踢球的响应值为0，针对风力这个特征属性，响应值为1的样本有14个，无风有6个样本，小风有5个，中风2个，大风1个，则排序的结果为：大风&lt;中风&lt;小风&lt;无风。然后我们按照这个顺序依次按照二叉树的分叉方式把样本分为左分支和右分支，并带入式12求使该式为最大值的那个分叉方式，即先把是大风的样本放入左分支，其余的放入右分支，带入式12，得到A，再把大风和中风放入左分支，其余的放入右分支，带入式12，得到B，再把大风、中风和小风放入左分支，无风的放入右分支，计算得到C，比较A、B、C，如果最大值为C，则按照C的分叉方式划分左右分支，其中阈值_β_可以设为3。对于非两类问题，采用的是聚类的方法。

Ⅱ、特征为数值的分类树：由于特征属性是用数值进行表示，我们就按照数值的大小顺序依次带入式12，计算最大值。如一共有14个样本，按照由小至大的顺序为：abcdefghijklmn，第一次分叉为：a\|bcdefghijklmn，竖线“\|”的左侧被划分到左分支，右侧被划分到右分支，带入式12计算其值，然后第二次分叉：ab\|cdefghijklmn，同理带入式12计算其值，以此类推，得到这13次分叉的最大值，该种分叉方式即为最佳的分叉方式，其中阈值_β_为分叉的次数。

Ⅲ、特征为类的回归树：计算每种特征属性各个种类的平均样本响应值，按照该值的大小进行排序，然后依次带入式19，计算其最大值。

Ⅳ、特征为数值的回归树：该种情况与特征为数值的分类树相同，就按照数值的大小顺序依次带入式19，计算最大值。

在训练决策树时，还有三个技术问题需要解决。第一个问题是，对于分类树，我们还需要考虑一种情况，当用户想要检测一些非常罕见的异常现象的时候，这是非常难办到的，这是因为训练可能包含了比异常多得多的正常情况，那么很可能分类结果就是认为每一个情况都是正常的。为了避免这种情况的出现，我们需要设置先验概率，这样异常情况发生的概率就被人为的增加（可以增加到0.5甚至更高），这样被误分类的异常情况的权重就会变大，决策树也能够得到适当的调整。先验概率需要根据各自情况人为设置，但还需要考虑各个分类的样本率，因此这个先验值还不能直接应用，还需要处理。设_Qj_为我们设置的第_j_个分类的先验概率，_Nj_为该分类的样本数，则考虑了样本率并进行归一化处理的先验概率_qj_为：



![](https://img-blog.csdn.net/20160112133634849)（20）

把先验概率带入式12中，则得到：



![](https://img-blog.csdn.net/20160112133717446)（21）

第二个需要解决的问题是，某些样本缺失了某个特征属性，但该特征属性又是最佳分叉属性，那么如何对该样本进行分叉呢？目前有几种方法可以解决该问题，一种是直接把该样本删除掉；另一种方法是用各种算法估计该样本的缺失属性值。还有一种方法就是用另一个特征属性来替代最佳分叉属性，该特征属性被称为替代分叉属性。因此在计算最佳分叉属性的同时，还要计算该特征属性的替代分叉属性，以防止最佳分叉属性缺失的情况。CART算法就是采用的该方法，下面我们就来介绍该方法。

寻找替代分叉属性总的原则就是使其分叉的效果与最佳分叉属性相似，即分叉的误差最小。我们仍然根据特征属性是类还是数值的形式，也把替代分叉属性的计算分为两种情况。

当特征属性是类的形式的时候，当最佳分叉属性不是该特征属性时，会把该特征属性的每个种类分叉为不同的分支，例如当最佳分叉属性不是风力时，极有可能把5个无风的样本分叉为不同的分支（如3个属于左分支，2个属于右分支），但当最佳分叉属性是风力时，这种情况就不会发生，也就是5个无风的样本要么属于左分支，要么属于右分支。因此我们把被最佳分叉属性分叉的特征属性种类的分支最大样本数量作为该种类的分叉值，计算该特征属性所有种类的这些分叉值，最终这些分叉值之和就作为该替代分叉属性的分叉值。我们还看前面的例子，无风的分叉值为3，再计算小风、中风、大风的分叉值，假如它们的值分别为4、4、3，则把风力作为替代分叉属性的分叉值为14。按照该方法再计算其他特征属性是类形式的替代分叉值，则替代性由替代分叉值按从大到小进行排序。在用替代分叉属性分叉时那些左分支大于右分支样本数的种类被分叉为左分支，反之为右分支，如上面的例子，无风的样本被分叉为左分支。

当特征属性是数值的形式的时候，样本被分割成了四个部分：LL、LR、RL和RR，前一个字母表示被最佳分叉属性分叉为左右分支，后一个字母表示被替代分叉属性分叉为左右分支，如LR表示被最佳分叉属性分叉为左分支，但被替代分叉属性分叉为右分支的样本，因此LL和RR表示的是被替代分叉属性分叉正确的样本，而LR和RL是被替代分叉属性分叉错误的样本，在该特征属性下，选取阈值对样本进行分割，使LL+RR或LR+RL达到最大值，则最终max{LL+RR，LR+RL}作为该特征属性的替代分叉属性的分叉值。按照该方法再计算其他特征属性是数值形式的替代分叉值，则替代性也由替代分叉值按从大到小进行排序。最终我们选取替代分叉值最大的那个特征属性作为该最佳分叉属性的替代分叉属性。

为了让替代分叉属性与最佳分叉属性相比较，我们还需要对替代分叉值进行规范化处理，如果替代分叉属性是类的形式，则替代分叉值需要乘以式12再除以最佳分叉属性中的种类数量，如果替代分叉属性是数值的形式，则替代分叉值需要乘以式19再除以所有样本的数量。规范化后的替代分叉属性如果就是最佳分叉属性时，两者的值是相等的。

第三个问题就是过拟合。由于决策树的建立完全是依赖于训练样本，因此该决策树对该样本能够产生完全一致的拟合效果。但这样的决策树对于预测样本来说过于复杂，对预测样本的分类效果也不够精确。这种现象就称为过拟合。

将复杂的决策树进行简化的过程称为剪枝，它的目的是去掉一些节点，包括叶节点和中间节点。剪枝常用的方法有预剪枝和后剪枝两种。预剪枝是在构建决策树的过程中，提前终止决策树的生长，从而避免过多的节点的产生。该方法虽然简单但实用性不强，因为我们很难精确的判断何时终止树的生长。后剪枝就是在决策树构建完后再去掉一些节点。常见后剪枝方法有四种：悲观错误剪枝（PEP）、最小错误剪枝（MEP）、代价复杂度剪枝（CCP）和基于错误的剪枝（EBP）。CCP算法能够应用于CART算法中，它的本质是度量每减少一个叶节点所得到的平均错误，在这里我们重点介绍CCP算法。

CCP算法会产生一系列树的序列{T0,T1,…,T_m_}，其中T0是由训练得到的最初的决策树，而T_m_只含有一个根节点。序列中的树是嵌套的，也就是序列中的T_i_+1是由T_i_通过剪枝得到的，即实现用T_i_+1中一个叶节点来替代T_i_中以该节点为根的子树。这种被替代的原则就是使误差的增加率_α_最小，即



![](https://img-blog.csdn.net/20160112133814466)（22）

式中，_R_\(_n_\)表示T_i_中节点_n_的预测误差，_R_\(_nt_\)表示T_i_中以节点_n_为根节点的子树的所有叶节点的预测误差之和，\|_nt_\|为该子树叶节点的数量，\|_nt_\|也被称为复杂度，因为叶节点越多，复杂性当然就越强。因此_α_的含义就是用一个节点_n_来替代以_n_为根节点的所有\|_nt_\|个节点的误差增加的规范化程度。在T_i_中，我们选择最小的_α_值的节点进行替代，最终得到T_i_+1。以此类推，每需要得到一棵决策树，都需要计算其前一棵决策树的_α_值，根据_α_值来对前一棵决策树进行剪枝，直到最终剪枝到只剩下含有一个根节点的T_m_为止。

根据决策树是分类树还是回归树，节点的预测误差的计算也分为两种情况。在分类树下，我们可以应用上面介绍过的式1～式3中的任意一个，如果我们应用式3来表示节点_n_的预测误差，则：



![](https://img-blog.csdn.net/20160112133857244)（23）

式中，_Nj_表示节点_n_下第_j_个分类的样本数，_N_为该节点的所有样本数，max{_Nj_}表示在_m_个分类中，拥有样本数最多的那个分类的样本数量。在回归树下，我们可以应用式14来表示节点_n_的预测误差：



![](https://img-blog.csdn.net/20160112134409562)（24）

式中，_yi_表示第_i_个样本的响应值，_N_为该节点的样本数量。我们把式23和式24的分子部分称为节点的风险值。

我们用全部样本得到的决策树序列为{T0,T1,…,T_m_}，其所对应的_α_值为_α_0&lt;_α_1&lt;…&lt;_αm_。下一步就是如何从这个序列中最优的选择一颗决策树T_i_。而与其说找到最优的T_i_，不如说找到其所对应的_αi_。这一步骤通常采用的方法是交叉验证（Cross-Validation）。

我们把_L_个样本随机划分为数量相等的_V_个子集_Lv_，_v_=1,…,_V_。第_v_个训练样本集为：



![](https://img-blog.csdn.net/20160112134509345)（25）

则_Lv_被用来做_L_\(_v_\)的测试样本集。对每个训练样本集_L_\(_v_\)按照CCP算法得到决策树的序列{T0\(_v_\),T1\(_v_\),…,T_m_\(_v_\)}，其对应的_α_值为_α_0\(_v_\)&lt;_α_1\(_v_\)&lt;…&lt;_αm_\(_v_\)。_α_值的计算仍然采用式22。对于分类树来说，第_v_个子集的节点_n_的预测误差为：



![](https://img-blog.csdn.net/20160112134601767)（26）

式中，_Nj_\(_v_\)表示训练样本集_L_\(_v_\)中节点_n_的第_j_个分类的样本数，_N_\(_v_\)为_L_\(_v_\)中节点_n_的所有样本数，max{_Nj_\(_v_\)}表示在_m_个分类中，_L_\(_v_\)中节点_n_拥有样本数最多的那个分类的样本数量。对于回归树来说，第_v_个子集的节点_n_的预测误差为：



![](https://img-blog.csdn.net/20160112134643526)（27）

式中，_yj_\(_v_\)表示训练样本集_L_\(_v_\)中节点_n_的第_i_个样本的响应值。我们仍然把式26和式27的分子部分称交叉验证子集中的节点风险值。

我们由训练样本集得到了树序列，然后应用这些树对测试样本集进行测试，测试的结果用错误率来衡量，即被错误分类的样本数量。对于分类树来说，节点_n_的错误率为：



![](https://img-blog.csdn.net/20160112134721820)（28）

式中，_Nv_表示测试样本集_Lv_中节点_n_的所有样本数，_Nv,j_表示_Lv_中第_j_个分类的样本数，这个_j_是式26中max{\|_Lj_\(_v_\)\|}所对应的_j_。对于回归树来说，节点_n_的错误率为：



![](https://img-blog.csdn.net/20160112134806297)（29）

式中，_yv_,_i_表示_Lv_的第_i_个样本响应值。决策树的总错误率_E_\(_v_\)等于该树所有叶节点的错误率之和。

虽然交叉验证子集决策树序列T\(_v_\)的数量要与由全体样本得到的决策树序列T的数量相同，但两者构建的形式不同，它需要比较两者的_α_值后再来构建。而为了比较由全体样本训练得到_α_值与交叉验证子集的_α_\(_v_\)值之间的大小，我们还需要对_α_值进行处理，即



![](https://img-blog.csdn.net/20160112134855328)（30）

其中_α’_0= 0，而_α’m_为无穷大。

我们设按照式22得到的初始训练子集_L_\(_v_\)决策树序列为{T0\(_v_\),T1\(_v_\),…,T_m_\(_v_\)}，其所对应的_α_\(_v_\)值为{_α_0\(_v_\),_α_1\(_v_\),…,_αm_\(_v_\)}。最终的树序列也是由这些T\(_v_\)组成，并且也是嵌套的形式，但可以重复，而且必须满足：



![](https://img-blog.csdn.net/20160112134936916)（31）

该式的含义是T\(_v_\)中第_k_个子树的_α_\(_v_\)值要小于_α’k_的最大的_α_\(_v_\)所对应的子树，因此最终的树序列有可能是这种形式：T0\(_v_\),T0\(_v_\),T1\(_v_\),T1\(_v_\),T2\(_v_\),T2\(_v_\),T2\(_v_\),T2\(_v_\),…，直到序列中树的数量为_m_为止。

子集的决策树序列构建好了，下面我们就可以计算_V_个子集中树序列相同的总错误率之和，即



![](https://img-blog.csdn.net/20160112135014019)（32）

则最佳的子树索引_J_为：



![](https://img-blog.csdn.net/20160112135057209)（33）

最终我们选择决策树序列{T0,T1,…,T_m_}中第_J_棵树为最佳决策树，该树的总错误率最小。

如果我们在选择决策树时使用1-SE（1 Standard Error of Minimum Error）规则的话，那么有可能最终的决策树不是错误率最小的子树，而是错误率略大，但树的结构更简单的那颗决策树。我们首先计算误差范围_SE_：



![](https://img-blog.csdn.net/20160112135136913)（34）

式中，_EJ_表示最佳子树的总错误率，_N_为总的样本数。则最终被选中的决策树的总错误率_EK_要满足：



![](https://img-blog.csdn.net/20160112135210812)（35）

并且决策树的结构最简单。
