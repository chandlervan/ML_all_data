# 浅谈Learning to Rank中的RankNet和LambdaRank算法

排序学习（Learning to Rank, LTR）是搜索算法中的重要一环，本文将对其中非常具有代表性的RankNet和LambdaRank算法进行研究。

## 搜索过程与LTR方法简介

本节将对搜索过程和LTR方法简单介绍，对这部分很熟悉的读者可直接跳过此节。

搜索这一过程的本质是自动选取与用户输入的关键词（query）最相关的一组文档（docs，或称网页, urls）的过程，如图1所示。目前主要通过如下两个步骤实现：

① query-doc匹配：寻找与当前输入的query相关度高的docs；

② 高相关度docs精确排序：对①中返回的docs，选取更多特征并按照用户点击该doc的可能性大小精确排序，如图2所示。有时我们还会选择不同的特征，召回多组①并将它们通过排序算法融合为一组。

![img](https://pic4.zhimg.com/80/v2-dc092bc421c0c7755158e5885f80ea4f_720w.jpg)图1 搜索过程示意图

![img](https://pic4.zhimg.com/80/v2-a8ec42beb590089c5f838c056497122b_720w.jpg)图2 对匹配阶段召回的N个候选文档根据相关性、重要程度和偏好等因素进行排序

Learning to  Rank就是一类目前最常用的，通过机器学习实现步骤②的算法。它主要包含单文档方法（pointwise）、文档对方法（pairwise）和文档列表（listwise）三种类型。pointwise单文档方法顾名思义：对于某一个query，它将每个doc分别判断与这个query的相关程度，由此将docs排序问题转化为了分类（比如相关、不相关）或回归问题（相关程度越大，回归函数的值越大）。但是pointwise方法只将query与单个doc建模，建模时未将其他docs作为特征进行学习，也就无法考虑到不同docs之间的顺序关系。而排序学习的目的主要是对搜索结果中的docs根据用户点击的可能性概率大小进行排序，所以pointwise势必存在一些缺陷。

针对pointwise存在的问题，pairwise文档对方法并不关心某一个doc与query相关程度的具体数值，而是将排序问题转化为任意两个不同docs ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7Bd_i%7D%5C%5D)和![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7Bd_j%7D%5C%5D)谁与当前query更相关的相对顺序的排序问题，一般分为 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7Bd_i%7D%5C%5D)比![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7Bd_j%7D%5C%5D) 更相关、更不相关和相关程度相等三个类别，分别记为{+1, -1, 0}，由此便又转化为了分类问题。本文重点关注的RankNet和LambdaRank算法同属于pairwise方法，是很多排序算法的雏形。

而listwise则是将一个query对应的所有相关文档看作一个整体，作为单个训练样本。

## RankNet算法基础及其训练加速

RankNet和LambdaRank同属于pairwise方法。对于某一个query，pairwise方法并不关心某个doc与这个query的相关程度的具体数值，而是将对所有docs的排序问题转化为求解任意两个docs的先后问题，即：根据docs与query的相关程度，比较任意两个不同文档 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5Bdo%7Bc_i%7D%5C%5D)和![[公式]](https://www.zhihu.com/equation?tex=%5C%5Bdo%7Bc_j%7D%5C%5D)的相对位置关系，并将query更相关的doc排在前面。一般使用 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5Bdo%7Bc_i%7D%5C%5D) 比 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5Bdo%7Bc_j%7D%5C%5D) 更相关、![[公式]](https://www.zhihu.com/equation?tex=%5C%5Bdo%7Bc_j%7D%5C%5D) 比 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5Bdo%7Bc_i%7D%5C%5D) 更相关和相关程度相等三个类别，并分别使用{+1, -1, 0}作为对应的类别标签，然后使用文档对![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%5Cleft%5Clangle+%7Bdo%7Bc_i%7D%2Cdo%7Bc_j%7D%7D+%5Cright%5Crangle+%5C%5D)作为样本的输入特征，由此将排序问题转化为了分类问题。这样做的另一个好处是，无需对每个doc与query的相关性进行精确标注（实际大规模数据应用场景下很难获得），只需获得docs之间的相对相关性，相对容易获得，可通过搜索日志、点击率数据等方式获得。

RankNet就是基于这样的思想的一种排序算法。若用 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7Bx_i%7D%5C%5D) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7Bx_j%7D%5C%5D) 来表示文档 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5Bdo%7Bc_i%7D%5C%5D) 和![[公式]](https://www.zhihu.com/equation?tex=%5C%5Bdo%7Bc_j%7D%5C%5D)的特征，![[公式]](https://www.zhihu.com/equation?tex=%5C%5Bs+%3D+f%28x%3Bw%29%5C%5D) 代表某种打分函数，*x*和*w*分别代表输入特征和参数。那么文档 *i* 和 *j* 的得分为

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D%7Bs_i%7D+%3D+f%28%7Bx_i%7D%3Bw%29+++++++++++++++%281%29%5Cend%7Bequation%7D%5C%5C) ![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D%7Bs_j%7D+%3D+f%28%7Bx_j%7D%3Bw%29++++++++%282%29%5Cend%7Bequation%7D%5C%5C) 由于pairwise方法只考虑 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5Bdo%7Bc_i%7D%5C%5D) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5Bdo%7Bc_j%7D%5C%5D) 的相对顺序，RankNet巧妙的借用了sigmoid函数来定义 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5Bdo%7Bc_i%7D%5C%5D) 比 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5Bdo%7Bc_j%7D%5C%5D) 更相关的概率为

![[公式]](https://www.zhihu.com/equation?tex=%7BP_%7Bij%7D%7D+%3D+P%28do%7Bc_i%7D+%5Ctriangleright+do%7Bc_j%7D%29+%3D+%5Cfrac%7B1%7D%7B%7B1+%2B+%7Be%5E%7B+-+%5Csigma+%28%7Bs_i%7D+-+%7Bs_j%7D%29%7D%7D%7D%7D%5C+++%283%29%5C%5C) 

其中 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%5Csigma+%5C%5D) 为待学习的参数（其实就是逻辑斯蒂回归）。若 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5Bdo%7Bc_i%7D%5C%5D) 比 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5Bdo%7Bc_j%7D%5C%5D) 更相关，则 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7BP_%7Bij%7D%7D+%3E+0.5%5C%5D) ，反之 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7BP_%7Bij%7D%7D+%3C+0.5%5C%5D) 。由此便借用了sigmoid函数将 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5Bdo%7Bc_i%7D%5C%5D) 比 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5Bdo%7Bc_j%7D%5C%5D) 更相关的概率  映射至[0, 1]上的实数，并从概率的角度对“谁更相关”这个问题进行了建模，也让我们得以使用分类问题的思想对两个文档的相对顺序问题进行求解。

在前文提到过pairwise的类别标签为{+1, -1, 0}，在RankNet中我们继续沿用这套标签并将其记为 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7BS_%7Bij%7D%7D+%5Cin+%5C%7B++%2B+1%2C+-+1%2C0%5C%7D+%5C%5D) 。由于接下来要使用交叉熵作为损失函数，因此将标签 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7BS_%7Bij%7D%7D%5C%5D) 与真实概率 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7B%5Cbar+P_%7Bij%7D%7D%5C%5D) （真实相关性）进行一一映射，有

![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cbar+P_%7Bij%7D%7D+%3D+%5Cfrac%7B1%7D%7B2%7D%5Cleft%28+%7B1+%2B+%7BS_%7Bij%7D%7D%7D+%5Cright%29%5C++%284%29%5C%5C) 

使用交叉熵函数作为损失函数，单个样本的交叉熵损失函数（loss）为

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%2A%7D+%7BC_%7Bij%7D%7D+%26%3D++-+%5Csum%5Climits_%7Bi+%3D+1%7D%5En+%7B%7B%7B%5Cbar+y%7D_i%7D%5Clog+%7By_i%7D%7D+%5C%5C++%26%3D++-+%5Cleft%5B+%7B%7B%7B%5Cbar+P%7D_%7Bij%7D%7D%5Clog+%7BP_%7Bij%7D%7D+%2B+%281+-+%7B%7B%5Cbar+P%7D_%7Bij%7D%7D%29%5Clog+%281+-+%7BP_%7Bij%7D%7D%29%7D+%5Cright%5D%5C%5C++%26%3D++-+%7B%7B%5Cbar+P%7D_%7Bij%7D%7D%5Clog+%7BP_%7Bij%7D%7D+-+%281+-+%7B%7B%5Cbar+P%7D_%7Bij%7D%7D%29%5Clog+%281+-+%7BP_%7Bij%7D%7D%29+%5Cend%7Balign%2A%7D%5C++%285%29%5C%5C) 

将 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7BP_%7Bij%7D%7D%5C%5D) 表达式（3）和 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7B%5Cbar+P_%7Bij%7D%7D%5C%5D) 表达式（4）代入（5）式中，得到单个样本的交叉熵损失函数具体表达式为

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%2A%7D+%7BC_%7Bij%7D%7D+%26%3D++-+%7B%7B%5Cbar+P%7D_%7Bij%7D%7D%5Clog+%7BP_%7Bij%7D%7D+-+%281+-+%7B%7B%5Cbar+P%7D_%7Bij%7D%7D%29%5Clog+%281+-+%7BP_%7Bij%7D%7D%29%5C%5C++%26%3D++-+%5Cfrac%7B1%7D%7B2%7D%281+%2B+%7BS_%7Bij%7D%7D%29+%5Ccdot+%5Clog+%5Cfrac%7B1%7D%7B%7B1+%2B+%7Be%5E%7B+-+%5Csigma+%28%7Bs_i%7D+-+%7Bs_j%7D%29%7D%7D%7D%7D+-+%5Cleft%5B+%7B1+-+%5Cfrac%7B1%7D%7B2%7D%281+%2B+%7BS_%7Bij%7D%7D%29%7D+%5Cright%5D+%5Ccdot+%5Clog+%5Cleft%5B+%7B1+-+%5Cfrac%7B1%7D%7B%7B1+%2B+%7Be%5E%7B+-+%5Csigma+%28%7Bs_i%7D+-+%7Bs_j%7D%29%7D%7D%7D%7D%7D+%5Cright%5D%5C%5C++%26%3D++-+%5Cfrac%7B1%7D%7B2%7D%281+%2B+%7BS_%7Bij%7D%7D%29+%5Ccdot+%5Clog+%5Cfrac%7B1%7D%7B%7B1+%2B+%7Be%5E%7B+-+%5Csigma+%28%7Bs_i%7D+-+%7Bs_j%7D%29%7D%7D%7D%7D+-+%5Cfrac%7B1%7D%7B2%7D%281+-+%7BS_%7Bij%7D%7D%29+%5Ccdot+%5Clog+%5Cleft%5B+%7B%5Cfrac%7B%7B%7Be%5E%7B+-+%5Csigma+%28%7Bs_i%7D+-+%7Bs_j%7D%29%7D%7D%7D%7D%7B%7B1+%2B+%7Be%5E%7B+-+%5Csigma+%28%7Bs_i%7D+-+%7Bs_j%7D%29%7D%7D%7D%7D%7D+%5Cright%5D%5C%5C++%26%3D++-+%5Cfrac%7B1%7D%7B2%7D%281+%2B+%7BS_%7Bij%7D%7D%29+%5Ccdot+%5Clog+%5Cfrac%7B1%7D%7B%7B1+%2B+%7Be%5E%7B+-+%5Csigma+%28%7Bs_i%7D+-+%7Bs_j%7D%29%7D%7D%7D%7D+-+%5Cfrac%7B1%7D%7B2%7D%281+-+%7BS_%7Bij%7D%7D%29+%5Ccdot+%5Cleft%5B+%7B+-+%5Csigma+%28%7Bs_i%7D+-+%7Bs_j%7D%29+%2B+%5Clog+%5Cfrac%7B1%7D%7B%7B1+%2B+%7Be%5E%7B+-+%5Csigma+%28%7Bs_i%7D+-+%7Bs_j%7D%29%7D%7D%7D%7D%7D+%5Cright%5D%5C%5C++%26%3D+%5Cfrac%7B1%7D%7B2%7D%281+-+%7BS_%7Bij%7D%7D%29+%5Ccdot+%5Csigma+%28%7Bs_i%7D+-+%7Bs_j%7D%29+%2B+%5Clog+%5Cleft%5B+%7B1+%2B+%7Be%5E%7B+-+%5Csigma+%28%7Bs_i%7D+-+%7Bs_j%7D%29%7D%7D%7D+%5Cright%5D+%5Cend%7Balign%2A%7D%5C++%286%29%5C%5C) 

所以 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7BC_%7Bij%7D%7D%5C%5D) 关于任一待优化参数 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7Bw_k%7D%5C%5D) 的偏导数为

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%7B%5Cpartial+%7BC_%7Bij%7D%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+%3D+%5Cfrac%7B%7B%5Cpartial+%7BC_%7Bij%7D%7D%7D%7D%7B%7B%5Cpartial+%7Bs_i%7D%7D%7D%5Cfrac%7B%7B%5Cpartial+%7Bs_i%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+%2B+%5Cfrac%7B%7B%5Cpartial+%7BC_%7Bij%7D%7D%7D%7D%7B%7B%5Cpartial+%7Bs_j%7D%7D%7D%5Cfrac%7B%7B%5Cpartial+%7Bs_j%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D%5C++%287%29%5C%5C) 

使用随机梯度下降法（SGD）对参数进行优化：

![[公式]](https://www.zhihu.com/equation?tex=%7Bw_k%7D+%5Cto+%7Bw_k%7D+-+%5Ceta+%5Cfrac%7B%7B%5Cpartial+%7BC_%7Bij%7D%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+%3D+%7Bw_k%7D+-+%5Ceta+%5Cleft%28+%7B%5Cfrac%7B%7B%5Cpartial+%7BC_%7Bij%7D%7D%7D%7D%7B%7B%5Cpartial+%7Bs_i%7D%7D%7D%5Cfrac%7B%7B%5Cpartial+%7Bs_i%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+%2B+%5Cfrac%7B%7B%5Cpartial+%7BC_%7Bij%7D%7D%7D%7D%7B%7B%5Cpartial+%7Bs_j%7D%7D%7D%5Cfrac%7B%7B%5Cpartial+%7Bs_j%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D%7D+%5Cright%29%5C++%288%29%5C%5C) 

其中 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%5Ceta+%5C%5D) 表示学习率。由于式中的

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%2A%7D+%5Cfrac%7B%7B%5Cpartial+%7BC_%7Bij%7D%7D%7D%7D%7B%7B%5Cpartial+%7Bs_i%7D%7D%7D+%26%3D+%5Cfrac%7B%5Cpartial+%7D%7B%7B%5Cpartial+%7Bs_i%7D%7D%7D%5Cleft%5C%7B+%7B%5Cfrac%7B1%7D%7B2%7D%281+-+%7BS_%7Bij%7D%7D%29+%5Ccdot+%5Csigma+%28%7Bs_i%7D+-+%7Bs_j%7D%29+%2B+%5Clog+%5Cleft%5B+%7B1+%2B+%7Be%5E%7B+-+%5Csigma+%28%7Bs_i%7D+-+%7Bs_j%7D%29%7D%7D%7D+%5Cright%5D%7D+%5Cright%5C%7D%5C%5C++%26%3D+%5Csigma+%5Cleft%5B+%7B%5Cfrac%7B1%7D%7B2%7D%281+-+%7BS_%7Bij%7D%7D%29+-+%5Cfrac%7B1%7D%7B%7B1+%2B+%7Be%5E%7B+-+%5Csigma+%28%7Bs_i%7D+-+%7Bs_j%7D%29%7D%7D%7D%7D%7D+%5Cright%5D+%5Cend%7Balign%2A%7D%5C++%289%29%5C%5C) 

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%2A%7D+%5Cfrac%7B%7B%5Cpartial+%7BC_%7Bij%7D%7D%7D%7D%7B%7B%5Cpartial+%7Bs_j%7D%7D%7D+%26%3D+%5Cfrac%7B%5Cpartial+%7D%7B%7B%5Cpartial+%7Bs_j%7D%7D%7D%5Cleft%5C%7B+%7B%5Cfrac%7B1%7D%7B2%7D%281+-+%7BS_%7Bij%7D%7D%29+%5Ccdot+%5Csigma+%28%7Bs_i%7D+-+%7Bs_j%7D%29+%2B+%5Clog+%5Cleft%5B+%7B1+%2B+%7Be%5E%7B+-+%5Csigma+%28%7Bs_i%7D+-+%7Bs_j%7D%29%7D%7D%7D+%5Cright%5D%7D+%5Cright%5C%7D%5C%5C++%26%3D++-+%5Csigma+%5Cleft%5B+%7B%5Cfrac%7B1%7D%7B2%7D%281+-+%7BS_%7Bij%7D%7D%29+-+%5Cfrac%7B1%7D%7B%7B1+%2B+%7Be%5E%7B+-+%5Csigma+%28%7Bs_i%7D+-+%7Bs_j%7D%29%7D%7D%7D%7D%7D+%5Cright%5D%5C%5C+%26+%3D++-+%5Cfrac%7B%7B%5Cpartial+%7BC_%7Bij%7D%7D%7D%7D%7B%7B%5Cpartial+%7Bs_i%7D%7D%7D+%5Cend%7Balign%2A%7D%5C++%2810%29%5C%5C) 

所以我们记

![[公式]](https://www.zhihu.com/equation?tex=%7B%5Clambda+_%7Bij%7D%7D%5Cmathop++%3D+%5Climits%5E%7Bdef%7D+%5Cfrac%7B%7B%5Cpartial+%7BC_%7Bij%7D%7D%7D%7D%7B%7B%5Cpartial+%7Bs_i%7D%7D%7D+%3D++-+%5Cfrac%7B%7B%5Cpartial+%7BC_%7Bij%7D%7D%7D%7D%7B%7B%5Cpartial+%7Bs_j%7D%7D%7D%5C++%2811%29%5C%5C) 

对于集合*I*中的样本，总的loss记为

![[公式]](https://www.zhihu.com/equation?tex=C+%3D+%5Csum%5Climits_%7B%5C%7B+i%2Cj%5C%7D++%5Cin+I%7D+%7B%7BC_%7Bij%7D%7D%7D+%5C++%2812%29%5C%5C) 

集合*I*将用于RankNet的加速训练公式的推导，它表示某一个query下，所有相关文档组成的文档对 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%5Cleft%5C%7B+%7Bi%2Cj%7D+%5Cright%5C%7D%5C%5D) ，每个文档对**仅在\*I\*中出现一次**，不能重复出现，即 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%5Cleft%5C%7B+%7Bi%2Cj%7D+%5Cright%5C%7D%5C%5D) 与 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%5Cleft%5C%7B+%7Bj%2Ci%7D+%5Cright%5C%7D%5C%5D) 等价，且 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5Bi+%5Cne+j%5C%5D) 。为方便起见，我们假设*I*中的文档对 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%5Cleft%5C%7B+%7Bi%2Cj%7D+%5Cright%5C%7D%5C%5D) 均满足 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5Bdo%7Bc_i%7D+%5Ctriangleright+do%7Bc_j%7D%5C%5D) ，即相关性大的文档下标*i*在前，相关性小的文档下标*j*在后。故对于*I*中所有的文档对 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%5Cleft%5C%7B+%7Bi%2Cj%7D+%5Cright%5C%7D%5C%5D) ，均满足 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7BS_%7Bij%7D%7D+%3D+1%5C%5D) 。则有

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%7B%5Cpartial+C%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D%5Cmathop++%3D+%5Climits%5E%7B%5Ctext%7B1%7D%7D+%5Csum%5Climits_%7B%5C%7B+i%2Cj%5C%7D++%5Cin+I%7D+%7B%5Cfrac%7B%7B%5Cpartial+%7BC_%7Bij%7D%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D%7D+%5Cmathop++%3D+%5Climits%5E%7B%5Ctext%7B2%7D%7D+%5Csum%5Climits_%7B%5C%7B+i%2Cj%5C%7D++%5Cin+I%7D+%7B%7B%5Clambda+_%7Bij%7D%7D%5Cleft%28+%7B%5Cfrac%7B%7B%5Cpartial+%7Bs_i%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+-+%5Cfrac%7B%7B%5Cpartial+%7Bs_j%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D%7D+%5Cright%29%7D+%5Cmathop++%3D+%5Climits%5E%7B%5Ctext%7B3%7D%7D+%5Csum%5Climits_i+%7B%7B%5Clambda+_i%7D%5Cfrac%7B%7B%5Cpartial+%7Bs_i%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D%7D+%5C++%2813%29%5C%5C) 

其中

![[公式]](https://www.zhihu.com/equation?tex=%7B%5Clambda+_i%7D+%3D+%5Csum%5Climits_%7Bj%3A%5Cleft%5C%7B+%7Bi%2Cj%7D+%5Cright%5C%7D+%5Cin+I%7D+%7B%7B%5Clambda+_%7Bij%7D%7D%7D++-+%5Csum%5Climits_%7Bj%3A%5Cleft%5C%7B+%7Bj%2Ci%7D+%5Cright%5C%7D+%5Cin+I%7D+%7B%7B%5Clambda+_%7Bij%7D%7D%7D+%5C++%2814%29%5C%5C) 

式（14）的含义是：对于文档*i*：我们首先找到所有相关性排在文档*i*后面的文档*j*（组成 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%5Cleft%5C%7B+%7Bi%2Cj%7D+%5Cright%5C%7D%5C%5D) ），并找到所有相关性排在文档*i*前面的文档*k*（组成 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%5Cleft%5C%7B+%7Bk%2Ci%7D+%5Cright%5C%7D%5C%5D) ）（排在前面的文档代表相关性更强）；再对所有的 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7B%5Clambda+_%7Bij%7D%7D%5C%5D) 求和，其组成了式（14）的第一项，对所有的 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7B%5Clambda+_%7Bki%7D%7D%5C%5D) 求和，其组成了式（14）的第二项。由于第一项和第二项的求和符号互不关联（互相没有联系），所以第二项中的k可改为j。

虽然上文描述了式（14）的含义，但是式（13）中的等号③以及式（14）的成立依然较难理解。我们以两个例子来说明等号③为什么成立（作者在原论文[1]中也是通过例子来说明的）。

例1：如果仅有两个文档与某一query相关，且其真实相关性满足 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7Bd_1%7D+%5Ctriangleright+%7Bd_2%7D%5C%5D) ，则集合 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5BI+%3D+%5Cleft%5C%7B+%7B%5Cleft%5C%7B+%7B1%2C2%7D+%5Cright%5C%7D%7D+%5Cright%5C%7D%5C%5D)，且 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5Bi+%3D+1%5C%5D) ， ![[公式]](https://www.zhihu.com/equation?tex=%5C%5Bj+%3D+2%5C%5D) ，此时

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%7B%5Cpartial+C%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+%3D+%5Csum%5Climits_%7B%5C%7B+i%2Cj%5C%7D++%5Cin+I%7D+%7B%7B%5Clambda+_%7Bij%7D%7D%5Cleft%28+%7B%5Cfrac%7B%7B%5Cpartial+%7Bs_i%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+-+%5Cfrac%7B%7B%5Cpartial+%7Bs_j%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D%7D+%5Cright%29%7D++%3D+%7B%5Clambda+_%7B12%7D%7D%5Cfrac%7B%7B%5Cpartial+%7Bs_1%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+-+%7B%5Clambda+_%7B12%7D%7D%5Cfrac%7B%7B%5Cpartial+%7Bs_2%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D%5C++%2815%29%5C%5C) 

式（15）的模样已经很像式（13）了，我们只需令

![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%5C%7B+%5Cbegin%7Bgathered%7D+++%7B%5Clambda+_1%7D+%3D+%7B%5Clambda+_%7B12%7D%7D+%5Chfill+%5C%5C+++%7B%5Clambda+_2%7D+%3D++-+%7B%5Clambda+_%7B12%7D%7D+%5Chfill+%5C%5C++%5Cend%7Bgathered%7D++%5Cright.%5C++%2816%29%5C%5C) 

便可得到

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%7B%5Cpartial+C%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+%3D+%7B%5Clambda+_1%7D%5Cfrac%7B%7B%5Cpartial+%7Bs_1%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+%2B+%7B%5Clambda+_2%7D%5Cfrac%7B%7B%5Cpartial+%7Bs_2%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+%3D+%5Csum%5Climits_%7Bi+%3D+1%7D%5E2+%7B%7B%5Clambda+_i%7D%5Cfrac%7B%7B%5Cpartial+%7Bs_i%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D%7D+%5C++%2817%29%5C%5C) 

式（17）便是式（13）的一个具体实例。在式（16）中我们用单数字下标（下标只有一个数字）的 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7B%5Clambda+_1%7D%5C%5D) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7B%5Clambda+_2%7D%5C%5D) 代替了 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%5Cfrac%7B%7B%5Cpartial+%7Bs_1%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D%5C%5D) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%5Cfrac%7B%7B%5Cpartial+%7Bs_2%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D%5C%5D) 前面由双数字下标 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B+%5Cpm+%7B%5Clambda+_%7B12%7D%7D%5C%5D) （下标有两个数字）组成的系数，得到了如式（17）这样的形式。反过来，式（16）也可用式（14）解释，我们以文档 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7Bd_2%7D%5C%5D) 前的系数 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7B%5Clambda+_2%7D%5C%5D) 为例：（此时i=2）由于真实相关性满足 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7Bd_1%7D+%5Ctriangleright+%7Bd_2%7D%5C%5D) ，因此 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7Bd_1%7D%5C%5D) 的相关性强于 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7Bd_2%7D%5C%5D) ，所以 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7B%5Clambda+_%7B12%7D%7D%5C%5D) 应该放在式（14）的第二项。同理可解释 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7B%5Clambda+_1%7D%5C%5D) 的表达式。

例2：如果仅有三个文档与某一query相关，且其真实相关性满足 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7Bd_1%7D+%5Ctriangleright+%7Bd_2%7D+%5Ctriangleright+%7Bd_3%7D%5C%5D) ，则集合 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5BI+%3D+%5Cleft%5C%7B+%7B%5Cleft%5C%7B+%7B1%2C2%7D+%5Cright%5C%7D%2C%5Cleft%5C%7B+%7B1%2C3%7D+%5Cright%5C%7D%2C%5Cleft%5C%7B+%7B2%2C3%7D+%5Cright%5C%7D%7D+%5Cright%5C%7D%5C%5D) ，共包含3对文档对，此时

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%2A%7D+++%5Cfrac%7B%7B%5Cpartial+C%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+%26%3D+%5Csum%5Climits_%7B%5C%7B+i%2Cj%5C%7D++%5Cin+I%7D+%7B%7B%5Clambda+_%7Bij%7D%7D%5Cleft%28+%7B%5Cfrac%7B%7B%5Cpartial+%7Bs_i%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+-+%5Cfrac%7B%7B%5Cpartial+%7Bs_j%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D%7D+%5Cright%29%7D++%5C%5C+++++%26%3D+%5Cleft%28+%7B%7B%5Clambda+_%7B12%7D%7D%5Cfrac%7B%7B%5Cpartial+%7Bs_1%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+-+%7B%5Clambda+_%7B12%7D%7D%5Cfrac%7B%7B%5Cpartial+%7Bs_2%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D%7D+%5Cright%29+%2B+%5Cleft%28+%7B%7B%5Clambda+_%7B13%7D%7D%5Cfrac%7B%7B%5Cpartial+%7Bs_1%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+-+%7B%5Clambda+_%7B13%7D%7D%5Cfrac%7B%7B%5Cpartial+%7Bs_3%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D%7D+%5Cright%29+%2B+%5Cleft%28+%7B%7B%5Clambda+_%7B23%7D%7D%5Cfrac%7B%7B%5Cpartial+%7Bs_2%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+-+%7B%5Clambda+_%7B23%7D%7D%5Cfrac%7B%7B%5Cpartial+%7Bs_3%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D%7D+%5Cright%29+%5C%5C+++++%26%3D+%5Cleft%28+%7B%7B%5Clambda+_%7B12%7D%7D+%2B+%7B%5Clambda+_%7B13%7D%7D%7D+%5Cright%29%5Cfrac%7B%7B%5Cpartial+%7Bs_1%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+%2B+%5Cleft%28+%7B%7B%5Clambda+_%7B23%7D%7D+-+%7B%5Clambda+_%7B12%7D%7D%7D+%5Cright%29%5Cfrac%7B%7B%5Cpartial+%7Bs_2%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+%2B+%5Cleft%28+%7B+-+%7B%5Clambda+_%7B13%7D%7D+-+%7B%5Clambda+_%7B23%7D%7D%7D+%5Cright%29%5Cfrac%7B%7B%5Cpartial+%7Bs_3%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+%5C%5C++%5Cend%7Balign%2A%7D+%5C++%2818%29%5C%5C) 

令

![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%5C%7B+%5Cbegin%7Bgathered%7D+++%7B%5Clambda+_1%7D+%3D+%7B%5Clambda+_%7B12%7D%7D+%2B+%7B%5Clambda+_%7B13%7D%7D+%5Chfill+%5C%5C+++%7B%5Clambda+_2%7D+%3D+%7B%5Clambda+_%7B23%7D%7D+-+%7B%5Clambda+_%7B12%7D%7D+%5Chfill+%5C%5C+++%7B%5Clambda+_3%7D+%3D++-+%7B%5Clambda+_%7B13%7D%7D+-+%7B%5Clambda+_%7B23%7D%7D+%5Chfill+%5C%5C++%5Cend%7Bgathered%7D++%5Cright.%5C+++%2819%29%5C%5C) 

最终得到

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%7B%5Cpartial+C%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+%3D+%7B%5Clambda+_1%7D%5Cfrac%7B%7B%5Cpartial+%7Bs_1%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+%2B+%7B%5Clambda+_2%7D%5Cfrac%7B%7B%5Cpartial+%7Bs_2%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+%2B+%7B%5Clambda+_3%7D%5Cfrac%7B%7B%5Cpartial+%7Bs_3%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+%3D+%5Csum%5Climits_%7Bi+%3D+1%7D%5E3+%7B%7B%5Clambda+_i%7D%5Cfrac%7B%7B%5Cpartial+%7Bs_i%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D%7D+%5C++%2820%29%5C%5C) 

式（18）对 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7B%5Clambda+_%7Bij%7D%7D%5C%5D) 进行了重新排列组合，接着在式（19）中我们使用单下标（下标只有一个数字）的 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7B%5Clambda+_1%7D%5C%5D) 、 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7B%5Clambda+_2%7D%5C%5D) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7B%5Clambda+_3%7D%5C%5D) 代替 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%5Cfrac%7B%7B%5Cpartial+%7Bs_1%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D%5C%5D) 、 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%5Cfrac%7B%7B%5Cpartial+%7Bs_2%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D%5C%5D) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%5Cfrac%7B%7B%5Cpartial+%7Bs_3%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D%5C%5D) 前双下标（下标有两个数字）组成的系数，得到了如式（14）这样的形式。反过来，式（19）也可由式（14）推得，我们以文档 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7Bd_2%7D%5C%5D) 前的系数 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7B%5Clambda+_2%7D%5C%5D) 为例：（此时i=2）由于真实相关性满足 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7Bd_1%7D+%5Ctriangleright+%7Bd_2%7D+%5Ctriangleright+%7Bd_3%7D%5C%5D) ，因此 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7Bd_3%7D%5C%5D) 的相关性弱于 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7Bd_2%7D%5C%5D) ， ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7Bd_1%7D%5C%5D) 的相关性强于 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7Bd_2%7D%5C%5D) ，所以 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7B%5Clambda+_%7B23%7D%7D%5C%5D) 应该放在式（14）第一项， ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7B%5Clambda+_%7B12%7D%7D%5C%5D) 应该在式（14）第二项。同理可得 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7B%5Clambda+_1%7D%5C%5D) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7B%5Clambda+_3%7D%5C%5D) 的表达式。

我们通过以上仅包含两个和三个相关文档的例子对式（13）和式（14）进行了说明。若使用他们进行优化迭代，便将SGD算法转化为了mini-batch SGD算法，如式（21）所示。此时，RankNet在单次迭代时会对同一query下所有docs遍历后更新权值，训练时间得以从 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5BO%5Cleft%28+%7B%7Bn%5E2%7D%7D+%5Cright%29%5C%5D) 降至 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5BO%5Cleft%28+%7B%7Bn%7D%7D+%5Cright%29%5C%5D) ，*n*为单条query下docs的平均数，它被称为**RankNet算法的加速训练**，具体证明可参见文献[1-2]。

![[公式]](https://www.zhihu.com/equation?tex=%7Bw_k%7D+%5Cto+%7Bw_k%7D+-+%5Ceta+%5Cfrac%7B%7B%5Cpartial+C%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D+%3D+%7Bw_k%7D+-+%5Ceta++%5Ccdot+%5Csum%5Climits_i+%7B%7B%5Clambda+_i%7D%5Cfrac%7B%7B%5Cpartial+%7Bs_i%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D%7D+%5C++%2821%29%5C%5C) 

我们再观察式（14）: ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7B%5Clambda+_i%7D+%3D+%5Csum%5Cnolimits_%7Bj%3A%5C%7B+i%2Cj%5C%7D++%5Cin+I%7D+%7B%7B%5Clambda+_%7Bij%7D%7D%7D++-+%5Csum%5Cnolimits_%7Bj%3A%5C%7B+j%2Ci%5C%7D++%5Cin+I%7D+%7B%7B%5Clambda+_%7Bij%7D%7D%7D+%5C%5D) ， ![[公式]](https://www.zhihu.com/equation?tex=%7B%5Clambda+_i%7D) 决定着第i个文档在迭代中的移动方向和幅度，表明每个文档下次调序的方向和强度取决于所有同一query下的其他不同文档。式（14）的第一项和第二项表明，若排在文档i后面的文档越少，或者排在文档i前面的文档越多，文档i向前移动的幅度就越大( ![[公式]](https://www.zhihu.com/equation?tex=%7B%5Clambda+_i%7D) 负的越多越向前移动)。我认为或许这也能说明RankNet为什么更倾向于对靠后位置的相关文档的排序位置的提升。

## LambdaRank算法

上面我们介绍了以错误pair最少为优化目标的RankNet算法，然而许多时候仅以错误pair数来评价排序的好坏是不够的，像NDCG或者ERR等信息检索中的评价指标就只关注top  k个结果的排序。由于这些指标不可导或导数不存在，当我们采用RankNet算法时，往往无法以它们为优化目标（损失函数）进行迭代，所以RankNet的优化目标和信息检索评价指标之间还是存在差距的。以下图为例：

![img](https://pic1.zhimg.com/80/v2-2d96d769ba10c7529780b52fbcfc9afe_720w.jpg)图3 RankNet中的损失函数的缺点举例 

在上图中，每一条线表示一个文档，蓝色表示相关文档，灰色表示不相关文档。RankNet以Error  pair（错误文档对数目）的方式计算cost。左边排序1排序错误的文档对（pair）共有13对，故cost为13，右边排序2通过把第一个相关文档下调3个位置，第二个相关文档上条5个位置，将cost降为11，但是像NDCG或者ERR等指标只关注top  k个结果的排序，在优化过程中下调前面相关文档的位置不是我们想要得到的结果。上图排序2左边黑色的箭头表示RankNet下一轮的调序方向和强度，但我们真正需要的是右边红色箭头代表的方向和强度，即更关注靠前位置的相关文档的排序位置的提升。LambdaRank正是基于这个思想演化而来，其中Lambda指的就是红色箭头，代表下一次迭代优化的方向和强度，也就是梯度。

具体来说，由于需要对现有的loss或loss的梯度进行改进，而NDCG等指标又不可导，我们便跳过loss，直接简单粗暴地在RankNet加速算法形式的梯度上（式（22））再乘一项，以此新定义了一个Lambda梯度，如式（23）所示。其中Z表示评价指标，可取NDCG、ERR等指标。把交换两个文档的位置引起的评价指标的变化作为其中一个因子，实验表明对模型效果有显著的提升[1]，具体证明可参见文献[2]，作者在其中还从力学的角度对LambdaRank进行了解释。

![[公式]](https://www.zhihu.com/equation?tex=%7B%5Clambda+_%7Bij%7D%7D+%3D+%5Csigma+%5Cleft%5B+%7B%5Cfrac%7B1%7D%7B2%7D%281+-+%7BS_%7Bij%7D%7D%29+-+%5Cfrac%7B1%7D%7B%7B1+%2B+%7Be%5E%7B+-+%5Csigma+%28%7Bs_i%7D+-+%7Bs_j%7D%29%7D%7D%7D%7D%7D+%5Cright%5D%2C%5C+%7BS_%7Bij%7D%7D+%3D+1%5C++%2822%29%5C%5C) 

![[公式]](https://www.zhihu.com/equation?tex=%7B%5Clambda+_%7Bij%7D%7D+%3D++-+%5Cfrac%7B%5Csigma+%7D%7B%7B1+%2B+%7Be%5E%7B%5Csigma+%28%7Bs_i%7D+-+%7Bs_j%7D%29%7D%7D%7D%7D%5Cleft%7C+%7B%5CDelta+%7BZ_%7Bij%7D%7D%7D+%5Cright%7C%5C+++%2823%29%5C%5C) 

损失函数的梯度代表了文档下一次迭代优化的方向和强度，由于引入了更关注头部正确率的评价指标，Lambda梯度得以让位置靠前的优质文档排序位置进一步提升。有效避免了排位靠前的优质文档的位置被下调的情况发生。LambdaRank相比RankNet的优势在于分解因式后训练速度变快，同时考虑了评价指标，直接对问题求解，效果更明显。此外需要注意的是，由于之前我们并未对得分函数 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5Bs+%3D+f%5Cleft%28+%7Bx%3Bw%7D+%5Cright%29%5C%5D) 具体规定，所以它的选择比较自由，可以是RankNet中使用的NN，也可以是LambdaMART使用的MART，还可以是GBDT等，随之求得的预测得分 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7Bs_i%7D%5C%5D) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7Bs_j%7D%5C%5D) 对于 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7Bw_k%7D%5C%5D) 的偏导数 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%5Cfrac%7B%7B%5Cpartial+%7Bs_i%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D%5C%5D) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%5Cfrac%7B%7B%5Cpartial+%7Bs_j%7D%7D%7D%7B%7B%5Cpartial+%7Bw_k%7D%7D%7D%5C%5D) 也不同。

但是本人对于式（13）有一个问题尚未想明白：我们假设了*I*中的文档对 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%5Cleft%5C%7B+%7Bi%2Cj%7D+%5Cright%5C%7D%5C%5D) 均满足 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5Bdo%7Bc_i%7D+%5Ctriangleright+do%7Bc_j%7D%5C%5D) ，故对于*I*中所有的文档对 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%5Cleft%5C%7B+%7Bi%2Cj%7D+%5Cright%5C%7D%5C%5D) ，均满足 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7BS_%7Bij%7D%7D+%3D+1%5C%5D) ，在此情况下便没有了 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5B%7BS_%7Bij%7D%7D+%3D+0%5C%5D) 的样本，会不会带来严重的样本不均衡问题，抑或是其实已经考虑了所有情形？

------

## 参考文献

[1] C. Burges. From RankNet to LambdaRank to LambdaMART: An overview[J]. 2014

[2] C. Burges, R Ragno, Q V Le. Learning to Rank with Nonsmooth Cost  Functions[C]. Advances in Neural Information Processing Systems 19,  Proceedings of the Twentieth Annual Conference on Neural Information  Processing Systems, Vancouver, British Columbia, Canada, 2006

[3] 笨兔勿应：Learning to Rank算法介绍：RankNet，LambdaRank，LambdaMart [https://www.cnblogs.com/bentuwuying/p/6690836.html](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/bentuwuying/p/6690836.html)

[4] 始终：LambdaMART 不太简短之介绍 [https://liam.page/2016/07/10/a-not-so-simple-introduction-to-lambdamart/](https://link.zhihu.com/?target=https%3A//liam.page/2016/07/10/a-not-so-simple-introduction-to-lambdamart/)

[5] 见鹿: 《Learning to Rank using Gradient Descent》https://zhuanlan.zhihu.com/p/20711017

[6] C. Burges, T. Shaked, E. Renshaw, et al. Learning to rank using  gradient descent[C]. Proceedings of the 22nd International Conference on Machine learning, Bonn, Germany, 2005, 89-96

编辑于 2019-07-20