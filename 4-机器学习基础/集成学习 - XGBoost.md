# 集成学习 - XGBoost

---

## 1. XGBoost

XGBoost 是大规模并行 boosting tree 的工具，它是目前最快最好的开源  boosting tree 工具包，比常见的工具包快 10 倍以上。Xgboost 和 GBDT 两者都是 boosting  方法，除了工程实现、解决问题上的一些差异外，最大的不同就是目标函数的定义。故本文将从数学原理和工程实现上进行介绍，并在最后介绍下 Xgboost 的优点。

### 1.1 数学原理

**1.1.1 目标函数**

我们知道 XGBoost 是由 ![[公式]](https://www.zhihu.com/equation?tex=k) 个基模型组成的一个加法运算式： 

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_i%3D%5Csum_%7Bt%3D1%7D%5E%7Bk%7D%5C+f_t%28x_i%29+%5C%5C) 

其中 ![[公式]](https://www.zhihu.com/equation?tex=f_k) 为第 ![[公式]](https://www.zhihu.com/equation?tex=k) 个基模型， ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_i) 为第 ![[公式]](https://www.zhihu.com/equation?tex=i) 个样本的预测值。

损失函数可由预测值 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_i) 与真实值 ![[公式]](https://www.zhihu.com/equation?tex=y_i) 进行表示： 

![[公式]](https://www.zhihu.com/equation?tex=L%3D%5Csum_%7Bi%3D1%7D%5En+l%28+y_i%2C+%5Chat%7By%7D_i%29+%5C%5C) 

其中 ![[公式]](https://www.zhihu.com/equation?tex=n) 为样本数量。

我们知道模型的预测精度由模型的偏差和方差共同决定，损失函数代表了模型的偏差，想要方差小则需要简单的模型，所以目标函数由模型的损失函数 ![[公式]](https://www.zhihu.com/equation?tex=L) 与抑制模型复杂度的正则项 ![[公式]](https://www.zhihu.com/equation?tex=%5COmega) 组成，所以我们有： 

![[公式]](https://www.zhihu.com/equation?tex=Obj+%3D%5Csum_%7Bi%3D1%7D%5En+l%28%5Chat%7By%7D_i%2C+y_i%29+%2B+%5Csum_%7Bt%3D1%7D%5Ek+%5COmega%28f_t%29+%5C%5C+) 

![[公式]](https://www.zhihu.com/equation?tex=%5COmega) 为模型的正则项，由于 XGBoost 支持决策树也支持线性模型，所以这里再不展开描述。

我们知道 boosting 模型是前向加法，以第 ![[公式]](https://www.zhihu.com/equation?tex=t) 步的模型为例，模型对第 ![[公式]](https://www.zhihu.com/equation?tex=i) 个样本 ![[公式]](https://www.zhihu.com/equation?tex=x_%7Bi%7D) 的预测为：

![[公式]](https://www.zhihu.com/equation?tex=++%5Chat%7By%7D_i%5Et%3D+%5Chat%7By%7D_i%5E%7Bt-1%7D+%2B+f_t%28x_i%29++%5C%5C) 

其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_i%5E%7Bt-1%7D) 由第 ![[公式]](https://www.zhihu.com/equation?tex=t-1) 步的模型给出的预测值，是已知常数，![[公式]](https://www.zhihu.com/equation?tex=f_t%28x_i%29) 是我们这次需要加入的新模型的预测值，此时，目标函数就可以写成： 

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+Obj%5E%7B%28t%29%7D+%26%3D+%5Csum_%7Bi%3D1%7D%5Enl%28y_i%2C+%5Chat%7By%7D_i%5Et%29+%2B+%5Csum_%7Bi%3D1%7D%5Et%5COmega%28f_i%29+%5C%5C++++%26%3D+%5Csum_%7Bi%3D1%7D%5En+l%5Cleft%28y_i%2C+%5Chat%7By%7D_i%5E%7Bt-1%7D+%2B+f_t%28x_i%29+%5Cright%29+%2B+%5Csum_%7Bi%3D1%7D%5Et++%5COmega%28f_i%29++%5Cend%7Balign%7D+%5C%5C) 

求此时最优化目标函数，就相当于求解 ![[公式]](https://www.zhihu.com/equation?tex=f_t%28x_i%29) 。

> 泰勒公式是将一个在 ![[公式]](https://www.zhihu.com/equation?tex=x%3Dx_0) 处具有 ![[公式]](https://www.zhihu.com/equation?tex=n) 阶导数的函数 ![[公式]](https://www.zhihu.com/equation?tex=f%28x%29) 利用关于 ![[公式]](https://www.zhihu.com/equation?tex=x-x_0) 的 ![[公式]](https://www.zhihu.com/equation?tex=n) 次多项式来逼近函数的方法，若函数 ![[公式]](https://www.zhihu.com/equation?tex=f%28x%29) 在包含 ![[公式]](https://www.zhihu.com/equation?tex=x_0) 的某个闭区间 ![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2Cb%5D) 上具有 ![[公式]](https://www.zhihu.com/equation?tex=n) 阶导数，且在开区间 ![[公式]](https://www.zhihu.com/equation?tex=%28a%2Cb%29) 上具有 ![[公式]](https://www.zhihu.com/equation?tex=n%2B1) 阶导数，则对闭区间 ![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2Cb%5D) 上任意一点 ![[公式]](https://www.zhihu.com/equation?tex=x) 有 ![[公式]](https://www.zhihu.com/equation?tex=%5Cdisplaystyle+f%28x%29%3D%5Csum_%7Bi%3D0%7D%5E%7Bn%7D%5Cfrac%7Bf%5E%7B%28i%29%7D%28x_0%29%7D%7Bi%21%7D%28x-x_0%29%5E+i%2BR_n%28x%29+) ，其中的多项式称为函数在 ![[公式]](https://www.zhihu.com/equation?tex=x_0) 处的泰勒展开式， ![[公式]](https://www.zhihu.com/equation?tex=R_n%28x%29) 是泰勒公式的余项且是 ![[公式]](https://www.zhihu.com/equation?tex=%28x%E2%88%92x_0%29%5En) 的高阶无穷小。

根据泰勒公式我们把函数 ![[公式]](https://www.zhihu.com/equation?tex=f%28x%2B%5CDelta+x%29) 在点 ![[公式]](https://www.zhihu.com/equation?tex=x) 处进行泰勒的二阶展开，可得到如下等式：

![[公式]](https://www.zhihu.com/equation?tex=f%28x%2B%5CDelta+x%29+%5Capprox+f%28x%29+%2B+f%27%28x%29%5CDelta+x+%2B+%5Cfrac12+f%27%27%28x%29%5CDelta+x%5E2++%5C%5C) 

我们把 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_i%5E%7Bt-1%7D) 视为 ![[公式]](https://www.zhihu.com/equation?tex=x) ， ![[公式]](https://www.zhihu.com/equation?tex=f_t%28x_i%29) 视为 ![[公式]](https://www.zhihu.com/equation?tex=%5CDelta+x) ，故可以将目标函数写为：

![[公式]](https://www.zhihu.com/equation?tex=Obj%5E%7B%28t%29%7D+%3D+%5Csum_%7Bi%3D1%7D%5En+%5Cleft%5B+l%28y_i%2C+%5Chat%7By%7D_i%5E%7Bt-1%7D%29+%2B+g_if_t%28x_i%29+%2B+%5Cfrac12h_if_t%5E2%28x_i%29+%5Cright%5D+%2B+%5Csum_%7Bi%3D1%7D%5Et++%5COmega%28f_i%29+%5C%5C) 

其中 ![[公式]](https://www.zhihu.com/equation?tex=g_%7Bi%7D) 为损失函数的一阶导， ![[公式]](https://www.zhihu.com/equation?tex=h_%7Bi%7D) 为损失函数的二阶导，**注意这里的导是对 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_i%5E%7Bt-1%7D) 求导**。

我们以平方损失函数为例：

![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bi%3D1%7D%5En+%5Cleft%28y_i+-+%28%5Chat%7By%7D_i%5E%7Bt-1%7D+%2B+f_t%28x_i%29%29+%5Cright%29%5E2++%5C%5C) 

则：

![[公式]](https://www.zhihu.com/equation?tex=++%5Cbegin%7Balign%7D++++++g_i+%26%3D+%5Cfrac%7B%5Cpartial+%28%5Chat%7By%7D%5E%7Bt-1%7D+-+y_i%29%5E2%7D%7B%5Cpartial+%7B%5Chat%7By%7D%5E%7Bt-1%7D%7D%7D+%3D+2%28%5Chat%7By%7D%5E%7Bt-1%7D+-+y_i%29+%5C%5C++++++h_i+%26%3D%5Cfrac%7B%5Cpartial%5E2%28%5Chat%7By%7D%5E%7Bt-1%7D+-+y_i%29%5E2%7D%7B%7B%5Chat%7By%7D%5E%7Bt-1%7D%7D%7D+%3D+2++++%5Cend%7Balign%7D++%5C%5C) 

由于在第 ![[公式]](https://www.zhihu.com/equation?tex=t) 步时 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_i%5E%7Bt-1%7D) 其实是一个已知的值，所以 ![[公式]](https://www.zhihu.com/equation?tex=l%28y_i%2C+%5Chat%7By%7D_i%5E%7Bt-1%7D%29) 是一个常数，其对函数的优化不会产生影响，因此目标函数可以写成：

![[公式]](https://www.zhihu.com/equation?tex=+Obj%5E%7B%28t%29%7D+%5Capprox+%5Csum_%7Bi%3D1%7D%5En+%5Cleft%5B+g_if_t%28x_i%29+%2B+%5Cfrac12h_if_t%5E2%28x_i%29+%5Cright%5D+%2B+%5Csum_%7Bi%3D1%7D%5Et++%5COmega%28f_i%29+%5C%5C) 

所以我们只需要求出每一步损失函数的一阶导和二阶导的值（由于前一步的 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D%5E%7Bt-1%7D) 是已知的，所以这两个值就是常数），然后最优化目标函数，就可以得到每一步的 ![[公式]](https://www.zhihu.com/equation?tex=f%28x%29) ，最后根据加法模型得到一个整体模型。

**1.1.2 基于决策树的目标函数**

我们知道 Xgboost 的基模型**不仅支持决策树，还支持线性模型**，这里我们主要介绍基于决策树的目标函数。

我们可以将决策树定义为 ![[公式]](https://www.zhihu.com/equation?tex=f_t%28x%29%3Dw_%7Bq%28x%29%7D) ， ![[公式]](https://www.zhihu.com/equation?tex=x) 为某一样本，这里的 ![[公式]](https://www.zhihu.com/equation?tex=q%28x%29) 代表了该样本在哪个叶子结点上，而 ![[公式]](https://www.zhihu.com/equation?tex=w_q) 则代表了叶子结点取值 ![[公式]](https://www.zhihu.com/equation?tex=w) ，所以 ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bq%28x%29%7D) 就代表了每个样本的取值 ![[公式]](https://www.zhihu.com/equation?tex=w) （即预测值）。

决策树的复杂度可由叶子数 ![[公式]](https://www.zhihu.com/equation?tex=T) 组成，叶子节点越少模型越简单，此外叶子节点也不应该含有过高的权重 ![[公式]](https://www.zhihu.com/equation?tex=w) （类比 LR 的每个变量的权重），所以目标函数的正则项可以定义为：

![[公式]](https://www.zhihu.com/equation?tex=%5COmega%28f_t%29%3D%5Cgamma+T+%2B+%5Cfrac12+%5Clambda+%5Csum_%7Bj%3D1%7D%5ET+w_j%5E2+%5C%5C) 

即决策树模型的复杂度由生成的所有决策树的叶子节点数量，和所有节点权重所组成的向量的 ![[公式]](https://www.zhihu.com/equation?tex=L_2) 范式共同决定。

![img](https://pic1.zhimg.com/80/v2-e0ab9287990a6098e4cdbc5a8cff4150_720w.jpg)

这张图给出了基于决策树的 XGBoost 的正则项的求解方式。

我们设 ![[公式]](https://www.zhihu.com/equation?tex=I_j%3D+%5C%7B+i+%5Cvert+q%28x_i%29%3Dj+%5C%7D) 为第 ![[公式]](https://www.zhihu.com/equation?tex=j) 个叶子节点的样本集合，故我们的目标函数可以写成：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+Obj%5E%7B%28t%29%7D+%26%5Capprox+%5Csum_%7Bi%3D1%7D%5En+%5Cleft%5B+g_if_t%28x_i%29+%2B+%5Cfrac12h_if_t%5E2%28x_i%29+%5Cright%5D+%2B+%5COmega%28f_t%29+%5C%5C++++%26%3D+%5Csum_%7Bi%3D1%7D%5En+%5Cleft%5B+g_iw_%7Bq%28x_i%29%7D+%2B+%5Cfrac12h_iw_%7Bq%28x_i%29%7D%5E2+%5Cright%5D+%2B+%5Cgamma+T+%2B+%5Cfrac12+%5Clambda+%5Csum_%7Bj%3D1%7D%5ETw_j%5E2+%5C%5C++++%26%3D+%5Csum_%7Bj%3D1%7D%5ET+%5Cleft%5B%28%5Csum_%7Bi+%5Cin+I_j%7Dg_i%29w_j+%2B+%5Cfrac12%28%5Csum_%7Bi+%5Cin+I_j%7Dh_i+%2B+%5Clambda%29w_j%5E2+%5Cright%5D+%2B+%5Cgamma+T+%5Cend%7Balign%7D+%5C%5C) 

第二步到第三步可能看的不是特别明白，这边做些解释：第二步是遍历所有的样本后求每个样本的损失函数，但样本最终会落在叶子节点上，所以我们也可以遍历叶子节点，然后获取叶子节点上的样本集合，最后在求损失函数。即我们之前样本的集合，现在都改写成叶子结点的集合，由于一个叶子结点有多个样本存在，因此才有了 ![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bi+%5Cin+I_j%7Dg_i) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bi+%5Cin+I_j%7Dh_i) 这两项， ![[公式]](https://www.zhihu.com/equation?tex=w_j) 为第 ![[公式]](https://www.zhihu.com/equation?tex=j) 个叶子节点取值。

为简化表达式，我们定义 ![[公式]](https://www.zhihu.com/equation?tex=G_j%3D%5Csum_%7Bi+%5Cin+I_j%7Dg_i) ， ![[公式]](https://www.zhihu.com/equation?tex=H_j%3D%5Csum_%7Bi+%5Cin+I_j%7Dh_i) ，则目标函数为： 

![[公式]](https://www.zhihu.com/equation?tex=Obj%5E%7B%28t%29%7D+%3D+%5Csum_%7Bj%3D1%7D%5ET+%5Cleft%5BG_jw_j+%2B+%5Cfrac12%28H_j+%2B+%5Clambda%29w_j%5E2+%5Cright%5D+%2B+%5Cgamma+T+%5C%5C) 

这里我们要注意 ![[公式]](https://www.zhihu.com/equation?tex=G_j) 和 ![[公式]](https://www.zhihu.com/equation?tex=H_j) 是前 ![[公式]](https://www.zhihu.com/equation?tex=t-1) 步得到的结果，其值已知可视为常数，只有最后一棵树的叶子节点 ![[公式]](https://www.zhihu.com/equation?tex=w_j) 不确定，那么将目标函数对 ![[公式]](https://www.zhihu.com/equation?tex=w_j) 求一阶导，并令其等于 ![[公式]](https://www.zhihu.com/equation?tex=0) ，则可以求得叶子结点 ![[公式]](https://www.zhihu.com/equation?tex=j) 对应的权值：

![[公式]](https://www.zhihu.com/equation?tex=w_j%5E%2A%3D-%5Cfrac%7BG_j%7D%7BH_j%2B%5Clambda%7D++%5C%5C) 

所以目标函数可以化简为： 

![[公式]](https://www.zhihu.com/equation?tex=Obj+%3D+-%5Cfrac12+%5Csum_%7Bj%3D1%7D%5ET+%5Cfrac%7BG_j%5E2%7D%7BH_j%2B%5Clambda%7D+%2B+%5Cgamma+T+%5C%5C) 

![img](https://pic4.zhimg.com/80/v2-f6db7af6c1e683192cb0ccf48eafaf99_720w.jpg)

上图给出目标函数计算的例子，求每个节点每个样本的一阶导数 ![[公式]](https://www.zhihu.com/equation?tex=g_i) 和二阶导数 ![[公式]](https://www.zhihu.com/equation?tex=h_i) ，然后针对每个节点对所含样本求和得到的 ![[公式]](https://www.zhihu.com/equation?tex=G_j) 和 ![[公式]](https://www.zhihu.com/equation?tex=H_j) ，最后遍历决策树的节点即可得到目标函数。

**1.1.3 最优切分点划分算法**

在决策树的生长过程中，一个非常关键的问题是如何找到叶子的节点的最优切分点，Xgboost 支持两种分裂节点的方法——贪心算法和近似算法。

**1）贪心算法**

1. 从深度为 ![[公式]](https://www.zhihu.com/equation?tex=0) 的树开始，对每个叶节点枚举所有的可用特征；
2. 针对每个特征，把属于该节点的训练样本根据该特征值进行升序排列，通过线性扫描的方式来决定该特征的最佳分裂点，并记录该特征的分裂收益；
3. 选择收益最大的特征作为分裂特征，用该特征的最佳分裂点作为分裂位置，在该节点上分裂出左右两个新的叶节点，并为每个新节点关联对应的样本集
4. 回到第 1 步，递归执行到满足特定条件为止

那么如何计算每个特征的分裂收益呢？

假设我们在某一节点完成特征分裂，则分列前的目标函数可以写为：

![[公式]](https://www.zhihu.com/equation?tex=Obj_%7B1%7D+%3D-%5Cfrac12+%5B%5Cfrac%7B%28G_L%2BG_R%29%5E2%7D%7BH_L%2BH_R%2B%5Clambda%7D%5D+%2B+%5Cgamma++%5C%5C) 

分裂后的目标函数为：

![[公式]](https://www.zhihu.com/equation?tex=Obj_2+%3D++-%5Cfrac12+%5B+%5Cfrac%7BG_L%5E2%7D%7BH_L%2B%5Clambda%7D+%2B+%5Cfrac%7BG_R%5E2%7D%7BH_R%2B%5Clambda%7D%5D+%2B2%5Cgamma+%5C%5C) 

则对于目标函数来说，分裂后的收益为：

![[公式]](https://www.zhihu.com/equation?tex=Gain%3D%5Cfrac12+%5Cleft%5B+%5Cfrac%7BG_L%5E2%7D%7BH_L%2B%5Clambda%7D+%2B+%5Cfrac%7BG_R%5E2%7D%7BH_R%2B%5Clambda%7D+-+%5Cfrac%7B%28G_L%2BG_R%29%5E2%7D%7BH_L%2BH_R%2B%5Clambda%7D%5Cright%5D+-+%5Cgamma+%5C%5C) 

注意该特征收益也可作为特征重要性输出的重要依据。

对于每次分裂，我们都需要枚举所有特征可能的分割方案，如何高效地枚举所有的分割呢？

我假设我们要枚举所有 ![[公式]](https://www.zhihu.com/equation?tex=x+%3C+a) 这样的条件，对于某个特定的分割点 ![[公式]](https://www.zhihu.com/equation?tex=a) 我们要计算 ![[公式]](https://www.zhihu.com/equation?tex=a) 左边和右边的导数和。

![img](https://pic2.zhimg.com/80/v2-79a82ed4f272bdf2d1cb77a514e40075_720w.jpg)

我们可以发现对于所有的分裂点 ![[公式]](https://www.zhihu.com/equation?tex=a) ，我们只要做一遍从左到右的扫描就可以枚举出所有分割的梯度和 ![[公式]](https://www.zhihu.com/equation?tex=G_L) 和 ![[公式]](https://www.zhihu.com/equation?tex=G_R) 。然后用上面的公式计算每个分割方案的分数就可以了。

观察分裂后的收益，我们会发现节点划分不一定会使得结果变好，因为我们有一个引入新叶子的惩罚项，也就是说引入的分割带来的增益如果小于一个阀值的时候，我们可以剪掉这个分割。

**2）近似算法**

贪婪算法可以的到最优解，但当数据量太大时则无法读入内存进行计算，近似算法主要针对贪婪算法这一缺点给出了近似最优解。

对于每个特征，只考察分位点可以减少计算复杂度。

该算法会首先根据特征分布的分位数提出候选划分点，然后将连续型特征映射到由这些候选点划分的桶中，然后聚合统计信息找到所有区间的最佳分裂点。

在提出候选切分点时有两种策略：

- Global：学习每棵树前就提出候选切分点，并在每次分裂时都采用这种分割；
- Local：每次分裂前将重新提出候选切分点。

直观上来看，Local 策略需要更多的计算步骤，而 Global 策略因为节点没有划分所以需要更多的候选点。

下图给出不同种分裂策略的 AUC 变换曲线，横坐标为迭代次数，纵坐标为测试集 AUC，eps 为近似算法的精度，其倒数为桶的数量。

![img](https://pic3.zhimg.com/80/v2-1da040923ad9beaf222a2dd60a8f3752_720w.jpg)

我们可以看到 Global 策略在候选点数多时（eps 小）可以和 Local 策略在候选点少时（eps 大）具有相似的精度。此外我们还发现，在 eps 取值合理的情况下，分位数策略可以获得与贪婪算法相同的精度。

![img](https://pic1.zhimg.com/80/v2-161382c979557b8bae1563a459cd1ed4_720w.jpg)

- **第一个 for 循环：**对特征 k 根据该特征分布的分位数找到切割点的候选集合 ![[公式]](https://www.zhihu.com/equation?tex=S_k%3D%5C%7Bs_%7Bk1%7D%2Cs_%7Bk2%7D%2C...%2Cs_%7Bkl%7D+%5C%7D) 。XGBoost 支持 Global 策略和 Local 策略。  
- **第二个 for 循环：**针对每个特征的候选集合，将样本映射到由该特征对应的候选点集构成的分桶区间中，即 ![[公式]](https://www.zhihu.com/equation?tex=%7Bs_%7Bk%2Cv%7D%E2%89%A5x_%7Bjk%7D%3Es_%7Bk%2Cv%E2%88%921%7D%7D) ，对每个桶统计 ![[公式]](https://www.zhihu.com/equation?tex=G%2CH+) 值，最后在这些统计量上寻找最佳分裂点。

下图给出近似算法的具体例子，以三分位为例：

![img](https://pic2.zhimg.com/80/v2-5d1dd1673419599094bf44dd4b533ba9_720w.jpg)

根据样本特征进行排序，然后基于分位数进行划分，并统计三个桶内的 ![[公式]](https://www.zhihu.com/equation?tex=G%2CH) 值，最终求解节点划分的增益。

**1.1.4 加权分位数缩略图**

事实上， XGBoost 不是简单地按照样本个数进行分位，而是以二阶导数值 ![[公式]](https://www.zhihu.com/equation?tex=h_i+) 作为样本的权重进行划分，如下：

![img](https://pic4.zhimg.com/80/v2-5f16246289eaa2a3ae72f971db198457_720w.jpg)

那么问题来了：为什么要用 ![[公式]](https://www.zhihu.com/equation?tex=h_i) 进行样本加权？

我们知道模型的目标函数为：

![[公式]](https://www.zhihu.com/equation?tex=+Obj%5E%7B%28t%29%7D+%5Capprox+%5Csum_%7Bi%3D1%7D%5En+%5Cleft%5B+g_if_t%28x_i%29+%2B+%5Cfrac12h_if_t%5E2%28x_i%29+%5Cright%5D+%2B+%5Csum_%7Bi%3D1%7D%5Et++%5COmega%28f_i%29+%5C%5C) 

我们稍作整理，便可以看出 ![[公式]](https://www.zhihu.com/equation?tex=h_i) 有对 loss 加权的作用。

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D++Obj%5E%7B%28t%29%7D+%26+%5Capprox+%5Csum_%7Bi%3D1%7D%5En+%5Cleft%5B+g_if_t%28x_i%29+%2B+%5Cfrac12h_if_t%5E2%28x_i%29+%5Cright%5D+%2B+%5Csum_%7Bi%3D1%7D%5Et++%5COmega%28f_i%29+%5C%5C+%5C%5C++++%26%3D+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5B+g_i+f_t%28x_i%29+%2B+%5Cfrac%7B1%7D%7B2%7Dh_i+f_t%5E2%28x_i%29+%5Ccolor%7Bred%7D%7B%2B+%5Cfrac%7B1%7D%7B2%7D%5Cfrac%7Bg_i%5E2%7D%7Bh_i%7D%7D%5D%2B%5COmega%28f_t%29+%5Ccolor%7Bred%7D%7B%2B+C%7D+%5C%5C++++%26%3D+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Ccolor%7Bred%7D%7B%5Cfrac%7B1%7D%7B2%7Dh_i%7D+%5Cleft%5B+f_t%28x_i%29+-+%5Cleft%28+-%5Cfrac%7Bg_i%7D%7Bh_i%7D+%5Cright%29+%5Cright%5D%5E2+%2B+%5COmega%28f_t%29+%2B+C+%5Cend%7Balign%7D+%5C%5C) 

其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7B2%7D%5Cfrac%7Bg_i%5E2%7D%7Bh_i%7D) 与 ![[公式]](https://www.zhihu.com/equation?tex=C) 皆为常数。我们可以看到 ![[公式]](https://www.zhihu.com/equation?tex=h_i) 就是平方损失函数中样本的权重。

对于样本权值相同的数据集来说，找到候选分位点已经有了解决方案（GK 算法），但是当样本权值不一样时，该如何找到候选分位点呢？（作者给出了一个 Weighted Quantile Sketch 算法，这里将不做介绍。）

**1.1.5 稀疏感知算法**

在决策树的第一篇文章中我们介绍 CART 树在应对数据缺失时的分裂策略，XGBoost 也给出了其解决方案。

XGBoost  在构建树的节点过程中只考虑非缺失值的数据遍历，而为每个节点增加了一个缺省方向，当样本相应的特征值缺失时，可以被归类到缺省方向上，最优的缺省方向可以从数据中学到。至于如何学到缺省值的分支，其实很简单，分别枚举特征缺省的样本归为左右分支后的增益，选择增益最大的枚举项即为最优缺省方向。

在构建树的过程中需要枚举特征缺失的样本，乍一看该算法的计算量增加了一倍，但其实该算法在构建树的过程中只考虑了特征未缺失的样本遍历，而特征值缺失的样本无需遍历只需直接分配到左右节点，故算法所需遍历的样本量减少，下图可以看到稀疏感知算法比 basic 算法速度块了超过 50 倍。

![img](https://pic3.zhimg.com/80/v2-e065bea4b424ea2d13b25ed2e7004aa8_720w.jpg)

### 1.2 工程实现

**1.2.1 块结构设计**

我们知道，决策树的学习最耗时的一个步骤就是在每次寻找最佳分裂点是都需要对特征的值进行排序。而 XGBoost 在训练之前对根据特征对数据进行了排序，然后保存到块结构中，并在每个块结构中都采用了稀疏矩阵存储格式（Compressed  Sparse Columns Format，CSC）进行存储，后面的训练过程中会重复地使用块结构，可以大大减小计算量。

- 每一个块结构包括一个或多个已经排序好的特征；
- 缺失特征值将不进行排序；
- 每个特征会存储指向样本梯度统计值的索引，方便计算一阶导和二阶导数值；

![img](https://picb.zhimg.com/80/v2-00c089b3439b2cdab85116d7cea511c5_720w.jpg)

这种块结构存储的特征之间相互独立，方便计算机进行并行计算。在对节点进行分裂时需要选择增益最大的特征作为分裂，这时各个特征的增益计算可以同时进行，这也是 Xgboost 能够实现分布式或者多线程计算的原因。

**1.2.2 缓存访问优化算法**

块结构的设计可以减少节点分裂时的计算量，但特征值通过索引访问样本梯度统计值的设计会导致访问操作的内存空间不连续，这样会造成缓存命中率低，从而影响到算法的效率。

为了解决缓存命中率低的问题，XGBoost 提出了缓存访问优化算法：为每个线程分配一个连续的缓存区，将需要的梯度信息存放在缓冲区中，这样就是实现了非连续空间到连续空间的转换，提高了算法效率。

此外适当调整块大小，也可以有助于缓存优化。

**1.2.3 “核外”块计算**

当数据量过大时无法将数据全部加载到内存中，只能先将无法加载到内存中的数据暂存到硬盘中，直到需要时再进行加载计算，而这种操作必然涉及到因内存与硬盘速度不同而造成的资源浪费和性能瓶颈。为了解决这个问题，XGBoost 独立一个线程专门用于从硬盘读入数据，以实现处理数据和读入数据同时进行。

此外，XGBoost 还用了两种方法来降低硬盘读写的开销：

- **块压缩：**对 Block 进行按列压缩，并在读取时进行解压；
- **块拆分：**将每个块存储到不同的磁盘中，从多个磁盘读取可以增加吞吐量。

### 1.3 优缺点

**1.3.1 优点**

1. **精度更高：**GBDT 只用到一阶泰勒展开，而 XGBoost  对损失函数进行了二阶泰勒展开。XGBoost 引入二阶导一方面是为了增加精度，另一方面也是为了能够自定义损失函数，二阶泰勒展开可以近似大量损失函数；
2. **灵活性更强：**GBDT 以 CART 作为基分类器，XGBoost 不仅支持 CART 还支持线性分类器，（使用线性分类器的 XGBoost 相当于带 L1 和 L2 正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题））。此外，XGBoost 工具支持自定义损失函数，只需函数支持一阶和二阶求导；
3. **正则化：**XGBoost 在目标函数中加入了正则项，用于控制模型的复杂度。正则项里包含了树的叶子节点个数、叶子节点权重的 L2 范式。正则项降低了模型的方差，使学习出来的模型更加简单，有助于防止过拟合；
4. **Shrinkage（缩减）：**相当于学习速率。XGBoost 在进行完一次迭代后，会将叶子节点的权重乘上该系数，主要是为了削弱每棵树的影响，让后面有更大的学习空间；
5. **列抽样：**XGBoost 借鉴了随机森林的做法，支持列抽样，不仅能降低过拟合，还能减少计算；
6. **缺失值处理：**XGBoost 采用的稀疏感知算法极大的加快了节点分裂的速度；
7. **可以并行化操作：**块结构可以很好的支持并行计算。

**1.3.2 缺点**

1. 虽然利用预排序和近似算法可以降低寻找最佳分裂点的计算量，但在节点分裂过程中仍需要遍历数据集；
2. 预排序过程的空间复杂度过高，不仅需要存储特征值，还需要存储特征对应样本的梯度统计值的索引，相当于消耗了两倍的内存。