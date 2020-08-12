之前在业务中应用了许多Faiss，也看了几篇关于Faiss的论文，简单记录下Faiss的一些属性和应用。Faiss是Facebook的AI团队开源的一套用于做聚类或者相似性搜索的软件库，底层是用C++实现。Faiss因为超级优越的性能，被广泛应用于推荐相关的业务当中。接下来分Faiss在推荐业务应用和Faiss的基本原理两部分进行介绍。

# **1** Faiss在推荐业务中的应用

在我的认知里，基本上50%以上的手机APP的推荐业务会应用到Faiss服务，可见应用之广。那Faiss究竟是在哪个模块使用呢，通过下方这个图给大家介绍：

![faiss召回](E:\面试资料\NLPer-Interview\img\faiss召回.png)

大家都知道推荐业务包含排序和召回两个模块，Faiss比较多的应用在召回模块。召回业务中有很多是向量生成类的算法，比如Graph  Embedding、ALS Embedding、FM  Embedding等。ALS就是经典的矩阵分解算法，它可以将User和Item的行为数据利用矩阵分解的方式生成User向量和Item向量，这些向量分别代表User和Item的属性（工科研究生矩阵论课程学过矩阵分解，不懂的同学要补课了）。

当我们拿到了User和Item的向量，只要计算出哪些Item和User的向量距离较短（最简单的解法是算欧式距离），就可以得出User偏爱的Item。但是当User和Item的数量巨大的时候，设想下某短视频平台，每天有上百万User登录，有存量的上千万的Item短视频，怎么能快速的计算出向量距离，就成了一个亟待解决的技术难点，因为推荐业务的召回模块需要在50ms以内拿到结果。这也就是Faiss的价值所在，Faiss几乎可以在10ms内完成百万*百万以上的向量距离计算，它是怎么实现的呢？

## PCA降维

PCA是一种降维手段，简单理解就是将高维向量变为低围，这样就可以有效的节省存储空间，PCA我之前介绍过，今天就不多说了。有兴趣可以看下我的博客：

[我的博客-PCA](https://blog.csdn.net/buptgshengod/article/details/37814597?ops_request_misc=%7B%22request%5Fid%22%3A%22158867265319724846406997%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fblog.%22%7D&request_id=158867265319724846406997&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_v2~rank_v25-1)

大家看下图绿色的点，它其实是二维的，既有纵向坐标的属性也有横向坐标的属性，可以用PCA方式让它变为一维，这样就成了红色这样的点簇、