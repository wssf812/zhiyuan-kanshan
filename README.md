# 智源 - 看山杯 专家发现算法大赛 2019-传统方法

任务：比赛提供知乎的问题信息、用户画像、用户回答记录，以及用户接受邀请的记录，要求选手预测这个用户是否会接受某个新问题的邀请。

模型：传统机器学习模型--LR(逻辑回归)。

输入特征：性别和对应问题（one-hot和tfidf编码数据）

输出：有多大的概率接受邀请   

比赛排行榜

![image](https://github.com/user-attachments/assets/03c131a9-9cbb-4f6e-bcc5-fd3344a13271)

本代码很简单，就是一个传统方法，在比赛中达到0.6分。因为输入了一部分训练数据。要是整个输入，应该可以达到0.7.可以当作baseline玩玩。
在总体深度学习，bert等预训练模型统治的时代。尝试传统方法，练练手也不错。(这个任务中bert还真用不了，因为数据都是脱敏的)<br>
(代码中只给出了部分数据，详细数据请去官网下载:https://www.biendata.net/competition/zhihu2019)

