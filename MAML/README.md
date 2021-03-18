## Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks

### Introduction

提出一种任务不确定的元学习算法来训练参数，使得少量的梯度更新得到较好的效果。

### Model-Agnostic Meta-Learning

任务被定义为

$$
\tau = \{\mathcal{L}(x_1, a_1, \cdots, x_H, a_H),~ q(x_1),~ q(x_{t + 1} \mid x_t, a_t),~ H\}
$$

* $x$：观测值；
* $a$：输出；
* $\mathcal{L}$：损失函数；
* $q(x_1)$：初始观测值的分布；
* $q(x_{t + 1} \mid x_t, a_t)$：转移分布；
* $H$：周期（episode）的长度，监督学习中 $H = 1$。

在 K 样本学习中，从 $p(\tau)$ 中采样 $\tau_i$，从 $q_i$ 中采样 K 个样本，并得到反馈 $\mathcal{L}_{\tau_i}$；而在元学习中，在得到 $\mathcal{L}_{\tau_i}$ 后还要对 $q_i$ 中新的样本进行测试，并在元学习的最后，对从 $p(\tau)$ 中采样的新任务进行测试。

显然，从任务分布 $p(\tau)$ 中比从单个任务中更容易学到可迁移的内部特征。

MAML 的核心思想是找到模型中对变化敏感的参数。形式化地，记以 $\theta$ 为参数的模型为 $f_\theta$，在适应新的任务 $\tau_i$ 时，模型的参数由 $\theta$ 变为 $\theta_i'$（可以理解为 $\theta$ 求出的是元学习模型的通用参数，而 $\theta_i'$ 求出的是每个任务的最佳参数），在该方法中有

$$
\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\tau_i} (f_\theta)
$$

更具体地，元学习的目标是

$$
\min_{\theta} \sum_{\tau_i \sim  p(\tau)} \mathcal{L}_{\tau_i}(f_{\theta_i'}) = \sum_{\tau_i \sim  p(\tau)} \mathcal{L}_{\tau_i}(f_{\theta - \alpha \nabla_\theta \mathcal{L}_{\tau_i} (f_\theta)})
$$

即最小化当前采样的任务集合上对参数 $\theta_i'$ 的损失函数值的和，而第 $i$ 个任务中学习到的模型参数 $\theta_i'$，实际上取决于参数 $\theta$。之后还要对通用参数 $\theta$ 进行更新

$$
\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\tau_i \sim p(\tau)} \mathcal{L}_{\tau_i} (f_{\theta_i'})
$$

算法伪代码如下

![1.png](1.png)

外层的循环相当于对参数 $\theta$ 求了一次二阶微分。下面是 MAML 的具体实例。

***Supervised Regression and Classification***

和上面过程类似的想法，为模型接受单一的输入和给出单一的输出，即 $H = 1$。过程中采样了两个数据集 $\mathcal{D}$ 和 $\mathcal{D}'$ 分别用来更新 $\theta_i'$ 和 $\theta$。

N-Way K-Shot 指的是 N 个不同的分类，每个分类 K 个样本的分类任务。

***Reinforcement Learning***

使用策略梯度的强化学习，和上述实例类似，采样过程改为对轨迹的采样。两个实例的算法伪代码如下。

![1.png](2.png)

实际上，$\theta$ 的每次更新相当于向一阶导数下降最快的方向更新，这使得下次更新 $\theta_i'$ 时有最好的更新效果（$\theta_i'$ 利用一阶导数值更新）。