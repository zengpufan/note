# 1-Introduction

# 2-softmax_regression
- Three ingredients of a machine learning  algorithm：
  - The hypothesis class： map the inputs to outputs
  - The loss function
  - An optimization method
- 参数假设
  - $n$为输入向量的维度
  - $k$为输出的分类数量
  - $m$为样本点个数
  - $h$为n空间向k空间的一个映射，$h(x)=\theta^T * x$ , 其中，$\theta$的维度是$(n,k)$

- softmax loss
  - compute the probability of each class

  \[
  p_i= 
  \frac  
      {exp(h_i(x))} 
      {
         \Sigma_{j=1}^{k} exp(h_j(x))
      }
  \]
  - compute the cross entropy loss
  \[
      loss
      =-log(p_i)
      =-h_i(x)+log(\Sigma_{j=1}^{k} exp(h_j(x)))
  \]
- 上式的推导过程：
  思路是将式子中的矩阵分量化为矩阵的乘积。
  设$e_i$为第i个元素为1，其余元素为0 的列向量
  故
  \[
   loss=-e_i\theta^Tx+
   log{1^Texp(\theta^Tx)}
   \]
   通过这种转化，可以将$\theta$变量提取出来，直接对$\theta$ 求导数即可。

- compute the gradient of softmax regression
   \[
      h(x)=\theta^Tx
   \]
   \[
      loss=-e_{y} h(x)+
      log{1^Texp(h(x))}
   \]
   根据矩阵乘法的的导数，$e_i$可以直接提出来
   \[
      \nabla_\theta loss=-e_y \frac{\partial h(x)}{\partial\theta}+
      \frac{\partial [log{1^Texp(h(x))]}}{\partial\theta}
   \]
   由于loss是标量，$log{1^Texp(h(x))}$也是标量，所以链式法则直接按照log的标量公式计算。
   \[
      \nabla_\theta loss=-e_y \frac{\partial h(x)}{\partial\theta}+
      \frac{\partial [1^Texp(h(x))]}{\partial\theta}*\frac{1}{1^Texp(h(x))}
   \]
   由于$exp(h(x))$是一个矩阵，而$1^Texp(h(x))$是一个标量，根据标量对矩阵的导数，可得：
   \[
      \nabla_\theta loss=-e_y \frac{\partial h(x)}{\partial\theta}+
      \frac{\partial [exp(h(x))]}{\partial\theta}*\frac{1}{1^Texp(h(x))}
   \]
   由于$exp(h(x))$是element wise的，所以其导数应该理解为每个元素的变化率组成的矩阵。
   \[
      \nabla_\theta loss=-e_y \frac{\partial h(x)}{\partial\theta}+
      \frac{\partial [h(x)]}{\partial\theta}*\frac{exp(h(x))}{1^Texp(h(x))}
   \]

   \[
      \nabla_\theta loss=
      \frac{\partial[ -e_i\theta^Tx+
   log{1^Texp(\theta^Tx)]}}{\partial \theta}
   \]
   \[
      =-\frac{\partial h_y(x)}{\partial \theta}
      +x\frac{exp(h(x))}{\Sigma_{j=1}^{k} exp(h_j(x))}
      =-\frac{\partial h_y(x)}{\partial \theta}+
      =x(z-e_y)^T   
   \]
   \[
      z=normalize(exp(\theta^Tx))
   \]

- 矩阵的导数计算
矩阵的导数计算可以使用链式法则，但是需要注意求导的对象是矩阵还是标量。
- - 矩阵对于矩阵的导数
右侧的参数取转置放到左边，左侧的直接放到右边。
   \[
      d(AXB)=B^TA
   \]
   如果X带有转置符号，则将左右的参数分别转置，但是不对调位置
   \[
      d(AX^TB)=BA^T
   \]
- - 矩阵对于标量的导数
   设$y=AX$要求解$y$关于$X$的导数。首先明确导数的是一个维度等于$X$的张量。张量中第$(i,j)$位置的数值为
   \[
      \frac{\partial y}{\partial x_{ij}}
   \]
   计算的时候只需要将y的构成公式列出来，分析然后对每个$x_{ij}$求偏导数，并找规律得出通式。
   例如，$y=tr(AX)$的公式为：
   \[
      y=\Sigma_{i=1}^{i=n}\Sigma_{j=1}^{j=m}a_{ij}x_{ij}
      \]
   故$y$的导数为A

- update weights
assume that the learning rate is $\alpha$
update the weights:
\[
   \theta=theta-\frac{\alpha}{BatchSize} * gradient
   \]




# 第三节：“Manual” Neural Networks



# 第四节：“Manual” Neural Networks
## 1. 导数的计算方法
- **前向差分（Forward Difference）**:
   \[
   f'(x) \approx \frac{f(x+h) - f(x)}{h}
   \]

- **后向差分（Backward Difference）**:
   \[
   f'(x) \approx \frac{f(x) - f(x-h)}{h}
   \]

- **中心差分（Central Difference）**:
   \[
   f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}
   \]
## 2. 前向自动微分和反向自动微分
![](./deep_learning_system/reverse_mode_automatic_differentation.png)
如上图所示，自动微分首先将函数转换成一个计算图。然后通过链式法则计算每个点的微分。
前向自动微分是从左向右逐个变量计算对于x的微分。
反向自动微分是从右向左逐个变量计算对于y的微分。

# 5 - Automatic Differentiation Implementation

# Homework 0
- Question 1

- Question 2
- Question 3
- Question 4
- Question 5



