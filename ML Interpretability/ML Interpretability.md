# 机器学习可解释性 (ML Interpretability) 

## 目录

1. [课程概览与前置知识](#1-课程概览与前置知识)
2. [什么是机器学习可解释性](#2-什么是机器学习可解释性)
3. [深度学习基础回顾：计算图与前向/反向传播](#3-深度学习基础回顾计算图与前向反向传播)
4. [对抗样本：如何欺骗一个分类器](#4-对抗样本如何欺骗一个分类器)
5. [可解释性引擎：看懂视觉模型](#5-可解释性引擎看懂视觉模型)
6. [特征可视化：探索神经元的“视觉偏好”](#6-特征可视化探索神经元的视觉偏好)
7. [语言模型的可解释性](#7-语言模型的可解释性)

---

## 1. 课程概览与前置知识

### Slide 1：课程来源与协议

本讲义的原始材料是一套名为 *ML Interpretability* 的 slide deck，由 **Umar Jamil** 制作并开源在 GitHub 上。该材料采用 **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** 协议发布，仅供学习和非商业用途。


### Slide 2：Topics 与 Prerequisites

#### 1.2.1 本课程涵盖的主题

整份讲义共 39 页，以**视觉模型**为主线，延伸到**语言模型**，系统性地讲解深度学习可解释性，涵盖以下板块：

1. **ML Interpretability 的定义与动机** — 我们为什么需要理解模型。
2. **深度学习与反向传播回顾** — 计算图、前向传播、反向传播，这是所有后续技术的数学基础。
3. **对抗样本（Adversarial Examples）** — 如何通过微小扰动欺骗一个训练良好的分类器。
4. **视觉模型的可解释性引擎** — Leap Labs 的 Engine：原型生成、类别纠缠、特征隔离。
5. **特征可视化（Feature Visualization）** — 如何“画出”神经元爱看的图案。
6. **语言模型的可解释性** — 如何在离散 token 的世界中寻找解释。

#### 1.2.2 前置知识

在学习本讲义之前，你需要具备：
- **微积分基础**：知道导数是什么，理解**链式法则（Chain Rule）**。
- **深度学习基础**：了解神经网络的基本结构（层、权重、偏置、激活函数）。
- **PyTorch 基础**（非必需但强烈推荐）：了解 `autograd`、`backward()`、`requires_grad` 等概念。


---

## 2. 什么是机器学习可解释性

### 2.1 一个警醒世人的真实案例：Tesla 自动驾驶事故

2016 年，美国佛罗里达州发生了一起震惊世界的致命车祸：一辆开启 Autopilot 模式的 Tesla Model S 以 74 英里/小时的高速，径直撞上了一辆正在横穿高速公路的白色 18 轮卡车挂车，驾驶员当场死亡。

根据《卫报》的报道和 Tesla 的事故说明：

> “在明亮的春日天空背景下，车辆的传感器系统未能识别出那辆大型白色卡车挂车。自动驾驶系统试图从挂车下方全速穿过，导致挂车底部撞击了 Model S 的挡风玻璃。”

这起事故的悲剧性在于：**模型做出了一个灾难性的错误决策，而在那之前，没有人能解释它为什么会“看不见”那辆卡车。** 如果研发团队能够提前理解模型在何种光照和背景下会失效，也许就能在部署前修复这个问题。

### 2.2 可解释性试图回答的核心问题

机器学习可解释性（ML Interpretability）本质上是一门研究**“如何理解模型决策过程”**的学问。它试图回答两个根本问题：

1. **What did the model learn?**  
   模型到底学到了什么？它内部的表征（representations）捕捉了数据中的哪些规律？这些规律是人类可以理解的吗？
2. **What features/patterns from the input make the model generate certain outputs?**  
   输入中的哪些特征或模式，使得模型产生了特定的输出？

### 2.3 可解释性的四大价值

| 价值维度 | 详细说明 |
|----------|----------|
| **Debug & Tune（调试与调优）** | 当模型表现不佳时，可解释性帮助我们定位问题。例如，模型是否过度依赖某个伪相关特征？超参数的选择是否导致了不健康的学习动态？ |
| **识别失效模式（Identify Failure Modes）** | 在模型部署到真实世界之前，通过可解释性工具提前发现它可能在哪些场景下出错（如对特定背景、光照、纹理敏感），从而进行针对性地增强训练。 |
| **建立信任（Building Trust）** | 在医疗、自动驾驶、金融等高风险领域，用户和监管者需要“看到”模型是可靠且经过良好训练的，才会放心采纳。黑盒模型即使准确率很高，也缺乏说服力。 |
| **发现新知识（Novel Insights）** | 模型可能从海量数据中捕捉到人类专家尚未注意到的模式。可解释性是把这些“机器洞察”翻译给人类的桥梁。例如，一个预测蛋白质结构的模型可能揭示出两种蛋白质之间出人意料的折叠相似性。 |

### 2.4 可解释性的分类维度

在学术文献中，可解释性可以从多个维度进行分类：

#### 2.4.1 Intrinsic vs. Post-hoc
- **Intrinsic（内在可解释性）**：模型本身结构简单、透明，人类可以直接阅读其参数。例如：线性回归、逻辑回归、决策树。
- **Post-hoc（事后解释性）**：模型本身是黑盒（如深度神经网络），需要借助外部工具生成解释。例如：Grad-CAM、LIME、SHAP。

#### 2.4.2 Local vs. Global
- **Local（局部可解释性）**：解释模型在**单个样本**上的预测原因。例如：为什么模型把这张图片预测为“猫”？
- **Global（全局可解释性）**：解释模型整体的**学习行为和数据偏好**。例如：模型认为“理想的猫”长什么样？

本讲义中的技术既包含 Local 方法（如 Saliency Map、FGSM），也包含 Global 方法（如 Feature Visualization、Prototype Generation）。

### 2.5 补充：Interpretability vs. Explainability

这两个词有时被区分使用：
- **Interpretability（可解释性）**：强调模型**本身**的透明度和内在可理解性。
- **Explainability（可说明性）**：强调为**黑盒**模型生成**事后解释（post-hoc explanations）**。

个人认为，传统机器学习注重的是可解释性，

在本讲义中，我们将这两个概念统称为“可解释性”，因为它们的目标一致：**让人类理解模型**。


---

## 3. 深度学习基础回顾：计算图与前向/反向传播

在深入可解释性技术之前，我们必须牢固掌握深度学习中的**计算图（Computational Graph）**和**反向传播（Backpropagation）**。因为后面要讲的几乎所有可解释性方法——无论是对抗样本、特征可视化，还是语言模型中的 embedding 优化——本质上都依赖于**“计算梯度并反向传播到输入”**这一核心能力。

### Deep Learning 与神经网络

深度学习是一类基于**人工神经网络（Artificial Neural Networks）**的机器学习算法。神经网络通过**逐层（layer by layer）**的非线性变换，从原始输入中自动提取越来越高层次的特征。

常见的神经网络架构包括：
- **卷积神经网络（CNN / ConvNet）**：擅长处理具有空间结构的数据，尤其是图像。
- **循环神经网络（RNN）**：擅长处理序列数据，如时间序列和自然语言。
- **Transformer**：基于自注意力机制（Self-Attention），是现代大语言模型（GPT、BERT 等）和视觉 Transformer（ViT）的核心架构。

框架如 PyTorch 和 TensorFlow 会将我们定义的模型自动转换为**计算图（Computational Graph）**。计算图是一个有向无环图（DAG），其中：
- **节点（Nodes）**表示数学运算（如乘法、加法、激活函数）。
- **边（Edges）**表示数据（张量）的流动。

这种表示方式使得计算机能够**自动求导（Automatic Differentiation）**，这是训练深度网络的关键。

![Deep Learning](slides_images/page_04.png)
*图：一个多层感知机（MLP）的结构示意图。输入层（黄色）连接到两个隐藏层（蓝色），再到输出层（黄色）。这种分层结构是深度学习的基础。*

### 计算图（Computational Graph）

为了理解计算图是如何工作的，我们来看一个**极简的房价预测网络**：用卧室数量 $x_1$ 和卫生间数量 $x_2$ 来预测房价。

#### 3.2.1 网络结构

- **输入层**：2 个神经元，对应 $x_1$ 和 $x_2$。
- **隐藏层（Layer 1）**：2 个神经元，激活函数为 ReLU。
- **输出层**：1 个神经元，直接输出预测价格。

每个隐藏层神经元 $j$ 的计算公式为：

$$
\text{output}_j = \text{ReLU}\left( \sum_{i} x_i w_{ji} + b_j \right) = \text{ReLU}(x_1 w_{j1} + x_2 w_{j2} + b_j)
$$

其中 $w_{ji}$ 是权重（weight），$b_j$ 是偏置（bias），ReLU 是激活函数。

![Computational Graph (1)](slides_images/page_05.png)
*图：房价预测网络的计算图，以及右侧对应的 PyTorch 代码实现 `SimpleNet`。该网络包含 `nn.Linear(2, 2)`（输入→隐藏层）和 `nn.Linear(2, 1)`（隐藏层→输出），中间插入 `torch.relu` 激活函数。*

#### 3.2.2 PyTorch 代码逐行解析

```python
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 2)   # 输入层 → 隐藏层（2个神经元）
        self.fc2 = nn.Linear(2, 1)   # 隐藏层 → 输出层（1个神经元）

    def forward(self, x):
        l1 = torch.relu(self.fc1(x)) # Layer 1：线性变换 + ReLU
        o = self.fc2(l1)             # 输出层：线性变换
        return o
```

- `nn.Linear(2, 2)` 会自动创建权重矩阵 $W \in \mathbb{R}^{2 \times 2}$ 和偏置向量 $b \in \mathbb{R}^2$。
- `torch.relu(...)` 对线性输出逐元素应用 ReLU 函数。
- 第二层没有显式激活函数（对于回归任务，输出层通常不加激活）。

#### 3.2.3 ReLU 激活函数

ReLU（Rectified Linear Unit）定义为：

$$
\text{ReLU}(x) = \max(0, x)
$$

它的导数非常简单：
- 当 $x > 0$ 时，$\frac{d}{dx}\text{ReLU}(x) = 1$
- 当 $x < 0$ 时，$\frac{d}{dx}\text{ReLU}(x) = 0$
- 当 $x = 0$ 时，通常在实践中定义为 0（PyTorch 中的默认行为）。

ReLU 引入非线性的同时计算非常高效，因此成为现代深度学习中最常用的激活函数。**如果没有激活函数，无论堆叠多少层，整个网络都等价于一个单层线性变换**，无法学习复杂的模式。

#### 3.2.4 展开的计算图

我们可以将上述网络进一步展开为具体的中间变量：

- $a_1 = x_1 w_{11} + x_2 w_{12}$
- $a_2 = x_1 w_{21} + x_2 w_{22}$
- $a_3 = \text{ReLU}(a_1 + b_1)$
- $a_4 = \text{ReLU}(a_2 + b_2)$
- $a_5 = a_3 w_{31} + a_4 w_{32}$
- $a_6 = a_5 + b_3$（最终输出）

![Computational Graph (2)](slides_images/page_06.png)
*图：更详细的计算图。每个椭圆节点代表一个中间变量（$a_1$ 到 $a_6$），展示了从输入到输出的完整数据流动。这种展开式的表示有助于我们手动追踪梯度。*

> **补充：为什么 PyTorch 需要计算图？**  
> 在训练神经网络时，我们需要计算损失函数对每个参数的梯度。对于复杂的网络（可能有数百万甚至数十亿个参数），手动写出所有偏导数是不可能的。计算图使得框架可以通过**反向模式自动微分（Reverse-Mode Automatic Differentiation）**高效地完成这件事。其核心思想就是链式法则：从输出节点开始，沿着图的边反向传播梯度信号，每个节点只关心自己的局部导数。

### Slide 7：训练神经网络的目标

通常我们拥有一个数据集，由成对的 **(input, target)** 组成。我们的目标是训练神经网络，使其最小化某个**损失函数（Loss Function）**。例如，对于回归任务（如房价预测），常用的损失函数是 **均方误差（Mean Squared Error, MSE）**：

$$
L_{\text{MSE}} = (\text{output} - \text{target})^2 = (a_6 - y)^2
$$

对于分类任务（如图像分类），常用的损失函数是 **交叉熵损失（Cross-Entropy Loss）**。假设模型输出 logits $z = [z_1, z_2, \dots, z_K]$，经过 softmax 后得到概率 $p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$。真实标签为 one-hot 向量 $y$（第 $c$ 类为 1，其余为 0），则：

$$
L_{\text{CE}} = -\sum_{i=1}^{K} y_i \log(p_i) = -\log(p_c) = -\log\left( \frac{e^{z_c}}{\sum_j e^{z_j}} \right)
$$

我们的希望是：神经网络不仅能记住训练集上的输入输出映射，还能**泛化（generalize）**到从未见过的新输入上。

实际训练步骤如下：
1. 选择一个损失函数（如 MSE 或 CrossEntropy）。
2. 将输入传入网络，计算输出和损失。
3. 计算损失对网络参数的梯度。
4. 沿着梯度的反方向更新参数，以减小损失。

![Backpropagation (1)](slides_images/page_07.png)
*图：训练流程示意图。输入经过隐藏层到达输出，计算与目标值之间的 MSE 损失。反向传播将损失信号带回网络，用于更新权重和偏置。*

### 前向传播（Forward Pass）

前向传播是指数据从输入层流向输出层的过程。我们通过一个**完整的数值例子**来手动走一遍：

#### 3.4.1 给定参数

假设输入为 $x_1 = 2, x_2 = 4$，目标值为 $y = 5$。网络参数如下：
- $w_{11} = 0.24, w_{12} = 0.29, b_1 = -0.70$
- $w_{21} = 0.62, w_{22} = -0.09, b_2 = -0.33$
- $w_{31} = 0.66, w_{32} = 0.70, b_3 = 0.29$

#### 3.4.2 逐步计算

| 步骤 | 公式 | 数值计算 | 结果 |
|------|------|----------|------|
| 1 | $a_1 = x_1 w_{11} + x_2 w_{12}$ | $2 \times 0.24 + 4 \times 0.29$ | **1.639** |
| 2 | $a_2 = x_1 w_{21} + x_2 w_{22}$ | $2 \times 0.62 + 4 \times (-0.09)$ | **0.879** |
| 3 | $a_3 = \text{ReLU}(a_1 + b_1)$ | $\text{ReLU}(1.639 - 0.70)$ | **0.939** |
| 4 | $a_4 = \text{ReLU}(a_2 + b_2)$ | $\text{ReLU}(0.879 - 0.33)$ | **0.549** |
| 5 | $a_5 = a_3 w_{31} + a_4 w_{32}$ | $0.939 \times 0.66 + 0.549 \times 0.70$ | **1.005** |
| 6 | $a_6 = a_5 + b_3$ | $1.005 + 0.29$ | **1.295** |
| 7 | $L = (a_6 - y)^2$ | $(1.295 - 5)^2$ | **13.724** |

![Forward Pass Numerical Example](slides_images/page_08.png)
*图：完整的前向传播数值示例。每个中间变量旁边都标有计算结果（红色数字），最终损失为 13.724。这张图是理解后续反向传播计算的重要基础。*

### Slides 9–10：反向传播（Backward Pass）与参数更新

反向传播的核心任务是：**计算损失函数 $L$ 对每个参数（权重 $w$ 和偏置 $b$）的偏导数**。这完全依赖于**链式法则（Chain Rule）**。

#### 3.5.1 链式法则

假设我们想计算 $L$ 对 $w_{11}$ 的梯度。从计算图中可以看出，$w_{11}$ 到 $L$ 的路径是：

$$
w_{11} \rightarrow a_1 \rightarrow a_3 \rightarrow a_5 \rightarrow a_6 \rightarrow L
$$

根据链式法则：

$$
\frac{\partial L}{\partial w_{11}} = \frac{\partial L}{\partial a_6} \cdot \frac{\partial a_6}{\partial a_5} \cdot \frac{\partial a_5}{\partial a_3} \cdot \frac{\partial a_3}{\partial a_1} \cdot \frac{\partial a_1}{\partial w_{11}}
$$

![Backward Pass Chain Rule](slides_images/page_09.png)
*图：反向传播链式法则示意图。为了计算 $\partial L / \partial w_{11}$，梯度信号需要沿着红色路径从右向左传播，经过 $a_6 \rightarrow a_5 \rightarrow a_3 \rightarrow a_1 \rightarrow w_{11}$。*

#### 3.5.2 各局部导数的推导（基于 MSE 损失）

使用标准 MSE 损失 $L = (a_6 - y)^2$：

1. **输出层梯度起点**：
   $$
   \frac{\partial L}{\partial a_6} = 2(a_6 - y) = 2(1.295 - 5) = -7.41
   $$

2. **$a_6 \rightarrow a_5$**：
   因为 $a_6 = a_5 + b_3$，所以 $\frac{\partial a_6}{\partial a_5} = 1$，$\frac{\partial a_6}{\partial b_3} = 1$。

3. **$a_5 \rightarrow a_3$ 和 $a_4$**：
   因为 $a_5 = a_3 w_{31} + a_4 w_{32}$，所以：
   - $\frac{\partial a_5}{\partial a_3} = w_{31} = 0.66$
   - $\frac{\partial a_5}{\partial a_4} = w_{32} = 0.70$
   - $\frac{\partial a_5}{\partial w_{31}} = a_3 = 0.939$
   - $\frac{\partial a_5}{\partial w_{32}} = a_4 = 0.549$

4. **$a_3 \rightarrow a_1$（ReLU 的导数）**：
   因为 $a_3 = \text{ReLU}(a_1 + b_1)$，且 $a_1 + b_1 = 0.939 > 0$，所以：
   $$
   \frac{\partial a_3}{\partial a_1} = 1, \quad \frac{\partial a_3}{\partial b_1} = 1
   $$
   同理，$a_2 + b_2 = 0.549 > 0$，所以 $\frac{\partial a_4}{\partial a_2} = 1$，$\frac{\partial a_4}{\partial b_2} = 1$。

5. **$a_1 \rightarrow w_{11}$**：
   因为 $a_1 = x_1 w_{11} + x_2 w_{12}$，所以：
   - $\frac{\partial a_1}{\partial w_{11}} = x_1 = 2$
   - $\frac{\partial a_1}{\partial w_{12}} = x_2 = 4$
   同理：
   - $\frac{\partial a_2}{\partial w_{21}} = x_1 = 2$
   - $\frac{\partial a_2}{\partial w_{22}} = x_2 = 4$

#### 3.5.3 手动计算所有参数的梯度

现在我们可以计算每个参数的完整梯度：

**输出层参数：**
- $\frac{\partial L}{\partial w_{31}} = (-7.41) \times 1 \times 0.939 = -6.95$
- $\frac{\partial L}{\partial w_{32}} = (-7.41) \times 1 \times 0.549 = -4.07$
- $\frac{\partial L}{\partial b_3} = (-7.41) \times 1 \times 1 = -7.41$

**隐藏层参数（以上方神经元为例，通过 $a_3$）：**
- $\frac{\partial L}{\partial w_{11}} = (-7.41) \times 1 \times 0.66 \times 1 \times 2 = -9.78$
- $\frac{\partial L}{\partial w_{12}} = (-7.41) \times 1 \times 0.66 \times 1 \times 4 = -19.56$
- $\frac{\partial L}{\partial b_1} = (-7.41) \times 1 \times 0.66 \times 1 \times 1 = -4.89$

**隐藏层参数（以下方神经元为例，通过 $a_4$）：**
- $\frac{\partial L}{\partial w_{21}} = (-7.41) \times 1 \times 0.70 \times 1 \times 2 = -10.37$
- $\frac{\partial L}{\partial w_{22}} = (-7.41) \times 1 \times 0.70 \times 1 \times 4 = -20.75$
- $\frac{\partial L}{\partial b_2} = (-7.41) \times 1 \times 0.70 \times 1 \times 1 = -5.19$

> **注意**：如果 $a_1 + b_1 < 0$，那么 ReLU 的导数为 0，导致上游所有梯度都变为 0。这种现象被称为 **Dying ReLU**。

#### 3.5.4 补充：Softmax 与 CrossEntropy 的梯度

在后续对抗样本章节中，我们需要 CrossEntropy 的梯度。这里提前推导一下。

Softmax 函数：
$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Softmax 对 logit $z_k$ 的导数为：
$$
\frac{\partial p_i}{\partial z_k} = p_i (\delta_{ik} - p_k)
$$

其中 $\delta_{ik}$ 是 Kronecker delta（当 $i=k$ 时为 1，否则为 0）。

CrossEntropy 损失 $L = -\log(p_c)$ 对 $z_k$ 的导数为：
$$
\frac{\partial L}{\partial z_k} = p_k - y_k
$$

这是一个非常优雅的结果！**CrossEntropy + Softmax 的梯度，就是预测概率与真实标签之间的差值。** 这也解释了为什么分类任务中 CrossEntropy 比 MSE 更受欢迎：它的梯度形式简洁，且在训练初期即使预测很差，梯度也不会像 MSE 那样趋于饱和。

#### 3.5.5 参数更新规则

计算出所有参数的梯度后，用**梯度下降法（Gradient Descent）**更新：

$$
w_{ij}^{\text{new}} = w_{ij}^{\text{old}} - \alpha \cdot \frac{\partial L}{\partial w_{ij}}
$$

其中 $\alpha > 0$ 是**学习率（Learning Rate）**。

**为什么要设置学习率？**
因为在实际训练中，我们通常使用的是**随机梯度下降（Stochastic Gradient Descent, SGD）**。每次梯度计算只基于一个 mini-batch（小批量样本），而不是整个数据集。因此，计算出的梯度方向只是真实总体梯度的一个**有噪声的近似**。如果步子太大，可能会在损失函数的山谷之间震荡；如果步子太小，收敛极慢。

![Parameter Update](slides_images/page_10.png)
*图：参数更新示意图。$w_{11}$ 梯度为负（如 -9.78），因此更新为 $w_{11} - \alpha \times (-9.78)$，即增大 $w_{11}$ 来减小损失。*

> **补充：现代优化器**  
> 除了最基础的 SGD，现代深度学习更常用 **Adam**、**AdamW**、**RMSprop**。它们通过维护梯度的一阶矩（均值）和二阶矩（方差）来加速收敛并稳定训练。

---

## 4. 对抗样本：如何欺骗一个分类器

### 分类器的结构与输出

分类器（Classifier）是一类特殊的神经网络，任务是将输入分配到预定义的类别之一。例如，ResNet 可以将 ImageNet 图像分类到 1000 个类别中。

分类器的最后一层输出称为 **logits**（未归一化的原始分数）。每个 logit 对应一个类别。经过 **softmax 函数**后，logits 转换为概率分布：

$$
P(y_i \mid x) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

其中 $z_i$ 是第 $i$ 类的 logit。概率最高的类别是模型的预测结果，其 softmax 值代表模型的置信度。

![Classifier Architecture](slides_images/page_11.png)
*图：分类器的通用结构。输入图像经过两个隐藏层后，输出 5 个类别的 logits（Fish, Dog, Volcano, Car, Pencil）。*

### “欺骗”分类器意味着什么？

假设有一张鱼的图片。正常训练好的分类器会高概率预测为“Fish”。但如果我们能添加**极其微小的、人眼不可见的噪声**，却让模型以 95% 的置信度将其预测为“Volcano”，这就称为**对抗样本（Adversarial Example）**。

| 状态 | 输入 | 模型预测 |
|------|------|----------|
| 正常 | 原始鱼图 | Fish 95% |
| 对抗 | 鱼图 + 微小噪声 | Volcano 95% |

![Tricking a Classifier (1)](slides_images/page_12.png)
*图：正常情况。鱼的图片被正确分类为 Fish（95%）。*

![Tricking a Classifier (2)](slides_images/page_13.png)
*图：对抗攻击后。同一张鱼的图片被高置信度分类为 Volcano（95%）。*

![Side-to-side Comparison](slides_images/page_14.png)
*图：原始图像（左）与对抗样本（右）并排对比。肉眼几乎看不出差别。*

#### 4.2.1 对抗样本的数学本质

设原始输入为 $x$，真实标签为 $y_{\text{true}}$，目标攻击标签为 $y_{\text{target}}$。对抗样本 $x^{\text{adv}}$ 需要满足两个条件：

1. **不可察觉性**：$\|x^{\text{adv}} - x\|_p \leq \epsilon$（通常 $p = \infty$，即每个像素的变化不超过 $\epsilon$）。
2. **攻击成功**：$f(x^{\text{adv}}) = y_{\text{target}}$，其中 $f$ 是分类器。

### 分类器的训练过程

分类器的训练：
1. 提供带标签的训练图像。
2. 模型前向传播，输出 logits，计算 **交叉熵损失（Cross-Entropy Loss）**：
   $$
   L_{\text{CE}} = -\sum_{i} y_i \log(P(y_i \mid x))
   $$
3. 反向传播计算损失对每个权重的梯度。
4. 梯度下降更新权重。

![How a Classifier Works](slides_images/page_15.png)
*图：分类器训练流程。输入图像与真实标签一同送入网络，计算 CrossEntropy Loss，反向传播更新权重。*

### 关键洞察——将梯度反向传播到输入

在训练过程中，我们计算损失对**参数**的梯度。但神经网络对**输入**也是可微的！只要设置 `input.requires_grad = True`，PyTorch 就能计算：

$$
\nabla_x L = \frac{\partial L}{\partial x}
$$

这个梯度告诉我们：**为了改变损失，应该如何调整输入图像的每一个像素。**

#### 4.4.1 生成对抗样本的策略

1. 冻结模型权重。
2. 选择目标类别（如 Volcano），以该类别为标签计算交叉熵损失。
3. 计算损失对输入图像的梯度 $\nabla_x L$。
4. 沿着**负梯度方向**微调输入图像（让目标类别的损失变小，即 logit 变大）：

$$
x^{\text{adv}} = x - \epsilon \cdot \text{sign}(\nabla_x L)
$$

这就是 **FGSM（Fast Gradient Sign Method）**。

![Backpropagate to Input](slides_images/page_16.png)
*图：核心提问——如果我们将梯度反向传播到输入图像会怎样？*

![Gradient w.r.t Input](slides_images/page_17.png)
*图：损失对输入的梯度指示了改变输入以使特定 logit（如 Volcano）增加的方向。*

### Slide 18：更完整的梯度视角

Slide 18 进一步用一张全网络图展示了梯度从输出层（CrossEntropy Loss）一直反向传播回输入层的过程。图中每个节点代表一个神经元，连线上的信号即为梯度流。核心洞察在于：**只要我们设定 `input.requires_grad = True`，PyTorch 就会自动追踪输入图像中每个像素对最终损失的贡献。** 这意味着，除了更新权重，我们还可以用完全相同的反向传播机制来“更新输入图像”，从而生成对抗样本。

![Tricking a Classifier (3)](slides_images/page_18.png)
*图：从输入到输出的全网络梯度传播示意图。CrossEntropy 损失对输入图像的梯度，沿着网络路径逐层反向传递，最终到达输入像素。*

### Slide 19：代码实现

核心代码逻辑：

```python
import torch
import torch.nn as nn

# 1. 告诉 PyTorch 计算输入的梯度
input_data.requires_grad = True

# 2. 前向传播 + 计算损失
output = model(input_data)
loss = criterion(output, target_class)  # target_class = "Volcano"

# 3. 反向传播
loss.backward()

# 4. 生成对抗样本（注意负号：沿着负梯度方向移动）
x_adv = input_data - epsilon * torch.sign(input_data.grad)
```

![Adversarial Code](slides_images/page_19.png)
*图：PyTorch 中生成对抗样本的关键代码。`requires_grad=True` 和最后的负号是核心。*

#### 4.5.1 FGSM 的完整数学形式

FGSM 由 Ian Goodfellow 等人于 2014 年提出：

$$
x^{\text{adv}} = x - \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y_{\text{target}}))
$$

- 单步计算，非常快（"Fast"）。
- $\epsilon$ 通常在 $[0, 255]$ 的像素空间中取 $1 \sim 8$。
- 更强大的攻击 **PGD（Projected Gradient Descent）**会在 FGSM 基础上进行多步迭代，并将扰动投影回允许的 $\epsilon$-ball 内。

#### 4.5.2 补充：PGD 攻击与对抗训练

**PGD（Projected Gradient Descent）**是目前最常用的强攻击方法之一。以下代码展示的是**目标攻击（targeted attack）**版本，即不断向着 ${target}$ 的 logit 增大方向优化：

```python
x_adv = x
for i in range(num_steps):
    x_adv.requires_grad = True
    loss = criterion(model(x_adv), y_target)
    loss.backward()
    
    # 多步小步长：沿负梯度方向移动以最小化目标类别的损失
    x_adv = x_adv - alpha * torch.sign(x_adv.grad)
    
    # 投影回 epsilon-ball 内，并裁剪到合法像素范围 [0, 1]
    x_adv = torch.clamp(x_adv, min=x - epsilon, max=x + epsilon)
    x_adv = torch.clamp(x_adv, 0, 1).detach()
```

**对抗训练（Adversarial Training）**是最有效的防御方法之一。它的核心思想是：在训练过程中，不仅使用原始图像，还使用 PGD 生成的对抗样本一起训练。这样模型就能学习到对对抗扰动更鲁棒的特征。

### Et voilà！模型被欺骗了

通过上述方法，一个在大规模数据集上训练好的分类器，就能把一条鱼 confidently 地识别为火山。

![Et voila](slides_images/page_20.png)
*图：成功欺骗分类器。这个例子揭示了模型可能依赖与人类直觉完全不同的视觉模式。*

### 从对抗样本中学到了什么？

对抗样本的存在深刻揭示了：

> **模型关注的图像模式，可能和人类观察到的语义内容完全不同。**

1. **Understand what your model has learned** — 通过分析使模型出错的微小扰动，我们可以反向推断出模型真正依赖的决策边界。
2. **Enable targeted fine-tuning** — 如果知道模型混淆了哪些类别（例如在特定背景下把卡车误识别为天空），就可以针对性地收集增强数据，进行精确修复。
3. **Predict behaviour on unseen data** — 理解失效模式后，我们能在模型部署前预判它在真实世界中的表现，提前规避安全风险。


---

## 5. 可解释性引擎：看懂视觉模型

从对抗样本的讨论中，我们看到了一个令人不安的事实：模型可能在利用人类无法察觉的奇怪模式。为了系统性地理解和分析视觉模型，我们需要更强大的工具。本节介绍 **Leap Labs 的 Interpretability Engine**，它为任意视觉模型提供了三类核心能力：
1. **洞察模型学到了什么（Prototype Generation）**
2. **识别模型在哪里混淆（Class Entanglement）**
3. **解释单个预测的依据（Feature Isolation）**

### 洞察学习成果——原型生成（Prototypes）

**原型生成**旨在回答这个问题：**模型眼中的“某样东西”长什么样？**

例如，问一个图像分类器：你心目中的 pancake 应该是什么样的？Interpretability Engine 会生成一系列图像，这些图像是模型内部对该类别的**理想化视觉表征**，不一定来自真实数据集。

这背后的数学原理与对抗样本和特征可视化相同：**通过梯度上升优化输入图像，使其最大化目标类别的 logit 或某个中间层的激活**。但 Leap Labs 使用了更先进的正则化方法，因此生成的原型图像比简单的 activation maximization 更加自然。

![Interpretability Engine - Prototypes](slides_images/page_22.png)
*图：Leap Labs Interpretability Engine 生成的 pancake 原型。这些图像展示了模型心中“理想 pancake”的视觉特征——可能与人类印象相似，也可能包含意想不到的模式。*

#### 5.1.1 原型生成的算法步骤

原型生成的完整流程可以概括为：

1. **选择目标类别 $c$**：例如 "pancake"。
2. **初始化输入**：通常从一张随机噪声图或某种先验分布中采样。
3. **定义目标函数**：
   $$
   \mathcal{O}(x) = \text{logit}_c(x) + \lambda_1 R_{\text{smooth}}(x) + \lambda_2 R_{\text{robust}}(x) + \dots
   $$
   其中 $\text{logit}_c(x)$ 是模型对类别 $c$ 的预测分数，后面的各项是各种正则化惩罚（如 TV、变换鲁棒性、多样性约束等）。
4. **梯度上升优化**：
   $$
   x_{t+1} = x_t + \alpha \nabla_x \mathcal{O}(x_t)
   $$
5. **后处理与选择**：由于优化可能存在多个局部最优，通常会生成一批原型，然后根据其自然度、多样性等指标进行筛选。

> **补充：数据独立可解释性（Data Independent Interpretability）**  
> 传统可解释性方法（如 LIME、SHAP）通常需要依赖真实数据样本来生成解释。而原型生成是一种**数据独立**的方法：它不需要查看任何真实图像，就能直接从模型内部提取出该类别的“理想图像”。这在隐私敏感或数据稀缺的场景下尤其有价值。
> 原型生成是 **Model Inversion** / **Activation Maximization** 的高级应用。早期 activation maximization 往往生成充满高频噪声的图像。Leap Labs 结合**生成模型先验**、**变换鲁棒性**、**频率惩罚**等正则化技术，使结果更自然。

### 识别混淆——类别纠缠（Class Entanglement）

**类别纠缠（Entanglement）**描述了不同类别之间共享视觉特征的程度。

- **Expected Entanglement**：Cheesecake 和 Apple Pie 都是圆形、金黄色，模型在两者间混淆是合理的。
- **Unexpected Entanglement**：如果 Cheesecake 和 Dog 之间也有高纠缠，这就是一个**红旗（Red Flag）**。它暗示模型可能在学习某种**伪相关（Spurious Correlation）**，比如特定的背景纹理或数据标注偏差。

#### 5.2.1 纠缠度的定量刻画

在实践中，我们可以通过以下方式量化类别 $i$ 和 $j$ 之间的纠缠度：

1. **基于原型的相似度**：计算类别 $i$ 和类别 $j$ 的原型图像在特征空间中的余弦相似度：
   $$
   E_{ij} = \frac{\phi(p_i) \cdot \phi(p_j)}{\|\phi(p_i)\| \|\phi(p_j)\|}
   $$
   其中 $\phi(\cdot)$ 表示模型某个中间层的特征提取器。

2. **基于混淆矩阵**：统计测试集中类别 $i$ 被误分为 $j$ 的频率，或两者 softmax 概率分布的 KL 散度。

3. **基于共享特征激活**：通过特征隔离技术，计算两个类别的原型中共同被激活的神经元比例。

通过分析纠缠度，我们可以：
- 识别类别边界的模糊区域。
- 发现异常的高纠缠对，追溯到训练数据或模型结构的问题。
- 为数据增强和模型改进提供方向。

![Interpretability Engine - Entanglement](slides_images/page_23.png)
*图：Leap Labs Engine 中的 Prototype Entanglement 可视化。图中展示了食物分类器中 "ice_cream" 与多个其他类别之间的纠缠关系。异常高的纠缠柱（如与 frozen_yogurt）是合理的，但如果出现与完全不相关类别的异常高纠缠，可能预示着数据中的伪相关或标注错误。*

### 解释个体预测——特征隔离（Feature Isolation）

**特征隔离（Feature Isolation）**回答的问题是：**对于这张具体的图片，模型是依据哪些特征做出这个预测的？**

它有两种应用方式：

1. **在原型上应用**：隔离出导致两个类别发生纠缠的共享特征。例如，对比 cheesecake 和 apple pie 的原型，高亮两者共有的“圆形饼状”区域，解释为什么模型会混淆。
2. **在真实输入上应用**：类似 **Saliency Map**。高亮输入图像中对当前预测贡献最大的像素区域。如果模型把草地上的狗预测为“草地”，而高亮区域在背景，就说明模型过度依赖背景特征。

![Interpretability Engine - Feature Isolation](slides_images/page_24.png)
*图：Feature Isolation 的实际效果。左侧展示了在原型上隔离出的共享特征（高亮部分）；右侧展示了在真实输入上应用时的显著性区域（类似于 Saliency Mapping）。*

#### 5.3.1 特征隔离与其他可解释性方法的对比

| 方法 | 类型 | 核心思想 | 优点 | 缺点 |
|------|------|----------|------|------|
| **Feature Isolation** | Global / Local | 隔离出模型用于区分类别的核心特征 | 既能解释类别关系，也能解释单个样本 | 需要生成原型，计算成本较高 |
| **Grad-CAM** | Local | 用最后一层卷积层的梯度加权得到热力图 | 空间分辨率高，解释力强，计算快 | 只能解释卷积网络，对全连接层效果差 |
| **LIME** | Local | 局部扰动采样 + 线性近似 | 模型无关，适用范围广 | 解释质量和采样策略强相关，可能不稳定 |
| **SHAP** | Local | 基于 Shapley 值计算特征边际贡献 | 理论基础扎实，具有良好的一致性 | 计算成本高，对高维图像数据需要近似 |
| **Saliency Map** | Local | 直接可视化损失对输入像素的梯度 | 实现最简单 | 梯度噪声大，可视化结果往往很粗糙 |

> **补充**：在实际工作中，通常会**组合使用**多种方法。例如，先用 Grad-CAM 快速定位关键区域，再用 Feature Isolation 深入分析该区域在模型内部对应的具体特征。

---

## 6. 特征可视化：探索神经元的“视觉偏好”

如果说 Interpretability Engine 更侧重于“类别级别”的解释，那么**特征可视化（Feature Visualization）**则深入到**神经网络内部**，回答一个更微观的问题：

> **这个神经元、这个卷积核、这一层通道，到底喜欢看什么样的图案？**

### 核心思想

神经网络对其输入是可微的。因此，我们可以选择一个感兴趣的单元（unit）——一个卷积核、一个通道、一个神经元，甚至整个层——将其激活值作为**优化目标**，通过梯度上升来调整输入图像，直到激活值最大。

![Feature Visualization Concept](slides_images/page_25.png)
*图：特征可视化的概念。A/B/C 展示了卷积层中不同特征图的可视化；D/E/F 展示了全连接层中不同神经元的激活偏好。*

#### 6.1.1 不同层的可视化差异

在深度神经网络中，不同层级学到的特征具有显著的层级结构：

- **浅层（Low-level layers）**：通常对低级视觉特征敏感，如**边缘、颜色、纹理、简单的几何形状**。这些特征与人类的初级视觉皮层（V1）处理的信息类似。
- **中层（Mid-level layers）**：开始组合低级特征，对更复杂的模式敏感，如**纹理图案、局部物体部件**（如车轮、眼睛、羽毛）。
- **深层（High-level layers）**：提取高度抽象的语义特征，对整个物体或场景概念有反应，如**人脸、建筑物、动物**的整体形态。

因此，当我们对网络的不同层进行特征可视化时，会观察到明显的“从简单到复杂”的过渡。这也是深度学习中**层次化特征提取（Hierarchical Feature Learning）**的直观证据。

### 一个优化问题

标准流程：
1. 初始化一张**随机噪声图像**。
2. 输入网络，计算目标单元的激活值（Objective）。
3. 反向传播计算 Objective 对输入图像的梯度。
4. 沿着梯度方向更新输入图像（让激活值更大）。
5. 重复直到收敛。

![Feature Viz Optimization Problem](slides_images/page_26.png)
*图：特征可视化的流程。从 Input (Noise) 出发，最大化中间层的 activation（Objective），通过 Backpropagation 不断更新输入图像。*

### 优化 Logits

如果我们直接以某个输出类别的 logit 为目标进行优化，就像生成对抗样本一样，只是没有真实的输入图像作为起点：

![Optimizing for Logits](slides_images/page_27.png)
*图：优化某个输出 logit。一切都从随机噪声开始演化。*

### 问题——高频伪影

直接从噪声优化，结果往往充满不自然的高频条纹、彩色斑点和噪点。

![Not So Easy](slides_images/page_28.png)
*图：直接从噪声优化的结果。右侧四个图像都包含大量高频伪影，更像抽象艺术而非自然照片。*

> **为什么会出现高频伪影？**  
> 神经网络对高频信号非常敏感。优化过程中，模型倾向于放大这些高频模式来最大化激活值。而自然图像的统计特性（局部平滑、颜色一致、几何结构）并没有被纳入优化目标。

### 引入正则化

为了让结果更像自然图像，我们需要给优化目标加上**正则化项**：

$$
\text{Objective} = \text{Activation}(x) + \lambda \cdot R(x)
$$

![Regularization](slides_images/page_29.png)
*图：引入正则化后，Objective 变为 logit + regularization，反向传播同时考虑激活最大化和图像自然度。*

#### 6.5.1 频率惩罚（Frequency Penalization / Total Variation）

惩罚相邻像素之间的高方差，使图像局部更平滑：

$$
R_{\text{TV}}(x) = -\sum_{i,j} \sqrt{(x_{i,j} - x_{i+1,j})^2 + (x_{i,j} - x_{i,j+1})^2}
$$

> **副作用**：可能过度平滑合法的边缘（如人物与背景的分界线）。

#### 6.5.2 变换鲁棒性（Transformation Robustness）

在每次迭代中，对输入图像施加轻微的随机变换（旋转、平移、缩放），分别计算激活值并取平均。这样优化过程会学习对几何变换不变的、更鲁棒的特征，而不是依赖特定位置的微小噪声。

![Transformation Robustness](slides_images/page_30.png)
*图：频率惩罚与变换鲁棒性的对比。左下角展示了不同 TV 强度下的结果；右下角展示了变换鲁棒性如何生成更结构化、更自然的图案。*

#### 6.5.3 TV 正则化的伪代码示例

```python
import torch
import torch.nn.functional as F

def total_variation_loss(img):
    """
    img: tensor of shape (1, 3, H, W)
    """
    # 计算水平和垂直方向上的像素差异
    diff_h = img[:, :, 1:, :] - img[:, :, :-1, :]
    diff_w = img[:, :, :, 1:] - img[:, :, :, :-1]
    
    # TV loss = sum of L2 norms of differences
    tv_loss = torch.sum(diff_h ** 2) + torch.sum(diff_w ** 2)
    return tv_loss

# 在优化循环中
objective = model.activation(layer_name, input_img)
loss = objective - lambda_tv * total_variation_loss(input_img)
loss.backward()
optimizer.step()
```

#### 6.5.4 变换鲁棒性的伪代码示例

```python
import torch
import torchvision.transforms as T

def transformation_robustness_objective(model, layer_name, input_img, target_unit, n_samples=4):
    total_activation = 0
    transforms = T.Compose([
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.RandomCrop(size=input_img.shape[-2:])
    ])
    
    for _ in range(n_samples):
        x_aug = transforms(input_img)
        activation = model.get_activation(layer_name, x_aug)[target_unit]
        total_activation += activation
    
    return total_activation / n_samples

# 在优化循环中
objective = transformation_robustness_objective(model, layer_name, input_img, target_unit)
loss = objective - lambda_tv * total_variation_loss(input_img)
loss.backward()
optimizer.step()
```

> **补充：更高级的正则化**  
> - **生成器约束**：在预训练 GAN/VAE 的潜空间中优化，结果天然具有自然图像统计特性。  
> - **多尺度优化**：从低分辨率开始，逐步上采样细化。  
> - **风格先验**：引入 Gram matrix 约束纹理和颜色分布。

### 为什么 Leap Labs 的原型如此自然？

Leap Labs 在论文 *Prototype Generation: Robust Feature Visualisation for Data Independent Interpretability* 中公布了他们的方法。他们不仅将原型 $P$ 定义为**最大化目标类别 logit 的输入**，更关键的是加入了一个**内部激活约束**：让原型 $P$ 在网络各层产生的激活模式 $A_P$ 尽可能接近真实自然图像 $I$ 的激活分布 $A_I$。

具体来说，论文使用 **Spearman correlation** 来衡量 $A_P$ 与 $A_I$ 的接近程度。这避免了优化结果只追求输出 logit 最大，而内部表征却极度异常（例如激活模式与自然图像完全不同）。从流程图上看，他们的方法大致如下：

1. **生成候选输入** $P$（从随机噪声开始）。
2. **计算内部激活** $A_P$ 并与自然图像的平均激活 $A_I$ 比较，施加距离惩罚。
3. **应用随机变换**（旋转、缩放、平移），对变换后的多个版本取平均激活，增强鲁棒性。
4. **迭代优化**，直到生成的原型既具有高分类置信度，又在内部表征上与自然图像统计一致。

这个“激活一致性”约束是 Leap Labs 原型看起来比传统 activation maximization 更加自然、更少高频伪影的关键秘诀之一。

![Why Leap Labs prototypes look so natural](slides_images/page_31.png)
*图：Leap Labs 论文中关于 Prototype Generation 的定义与流程图。原型不仅要最大化目标 logit，还要在网络各层保持与自然图像相近的激活模式（通过 Spearman correlation 度量）。*

---

## 7. 语言模型的可解释性

前面讨论的技术主要针对**视觉模型**。那么，对于现代 AI 的核心——**语言模型（Language Models, LMs）**，我们是否也能应用类似的方法？

答案是：**可以，但必须克服输入离散性的障碍。**

### 能否将技术迁移到语言模型？

> **给定一个期望的输出，我们能否找到什么样的 prompt 会生成它？**

这正是语言模型可解释性的根本问题。前面我们学过的所有基于梯度的可解释性技术（对抗样本、特征可视化），其核心都依赖于“反向传播到输入”。但在语言模型中，输入是**离散的 token**，这似乎阻断了一切。

然而，正如我们即将看到的，通过巧妙地绕过离散性，我们仍然可以为语言模型找到可解释的 prompt。


### 语言模型基础

语言模型为词序列（token 序列）分配概率。在实际应用中，给定一个提示（Prompt），语言模型会输出词汇表中每个 token 成为“下一个词”的概率：

$$
P(\text{next token} \mid \text{Prompt})
$$

例如，给定 "Shanghai is a city in"：
- China: 85%
- Beijing: 10%
- Cat: 2.5%
- Pizza: ...

![Short Intro to Language Models](slides_images/page_33.png)
*图：语言模型的核心任务。给定 prompt "Shanghai is a city in"，China 以 85% 概率成为最可能的下一个 token。*

#### 7.2.1 语言模型的可解释性挑战

与视觉模型相比，语言模型的可解释性面临三个独特的挑战：

1. **离散性**：输入是整数 token IDs，不可微。
2. **组合爆炸性**：对于长度为 $n$ 的 prompt，搜索空间是 $|V|^n$，其中 $|V|$ 通常是 5 万到 20 万。
3. **上下文依赖性**：同一个 token 在不同上下文中的含义可能截然不同（一词多义、语法角色等）。

### 离散输入带来的障碍

如果我们想问：**什么样的 3-token prompt 会让模型输出 "Girl"？**

直接梯度优化会立刻碰壁：语言模型的输入不是连续像素，而是**离散的整数 token IDs**。

![Discrete Token Problem](slides_images/page_34.png)
*图：由于输入是离散 token ids，我们无法直接计算梯度并更新 token。这是将视觉可解释性方法迁移到语言模型时的核心障碍。*

### GPT-2 上的实验结果

Jessica Rumbelow（Leap Labs 创始人）在 **GPT2-xl**（15 亿参数）上进行了实验，用后续的 embedding 优化算法寻找能引发特定输出的 prompt。结果以**聚合词频（Aggregated Token Frequencies）**的形式呈现，形成了引人注目的词云。

![GPT2-xl Examples (1)](slides_images/page_35.png)
*图：针对 GPT2-xl 的 feature visualization。左侧展示了优化流程（从随机输入 embedding 出发，最大化目标类别的 output logit）；右侧展示了针对 "girl"、"boy" 等目标优化出的 prompt 中的高频 token 列表。*

![GPT2-xl Examples (2)](slides_images/page_36.png)
*图：GPT2-xl 实验的聚合词云。针对目标输出 'girl'，优化出的高频输入词包括 girlfriend、sexy、witch、boy 等；针对 'boy' 则有 rebellious、Monkey、girl、kid 等。这些词云揭示了模型内部可能存在的性别刻板印象和意想不到的语义关联。*

#### 7.4.1 对 GPT-2 实验结果的深入分析

这些词云不仅展示了算法的有效性，更揭示了大型语言模型内部的一些深层模式：

- **性别刻板印象**：目标输出 "girl" 的优化 prompt 中高频出现 "sexy"、"girlfriend"，而 "boy" 的 prompt 中高频出现 "rebellious"。这强烈暗示训练数据（互联网的广泛文本）中存在着显著的性别刻板印象，并且模型已经内化了这些关联。

- **意想不到的语义桥梁**："girl" 和 "boy" 互相出现在对方的优化 prompt 中，说明模型在语义空间中把这两个概念放得很近，这符合直觉。但更有趣的是 "Monkey" 出现在 "boy" 的 prompt 中——这可能是某些训练文本中的俚语、昵称或特定亚文化用语的统计残留。

- **领域特异性**：目标输出 "science" 的 prompt 包含 "biology"、"physics"、"scientists"；"art" 的 prompt 包含 "artwork"、"artists"、"ceramics"。这说明模型已经很好地学习了不同领域之间的语义聚类。

这些发现的价值在于：它们为我们提供了一种**考古学式的方法**——通过逆向工程 prompt，我们可以挖掘出模型在训练过程中潜移默化学到的、甚至连设计者都未曾意识到的关联模式。

### 绕过障碍——优化 Embedding

解决方案：**不直接优化离散的 token，而是优化它们的连续 embedding 表示。**

语言模型内部有一个**嵌入矩阵（Embedding Matrix）** $E \in \mathbb{R}^{V \times d}$，将每个离散 token ID $t$ 映射为 $d$ 维连续向量 $e_t = E[t]$。这个向量是可微的！

流程：
1. 初始化一组**随机的连续向量**作为伪输入 embedding。
2. 将它们输入到语言模型中。
3. 定义目标为最大化某个特定输出 token 的 logit（例如 "Girl"）。
4. 通过反向传播不断更新这些输入向量。

![What About Embeddings](slides_images/page_37.png)
*图：通过优化输入 embeddings 而非 token ids，我们绕过了离散性障碍。从随机 embeddings 出发，反向传播最大化目标 token 的 logit。注意图中的 Transformer 架构：输入 embeddings 经过 Multi-Head Attention、Add & Norm、Feed Forward 等层后到达输出。*

#### 7.5.1 剩余的两个问题

然而，上述方法还留下了两个未解决的问题：

1. **字典外问题（Out-of-Vocabulary Problem）**：优化得到的向量可能不对应 vocabulary 中的任何真实 token。
2. **收敛问题**：我们需要一种机制，迫使优化过程向词表中真实存在的 embedding 收敛，而不是停留在 embedding 空间的任意位置上。

### Slide 38：解决方案——K-NN + 距离正则化

Leap Labs 提出的解决方案巧妙地结合了 **K-Nearest Neighbor (K-NN)** 和**距离正则化**：

#### 7.6.1 K-NN 映射

为每个优化后的向量 $e_{\text{opt}}$ 找到词表嵌入矩阵 $E$ 中最接近的邻居：

$$
t^* = \arg\min_{t \in \text{Vocab}} \| e_{\text{opt}} - E[t] \|_2
$$

这样我们就能将"伪输入"翻译回人类可读的 token。

#### 7.6.2 距离正则化

在损失函数中加入距离惩罚项：

$$
L_{\text{total}} = -\text{logit}_{\text{target}} + \lambda \sum_{i} \min_{t \in \text{Vocab}} \| e_i - E[t] \|_2^2
$$

- 第一项：最大化目标 token 的 logit（取负后最小化）。
- 第二项：惩罚每个优化向量与其最近邻真实 embedding 的距离，驱使它向 vocabulary 收敛。

![K-NN and Regularization](slides_images/page_38.png)
*图：K-NN 与距离正则化结合。Loss = 目标 logit + 正则化项。反向传播会同时考虑：1) 让模型输出目标 token；2) 让输入 embedding 尽量贴近词表中真实存在的 token embedding。*

#### 7.6.3 算法伪代码

```python
import torch
import torch.nn as nn

# 假设 model 是 GPT-2，embeddings 是模型的 token embedding 矩阵
vocab_embeddings = model.transformer.wte.weight  # shape: (V, d)

# 初始化随机输入 embedding (例如 3 个 token)
opt_embeddings = torch.randn(1, 3, d, requires_grad=True)
optimizer = torch.optim.Adam([opt_embeddings], lr=0.1)

target_token_id = tokenizer.encode("Girl")[0]

for step in range(1000):
    optimizer.zero_grad()
    
    # 前向传播
    logits = model(inputs_embeds=opt_embeddings).logits
    target_logit = logits[0, -1, target_token_id]  # 最后一个位置的目标 logit
    
    # K-NN 距离正则化
    distances = torch.cdist(opt_embeddings.squeeze(0), vocab_embeddings)  # (3, V)
    min_distances = torch.min(distances, dim=1)[0]  # (3,)
    knn_reg = torch.sum(min_distances ** 2)
    
    loss = -target_logit + lambda_reg * knn_reg
    loss.backward()
    optimizer.step()
    
    # 每隔一段时间，用 K-NN 查看当前最接近的 tokens
    if step % 100 == 0:
        nearest_ids = torch.argmin(distances, dim=1)
        tokens = tokenizer.decode(nearest_ids)
        print(f"Step {step}: closest tokens = {tokens}")
```

#### 7.6.4 与 Prompt Tuning 的联系

优化输入 embedding 的方法与现代 NLP 中的 **Prompt Tuning**（Lester et al., 2021）深刻相关。在 Prompt Tuning 中，我们在输入前添加可学习的连续向量（soft prompts），冻结预训练模型参数进行端到端微调。这些 soft prompts 在 embedding 空间中优化，但在推理时并不对应任何离散的 human-readable token。

而 Leap Labs 的方法额外加入了 K-NN 映射和距离正则化，目的是**将优化结果硬约束（或鼓励收敛）到可解释的离散 token 序列上**。这实际上是连接 soft prompting 和 hard prompt engineering 的一座桥梁。

> **可解释性让我们从“模型预测对了”走向“模型为什么这样预测”——而后者，才是我们真正能够信任和改进 AI 的起点。**

