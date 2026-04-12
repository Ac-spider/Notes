# 6.S184 Flow Matching and Diffusion Models — 公式、算法与推导笔记

## 目录

1. [引言：生成建模即采样](#1-引言生成建模即采样)
2. [Flow 与 Diffusion 模型](#2-flow-与-diffusion-模型)
3. [Flow Matching](#3-flow-matching)
4. [Score Functions 与 Score Matching](#4-score-functions-与-score-matching)
5. [Guidance：如何基于 Prompt 条件生成](#5-guidance如何基于-prompt-条件生成)
6. [大规模图像/视频生成器架构](#6-大规模图像视频生成器架构)
7. [离散扩散模型（CTMC / 语言模型）](#7-离散扩散模型ctmc--语言模型)
8. [算法汇总](#8-算法汇总)

---

## 1. 引言：生成建模即采样

### 核心设定

- 数据表示为向量：$z \in \mathbb{R}^d$（图像、视频、分子结构等）
- **Dataset（数据集）**：$z_1, \dots, z_N \sim p_{\text{data}}$
- **Generation（生成）**：采样 $z \sim p_{\text{data}}$
- **Guided Generation（条件生成）**：采样 $z \sim p_{\text{data}}(\cdot \mid y)$，其中 $y$ 为条件（如文本 prompt）

> **使用场景**：将“生成一张狗的图片”这一模糊任务，转化为从 data distribution 中采样的数学问题。

---

## 2. Flow 与 Diffusion 模型

### 2.1 Flow Models（基于 ODE）

#### ODE 定义
$$\frac{\mathrm{d}}{\mathrm{d}t} X_t = u_t(X_t) \quad \text{(ODE)} \tag{1a} $$ ^eq-1a
$$X_0 = x_0 \quad \text{(initial condition)} \tag{1b} $$ ^eq-1b

- $X: [0,1] \to \mathbb{R}^d$：轨迹（trajectory）
- $u: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$：向量场（vector field），$u_t(x)$ 表示在时刻 $t$、位置 $x$ 处的速度方向

#### Flow（流）的定义

流 $\psi_t(x_0)$ 回答：从 $x_0$ 出发，按向量场 $u_t$ 演化，时刻 $t$ 会到达哪里？
$$\psi: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d, \quad (x_0, t) \mapsto \psi_t(x_0) \tag{2a} $$ ^eq-2a
$$\frac{\mathrm{d}}{\mathrm{d}t} \psi_t(x_0) = u_t(\psi_t(x_0)) \tag{2b} $$ ^eq-2b
$$\psi_0(x_0) = x_0 \tag{2c} $$ ^eq-2c

对给定初值 $X_0 = x_0$，有 $X_t = \psi_t(X_0)$。

#### 定理 3：Flow 存在唯一性

若 $u: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$ **连续可微且导数有界**，则 ODE [(2a)](#^eq-2a) 存在唯一解流 $\psi_t$，且对每个 $t$，$\psi_t$ 是**微分同胚**（diffeomorphism，即连续可微且逆也连续可微）。

> **直观**：在深度学习里，神经网络参数化 $u_t^\theta(x)$ 天然满足这些条件，因此 Flow 总是存在且唯一的。

#### 例子 4：线性向量场

设 $u_t(x) = -\theta x$（$\theta > 0$），则
$$\psi_t(x_0) = \exp(-\theta t) x_0 \tag{3} $$ ^eq-3

**推导验证**：
- 初始条件：$\psi_0(x_0) = e^0 x_0 = x_0$ ✓
- 对时间求导：$\frac{\mathrm{d}}{\mathrm{d}t}\psi_t(x_0) = -\theta e^{-\theta t} x_0 = -\theta \psi_t(x_0) = u_t(\psi_t(x_0))$ ✓

> **直观**：所有粒子向原点指数衰减，$\sigma=0$ 时是确定性 Flow。

#### Euler 数值模拟方法
$$X_{t+h} = X_t + h u_t(X_t), \quad h = \frac{1}{n} \tag{4} $$ ^eq-4

#### Heun 方法（二阶）

$$X'_{t+h} = X_t + h u_t(X_t) \quad \text{(Euler 预测步)}$$

$$X_{t+h} = X_t + \frac{h}{2}\left(u_t(X_t) + u_{t+h}(X'_{t+h})\right) \quad \text{(校正步，取平均方向)}$$

#### Flow Model（生成模型）

$$X_0 \sim p_{\text{init}} = \mathcal{N}(0, I_d)$$

$$\frac{\mathrm{d}}{\mathrm{d}t} X_t = u_t^\theta(X_t)$$

目标是让终点满足：

$$X_1 \sim p_{\text{data}} \quad \Leftrightarrow \quad \psi_1^\theta(X_0) \sim p_{\text{data}}$$

> **关键理解**：神经网络参数化的是**向量场** $u_t^\theta$，而不是流 $\psi_t$ 本身。要得到样本，必须数值积分（模拟 ODE）。

---

### 2.2 Diffusion Models（基于 SDE）

#### Brownian Motion（布朗运动 / Wiener 过程）

$W = (W_t)_{0 \le t \le 1}$ 满足：
- $W_0 = 0$
- 轨迹 $t \mapsto W_t$ 连续
- **Normal increments**：$W_t - W_s \sim \mathcal{N}(0, (t-s)I_d)$
- **Independent increments**：不重叠区间上的增量相互独立

数值模拟（步长 $h$）：
$$W_{t+h} = W_t + \sqrt{h} \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I_d) \tag{5} $$ ^eq-5

> **直观**：把布朗运动想象成“连续时间的随机游走”，每一步增加一个方差为 $h$ 的高斯噪声。

#### 从 ODE 到 SDE

ODE 的导数形式：

$$\frac{1}{h}(X_{t+h} - X_t) = u_t(X_t) + R_t(h) \quad \Rightarrow \quad X_{t+h} = X_t + h u_t(X_t) + h R_t(h)$$

加入随机项后得到 SDE：
$$X_{t+h} = X_t + \underbrace{h u_t(X_t)}_{\text{确定性}} + \underbrace{\sigma_t (W_{t+h} - W_t)}_{\text{随机性}} + \underbrace{h R_t(h)}_{\text{误差项}} \tag{6} $$ ^eq-6

符号记法（**非严格意义**，仅为方便）：
$$\mathrm{d}X_t = u_t(X_t)\,\mathrm{d}t + \sigma_t\,\mathrm{d}W_t \quad \text{(SDE)} \tag{7a} $$ ^eq-7a
$$X_0 = x_0 \tag{7b} $$ ^eq-7b

#### 例子 6：Ornstein-Uhlenbeck (OU) 过程
$$\mathrm{d}X_t = -\theta X_t \,\mathrm{d}t + \sigma\,\mathrm{d}W_t \tag{8} $$ ^eq-8

- 漂移项 $-\theta X_t$：始终指向原点，把粒子拉回中心
- 扩散项 $\sigma\,\mathrm{d}W_t$：不断注入高斯噪声
- 当 $t \to \infty$ 时，$X_t$ 收敛到稳态分布 $\mathcal{N}\left(0, \frac{\sigma^2}{2\theta}\right)$
- 当 $\sigma = 0$ 时，退化为例子 4 的线性 Flow

> **使用场景**：OU 过程是扩散模型早期的基础（如 DDPM 的前向加噪过程可视为离散化的 OU 过程）。

#### Euler-Maruyama 模拟方法
$$X_{t+h} = X_t + h u_t(X_t) + \sqrt{h}\,\sigma_t \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I_d) \tag{9} $$ ^eq-9

#### 定理 5：SDE 解的存在唯一性

若 $u$ 连续可微且导数有界，$\sigma_t$ 连续，则 SDE [(7a)](#^eq-7a) 存在唯一随机过程解 $(X_t)_{0 \le t \le 1}$。

---

## 3. Flow Matching

### 3.1 条件与边缘概率路径

#### 条件概率路径

对每个数据点 $z$，定义条件概率路径 $p_t(x \mid z)$，满足：
$$p_0(\cdot \mid z) = p_{\text{init}}, \quad p_1(\cdot \mid z) = \delta_z \tag{11} $$ ^eq-11

#### 边缘概率路径
$$p_t(x) = \int p_t(x \mid z) p_{\text{data}}(z)\,\mathrm{d}z \tag{12} $$ ^eq-12

满足：
$$p_0 = p_{\text{init}}, \quad p_1 = p_{\text{data}} \tag{14} $$ ^eq-14

#### 例子 8：Gaussian 条件概率路径（最重要！）

设噪声调度器 $\alpha_t, \beta_t$ 满足：
- 连续可微、单调
- $\alpha_0 = \beta_1 = 0$，$\alpha_1 = \beta_0 = 1$

则定义：
$$p_t(\cdot \mid z) = \mathcal{N}\left(\alpha_t z, \; \beta_t^2 I_d\right) \tag{15} $$ ^eq-15

验证边界：
- $t=0$：$p_0(\cdot \mid z) = \mathcal{N}(0, I_d) = p_{\text{init}}$
- $t=1$：$p_1(\cdot \mid z) = \mathcal{N}(z, 0) = \delta_z$

采样方式：
$$x = \alpha_t z + \beta_t \epsilon, \quad z \sim p_{\text{data}},\; \epsilon \sim \mathcal{N}(0, I_d) \quad \Rightarrow \quad x \sim p_t \tag{16} $$ ^eq-16

> **直观**：$\alpha_t$ 控制数据成分的权重，$\beta_t$ 控制噪声成分的权重。$t$ 越小，噪声越多；$t=1$ 时只有纯净数据。

---

### 3.2 条件与边缘向量场

#### 条件向量场

对每个 $z$，寻找条件向量场 $u_t^{\text{target}}(x \mid z)$，使其 ODE 轨迹满足：
$$X_0 \sim p_{\text{init}}, \quad \frac{\mathrm{d}}{\mathrm{d}t}X_t = u_t^{\text{target}}(X_t \mid z) \quad \Rightarrow \quad X_t \sim p_t(\cdot \mid z) \tag{17} $$ ^eq-17

#### 定理 9：边缘化技巧（Marginalization Trick）

若 $u_t^{\text{target}}(x \mid z)$ 是条件向量场，则下式定义的边缘向量场能跟随边缘概率路径：
$$u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x \mid z) \frac{p_t(x \mid z) p_{\text{data}}(z)}{p_t(x)} \,\mathrm{d}z \tag{18} $$ ^eq-18

从而：
$$X_0 \sim p_{\text{init}}, \quad \frac{\mathrm{d}}{\mathrm{d}t}X_t = u_t^{\text{target}}(X_t) \quad \Rightarrow \quad X_t \sim p_t \quad (0 \le t \le 1) \tag{19} $$ ^eq-19

特别地，$X_1 \sim p_{\text{data}}$，即边缘向量场“将噪声转化为数据”。

> **直观解释**：$\frac{p_t(x \mid z) p_{\text{data}}(z)}{p_t(x)}$ 是给定含噪样本 $x$ 时，原始数据为 $z$ 的后验概率。边缘向量场就是“按后验概率加权平均”所有可能数据点 $z$ 对应的方向。

#### 例子 10：Gaussian 概率路径的条件向量场

对 $p_t(\cdot \mid z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$，条件向量场为：
$$u_t^{\text{target}}(x \mid z) = \left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right) z + \frac{\dot{\beta}_t}{\beta_t} x \tag{20} $$ ^eq-20

**推导**：

先构造条件流：$\psi_t^{\text{target}}(x \mid z) = \alpha_t z + \beta_t x$。若 $X_0 \sim \mathcal{N}(0, I_d)$，则

$$X_t = \psi_t^{\text{target}}(X_0 \mid z) = \alpha_t z + \beta_t X_0 \sim \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$$

由流的定义 $\frac{\mathrm{d}}{\mathrm{d}t}\psi_t^{\text{target}}(x \mid z) = u_t^{\text{target}}(\psi_t^{\text{target}}(x \mid z) \mid z)$，左边求导得：

$$\dot{\alpha}_t z + \dot{\beta}_t x = u_t^{\text{target}}(\alpha_t z + \beta_t x \mid z)$$

令 $x' = \alpha_t z + \beta_t x$，即 $x = \frac{x' - \alpha_t z}{\beta_t}$，回代得：

$$u_t^{\text{target}}(x' \mid z) = \dot{\alpha}_t z + \dot{\beta}_t \frac{x' - \alpha_t z}{\beta_t} = \left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right) z + \frac{\dot{\beta}_t}{\beta_t} x'$$

#### 定理 11：连续性方程（Continuity Equation）

对向量场 $u_t^{\text{target}}$ 与概率路径 $p_t$，有 $X_t \sim p_t$ 当且仅当：
$$\partial_t p_t(x) = -\operatorname{div}\left(p_t u_t^{\text{target}}\right)(x) \tag{23} $$ ^eq-23

其中散度定义为：
$$\operatorname{div}(v_t)(x) = \sum_{i=1}^d \frac{\partial}{\partial x_i} v_t^i(x) \tag{22} $$ ^eq-22

> **直观**：$\partial_t p_t$ 是概率密度的变化率，它等于概率质量净流入率的负数。散度 $\operatorname{div}(p_t u_t)$ 衡量从 $x$ 处向外流出的概率质量，故前面加负号表示净流入。

---

### 3.3 学习边缘向量场

#### Flow Matching Loss（理想但不可行）
$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \text{Unif},\, x \sim p_t}\left[ \left\| u_t^\theta(x) - u_t^{\text{target}}(x) \right\|^2 \right] \tag{24} $$ ^eq-24

问题：$u_t^{\text{target}}(x)$ 需要通过积分 [(18)](#^eq-18) 计算，**不可行**。

#### Conditional Flow Matching Loss（可训练的目标）
$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \text{Unif},\, z \sim p_{\text{data}},\, x \sim p_t(\cdot \mid z)}\left[ \left\| u_t^\theta(x) - u_t^{\text{target}}(x \mid z) \right\|^2 \right] \tag{26} $$ ^eq-26

#### 定理 12：CFM 与 FM 等价（仅差与 $\theta$ 无关的常数）

$$\mathcal{L}_{\text{FM}}(\theta) = \mathcal{L}_{\text{CFM}}(\theta) + C, \quad \nabla_\theta \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{CFM}}(\theta)$$

**核心推导**：展开 FM loss：

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t,x}\left[\|u_t^\theta\|^2\right] - 2\mathbb{E}_{t,x}\left[u_t^\theta(x)^T u_t^{\text{target}}(x)\right] + C_1$$

关键是将第二项中的边缘向量场替换为条件向量场：

$$\mathbb{E}_{t,x}\left[u_t^\theta(x)^T u_t^{\text{target}}(x)\right]$$

$$= \int_0^1 \int p_t(x) u_t^\theta(x)^T \left[\int u_t^{\text{target}}(x \mid z) \frac{p_t(x \mid z) p_{\text{data}}(z)}{p_t(x)} \,\mathrm{d}z \right] \,\mathrm{d}x \,\mathrm{d}t$$

$$= \int_0^1 \int\int u_t^\theta(x)^T u_t^{\text{target}}(x \mid z) p_t(x \mid z) p_{\text{data}}(z) \,\mathrm{d}z \,\mathrm{d}x \,\mathrm{d}t$$

$$= \mathbb{E}_{t, z, x \sim p_t(\cdot \mid z)}\left[u_t^\theta(x)^T u_t^{\text{target}}(x \mid z)\right]$$

回代后，通过配平方 $\|a - b\|^2 = \|a\|^2 - 2a^T b + \|b\|^2$，并加/减 $\|u_t^{\text{target}}(x \mid z)\|^2$，最终得到：

$$\mathcal{L}_{\text{FM}} = \mathcal{L}_{\text{CFM}} + C$$

> **重大意义**：我们不需要模拟 ODE 或计算边缘向量场的积分，只需用**可解析求得的条件向量场**做简单的均方误差回归即可！这正是 Flow Matching 的核心优势——**simulation-free training**。

#### 例子 13：Gaussian 路径的 CFM Loss

> 想象从噪声 $\epsilon$ 到真实数据 $z$ 画一条 **直线**：
>
> $$x_t = t \cdot z + (1 - t) \cdot \epsilon$$
>
> - $t = 0$ 时：$x_0 = \epsilon$（纯噪声）
> - $t = 1$ 时：$x_1 = z$（真实数据）
>
> 这条直线上每一点的**切线方向**（速度）都是：
>
> $$\frac{dx_t}{dt} = z - \epsilon$$
>
> 这是一个**常数方向**，即目标速度场。

对 $p_t(\cdot \mid z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$，将 [(20)](#^eq-20) 代入 [(26)](#^eq-26)：
$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, z, \epsilon \sim \mathcal{N}(0,I_d)}\left[ \left\| u_t^\theta(\alpha_t z + \beta_t \epsilon) - (\dot{\alpha}_t z + \dot{\beta}_t \epsilon) \right\|^2 \right] \tag{31} $$ ^eq-31

**推导**：把 $x = \alpha_t z + \beta_t \epsilon$ 代入条件向量场 [(20)](#^eq-20)：

$$u_t^{\text{target}}(x \mid z) = \left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right) z + \frac{\dot{\beta}_t}{\beta_t}(\alpha_t z + \beta_t \epsilon) = \dot{\alpha}_t z + \dot{\beta}_t \epsilon$$

CondOT 特例（$\alpha_t = t, \beta_t = 1-t$）：

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, z, \epsilon}\left[ \left\| u_t^\theta(tz + (1-t)\epsilon) - (z - \epsilon) \right\|^2 \right]$$

> **使用场景**：Stable Diffusion 3、Meta Movie Gen Video 等众多 SOTA 模型均使用该训练目标。

---

## 4. Score Functions 与 Score Matching

### 4.1 条件与边缘 Score Function

对任意分布 $q(x)$，定义其 **score function** 为：

$$\nabla \log q(x)$$

它指向对数似然增长最快的方向。

对条件/边缘概率路径，定义：
- 条件 score：$\nabla \log p_t(x \mid z)$
- 边缘 score：$\nabla \log p_t(x)$

类似 [(18)](#^eq-18)，边缘 score 也可写成条件 score 的后验加权平均：
$$\nabla \log p_t(x) = \int \nabla \log p_t(x \mid z) \frac{p_t(x \mid z) p_{\text{data}}(z)}{p_t(x)} \,\mathrm{d}z \tag{38} $$ ^eq-38

**推导**：

$$\nabla \log p_t(x) = \frac{\nabla p_t(x)}{p_t(x)} = \frac{\nabla \int p_t(x \mid z) p_{\text{data}}(z) \,\mathrm{d}z}{p_t(x)}$$

$$= \frac{\int \nabla p_t(x \mid z) p_{\text{data}}(z) \,\mathrm{d}z}{p_t(x)} = \int \frac{\nabla p_t(x \mid z)}{p_t(x \mid z)} \cdot \frac{p_t(x \mid z) p_{\text{data}}(z)}{p_t(x)} \,\mathrm{d}z$$

$$= \int \nabla \log p_t(x \mid z) \frac{p_t(x \mid z) p_{\text{data}}(z)}{p_t(x)} \,\mathrm{d}z$$

#### 例子 15：Gaussian 路径的 Score Function

对 $p_t(x \mid z) = \mathcal{N}(x; \alpha_t z, \beta_t^2 I_d)$：
$$\nabla \log p_t(x \mid z) = -\frac{x - \alpha_t z}{\beta_t^2} \tag{40} $$ ^eq-40

#### 命题 1：Gaussian 路径下向量场与 score 的转换
$$u_t^{\text{target}}(x \mid z) = a_t \nabla \log p_t(x \mid z) + b_t x \tag{41} $$ ^eq-41


$$u_t^{\text{target}}(x) = a_t \nabla \log p_t(x) + b_t x \tag{42} $$ ^eq-42

其中：

$$a_t = \beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t, \quad b_t = \frac{\dot{\alpha}_t}{\alpha_t}$$

**推导验证**：

将 [(40)](#^eq-40) 代入 [(41)](#^eq-41) 右端：

$$a_t \left(-\frac{x - \alpha_t z}{\beta_t^2}\right) + b_t x = \left(\beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t\right) \frac{\alpha_t z - x}{\beta_t^2} + \frac{\dot{\alpha}_t}{\alpha_t} x$$

$$= \left(\dot{\alpha}_t - \frac{\dot{\beta}_t \alpha_t}{\beta_t}\right) z + \left(-\frac{\dot{\alpha}_t}{\alpha_t} + \frac{\dot{\beta}_t}{\beta_t} + \frac{\dot{\alpha}_t}{\alpha_t}\right) x = \left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right) z + \frac{\dot{\beta}_t}{\beta_t} x$$

这正是 [(20)](#^eq-20)。对边缘版本，两边对后验积分即可。

> **重大意义**：学得了向量场就等于学得了 score，反之亦然。这使得扩散模型既可以用 Flow Matching 训练，也可以用 Score Matching 训练。

#### Denoiser（去噪器）
$$D_t(x) = \int z \frac{p_t(x \mid z) p_{\text{data}}(z)}{p_t(x)} \,\mathrm{d}z = \frac{1}{\dot{\alpha}_t \beta_t - \alpha_t \dot{\beta}_t} \left(\beta_t u_t^{\text{target}}(x) - \dot{\beta}_t x\right) \tag{43} $$ ^eq-43

> **直观**：$D_t(x)$ 是给定含噪数据 $x$ 后，对原始干净数据 $z$ 的后验期望估计。因此模型也被称为“denoising diffusion model”。$$D_t(x) = \mathbb{E}[z \mid x_t = x]$$

---

### 4.2 用 SDE 采样

#### 定理 17：SDE Extension Trick

对任意扩散系数 $\sigma_t \ge 0$，构造如下 SDE：
$$X_0 \sim p_{\text{init}}, \quad \mathrm{d}X_t = \left[ u_t^{\text{target}}(X_t) + \frac{\sigma_t^2}{2} \nabla \log p_t(X_t) \right] \mathrm{d}t + \sigma_t \,\mathrm{d}W_t \tag{44} $$ ^eq-44

则仍有 $X_t \sim p_t$（$0 \le t \le 1$）。特别地，$X_1 \sim p_{\text{data}}$。

> **直观**：在保持边缘概率路径不变的前提下，向 ODE 中“注入”了与 score 相关的漂移项和布朗运动噪声。随机性使轨迹呈锯齿状，但各时刻的分布不变。

#### 例子 18：Gaussian 路径的 SDE 形式

利用命题 1 将 [(44)](#^eq-44) 完全用 score 表示：
$$\mathrm{d}X_t = \left[ \left(a_t + \frac{\sigma_t^2}{2}\right) \nabla \log p_t(X_t) + b_t X_t \right] \mathrm{d}t + \sigma_t \,\mathrm{d}W_t \tag{46} $$ ^eq-46

#### 拉普拉斯算子
$$\Delta w_t(x) = \sum_{i=1}^d \frac{\partial^2}{\partial x_i^2} w_t(x) = \operatorname{div}(\nabla w_t)(x) \tag{48} $$ ^eq-48

#### 定理 19：Fokker-Planck 方程

对 SDE $X_0 \sim p_{\text{init}}, \; \mathrm{d}X_t = u_t(X_t)\mathrm{d}t + \sigma_t\,\mathrm{d}W_t$，有 $X_t \sim p_t$ 当且仅当：
$$\partial_t p_t(x) = -\operatorname{div}(p_t u_t)(x) + \frac{\sigma_t^2}{2} \Delta p_t(x) \tag{49} $$ ^eq-49

> **直观**：相比连续性方程 [(23)](#^eq-23)，Fokker-Planck 方程多了一个 Laplacian 项。它类似热传导方程中的扩散项，刻画了随机噪声导致的概率密度“摊开”效应。当 $\sigma_t = 0$ 时，[(49)](#^eq-49) 退化为 [(23)](#^eq-23)。

**定理 17 的证明核心**：将 [(44)](#^eq-44) 的漂移项代入 [(49)](#^eq-49)，通过加减 $\frac{\sigma_t^2}{2}\Delta p_t$、利用 $\nabla \log p_t = \frac{\nabla p_t}{p_t}$ 和散度的线性性，可直接验证满足 Fokker-Planck 方程。

#### Langevin Dynamics（朗之万动力学）

当概率路径为常数 $p_t = p$ 时，令 $u_t^{\text{target}} = 0$，得到：
$$\mathrm{d}X_t = \frac{\sigma_t^2}{2} \nabla \log p(X_t) \,\mathrm{d}t + \sigma_t \,\mathrm{d}W_t \tag{50} $$ ^eq-50

这是著名的 Langevin dynamics，$p$ 是其平稳分布（stationary distribution）。

> **使用场景**：Langevin dynamics 是分子动力学模拟、MCMC 采样等众多科学计算的基石。OU 过程就是 Langevin dynamics 在 $p = \mathcal{N}(0, \sigma^2/(2\theta))$ 时的特例。

---

### 4.3 Score Matching

用神经网络 $s_t^\theta(x)$ 近似边缘 score $\nabla \log p_t(x)$。

#### Score Matching Loss

$$\mathcal{L}_{\text{SM}}(\theta) = \mathbb{E}_{t, z, x \sim p_t(\cdot \mid z)}\left[ \left\| s_t^\theta(x) - \nabla \log p_t(x) \right\|^2 \right]$$

#### Conditional Score Matching Loss（可训练）

$$\mathcal{L}_{\text{CSM}}(\theta) = \mathbb{E}_{t, z, x \sim p_t(\cdot \mid z)}\left[ \left\| s_t^\theta(x) - \nabla \log p_t(x \mid z) \right\|^2 \right]$$

#### 定理 22：CSM 与 SM 等价

$$\mathcal{L}_{\text{SM}}(\theta) = \mathcal{L}_{\text{CSM}}(\theta) + C \quad \Rightarrow \quad \nabla_\theta \mathcal{L}_{\text{SM}} = \nabla_\theta \mathcal{L}_{\text{CSM}}$$

证明与定理 12 完全相同，只须将 $u_t^{\text{target}}$ 替换为 $\nabla \log p_t$。

#### 例子 23：Gaussian 路径的 Denoising Score Matching

对 $p_t(x \mid z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$，有条件 score [(40)](#^eq-40)，代入 CSM：

$$\mathcal{L}_{\text{CSM}} = \mathbb{E}_{t,z,\epsilon}\left[ \left\| s_t^\theta(\alpha_t z + \beta_t \epsilon) + \frac{\epsilon}{\beta_t} \right\|^2 \right]$$

$$= \mathbb{E}_{t,z,\epsilon}\left[ \frac{1}{\beta_t^2} \left\| \beta_t s_t^\theta(\alpha_t z + \beta_t \epsilon) + \epsilon \right\|^2 \right]$$

为避免 $\beta_t \approx 0$ 时的数值不稳定，DDPM [17] 通过重参数化定义 noise predictor：

$$\epsilon_t^\theta(x) = -\beta_t s_t^\theta(x)$$

得到 DDPM 训练目标：

$$\mathcal{L}_{\text{DDPM}}(\theta) = \mathbb{E}_{t,z,\epsilon}\left[ \left\| \epsilon_t^\theta(\alpha_t z + \beta_t \epsilon) - \epsilon \right\|^2 \right]$$

> **直观**：网络 $\epsilon_t^\theta$ 学习预测“加到干净数据上的噪声”。训练过程与 Flow Matching 类似——都是简单的 MSE 回归，只是回归目标不同。

---

## 5. Guidance：如何基于 Prompt 条件生成

### 5.1 Vanilla Guidance

将条件 $y$ 直接输入网络，训练 guided 向量场 $u_t^\theta(x \mid y)$。Guided CFM 目标：
$$\mathcal{L}_{\text{CFM}}^{\text{guided}}(\theta) = \mathbb{E}_{(z,y) \sim p_{\text{data}},\, t,\, x \sim p_t(\cdot \mid z)} \left[ \left\| u_t^\theta(x \mid y) - u_t^{\text{target}}(x \mid z) \right\|^2 \right] \tag{58} $$ ^eq-58

> **问题**：仅凭 vanilla guidance，生成样本往往与 prompt 对齐度不够。

### 5.2 Classifier-Free Guidance (CFG)

#### Classifier Guidance 启发式

对 Gaussian 路径，利用 Bayes 规则：

$$p_t(x \mid y) = \frac{p_t(x) p_t(y \mid x)}{p_t(y)}$$
$$\nabla \log p_t(x \mid y) = \nabla \log p_t(x) + \nabla \log p_t(y \mid x) \tag{61} $$ ^eq-61

将 [(61)](#^eq-61) 代入条件向量场（类比 [(41)](#^eq-41),[(42)](#^eq-42)），可得 guided 向量场：

$$u_t^{\text{target}}(x \mid y) = u_t^{\text{target}}(x) + a_t \nabla \log p_t(y \mid x) \tag{59} $$ ^eq-59

为加强对 prompt 的依赖，在 [(59)](#^eq-59) 的基础上引入 guidance scale $w > 1$：
$$\tilde{u}_t(x \mid y) = u_t^{\text{target}}(x) + w a_t \nabla \log p_t(y \mid x) \tag{62} $$ ^eq-62

#### Classifier-Free Guidance 推导

将 $\nabla \log p_t(y \mid x) = \nabla \log p_t(x \mid y) - \nabla \log p_t(x)$ 代入 [(62)](#^eq-62)：

$$\tilde{u}_t(x \mid y) = u_t^{\text{target}}(x) + w a_t \left(\nabla \log p_t(x \mid y) - \nabla \log p_t(x)\right)$$

利用[(41)](#^eq-41),[(42)](#^eq-42)将 score 换回向量场：
$$\boxed{\tilde{u}_t(x \mid y) = (1-w) u_t^{\text{target}}(x \mid \varnothing) + w u_t^{\text{target}}(x \mid y)} \tag{65} $$ ^eq-65

其中 $u_t^{\text{target}}(x \mid \varnothing)$ 表示无条件（unconditional）向量场。$\varnothing$ 表示“空标签”。

> **重大意义**：无需单独训练分类器 $p_t(y \mid x)$，只需训练一个网络同时估计无条件和有条件输出，然后在推理时用线性插值即可增强条件 adherence。这是现代文生图/视频模型的标配技术。

#### CFG 训练目标

训练时以概率 $\eta$ 将标签 $y$ 替换为 $\varnothing$：
$$\mathcal{L}_{\text{CFM}}^{\text{CFG}}(\theta) = \mathbb{E}_{(z,y) \sim p_{\text{data}},\, t,\, x \sim p_t(\cdot \mid z)} \left[ \left\| u_t^\theta(x \mid y) - u_t^{\text{target}}(x \mid z) \right\|^2 \right] \tag{63} $$ ^eq-63

其中 $y$ 以概率 $\eta$ 被替换为 $\varnothing$。

> **使用场景**：Stable Diffusion、DALL-E、Movie Gen 等几乎所有 SOTA 图像/视频生成模型都在推理时使用 CFG，$w$ 通常取 2~5 甚至更高。

---

## 6. 大规模图像/视频生成器架构

### 6.1 神经网络架构

#### 6.1.1 条件嵌入

神经网络 $u_t^\theta(x \mid y)$ 接收三类输入：当前状态 $x$、时间 $t$、以及条件 $y$（文本 prompt 或类别标签）。为了让模型能充分利用这些异构信号，需要先把它们嵌入到同一维度的向量空间中。

**Time Embedding（Fourier 特征）**

直接将标量 $t$ 拼接到输入往往不足以捕捉高频的时间依赖。标准的做法是利用 Fourier 特征将 $t$ 映射到 $d$ 维空间：
$$\text{TimeEmb}(t) = \sqrt{\frac{2}{d}} \left[ \cos(2\pi w_1 t),\; \dots,\; \cos(2\pi w_{d/2} t),\; \sin(2\pi w_1 t),\; \dots,\; \sin(2\pi w_{d/2} t) \right]^T \tag{68} $$ ^eq-68

其中频率按几何级数选取：
$$w_i = w_{\min} \left(\frac{w_{\max}}{w_{\min}}\right)^{\frac{i-1}{d/2 - 1}}, \quad i = 1, \dots, d/2 \tag{69} $$ ^eq-69

**推导与直观**：
- 由于 $\sin^2 + \cos^2 = 1$，可直接验证 $\|\text{TimeEmb}(t)\| = 1$（归一化嵌入）。
- 低频分量 $w_{\min}$ 负责捕捉整体趋势，高频分量 $w_{\max}$ 负责捕捉局部快速变化。
- 这种嵌入等价于将时间映射到单位圆上的多个旋转角度，使神经网络更容易通过线性组合重构任意时间依赖函数。

**Prompt Embedding**
- **类别标签**：直接学习一个可训练的嵌入表，每个类对应一个向量。
- **文本 prompt**：通常使用冻结的预训练模型。
  - **CLIP**：在共享的图像-文本嵌入空间中对 prompt 进行编码，得到全局语义向量 $y \in \mathbb{R}^{d_{\text{CLIP}}}$。
  - **T5/UL2/ByT5**：将文本编码为序列嵌入 $y \in \mathbb{R}^{S \times k}$，提供更细粒度的 token-level 条件，便于模型通过 cross-attention 聚焦到 prompt 的特定词汇。

> **使用场景**：Stable Diffusion 3 同时使用了 CLIP（全局语义）和 T5-XXL（序列级细粒度）两种文本嵌入，以兼顾整体构图与局部细节。

---

#### 6.1.2 Diffusion Transformer (DiT)

DiT 是近年 SOTA 图像/视频生成器的核心骨干（如 Stable Diffusion 3、Movie Gen）。它将视觉 Transformer（ViT）的思想引入扩散模型，通过自注意力机制在 patch token 之间建模全局关系。

**从图像到序列**

给定图像 $x \in \mathbb{R}^{C \times H \times W}$，先将其切分成不重叠的 patch：

$$\text{Patchify}(x) \in \mathbb{R}^{N \times C'}, \quad C' = C P^2, \; N = \frac{H}{P} \cdot \frac{W}{P}$$

其中 $P$ 为 patch 大小。再通过线性投影映射到隐藏维度 $d$：

$$\text{PatchEmb}(x) = \text{Patchify}(x) W \in \mathbb{R}^{N \times d}, \quad W \in \mathbb{R}^{C' \times d}$$

Transformer 层的输入包括：
- 图像 token：$\tilde{x}_0 = \text{PatchEmb}(x) \in \mathbb{R}^{N \times d}$
- 时间嵌入：$\tilde{t} = \text{TimeEmb}(t) \in \mathbb{R}^{d}$
- 条件嵌入：$\tilde{y} = \text{PromptEmb}(y) \in \mathbb{R}^{S \times d}$

然后通过 $L$ 层 DiTBlock 迭代更新：
$$\tilde{x}_{i+1} = \text{DiTBlock}(\tilde{x}_i, \tilde{t}, \tilde{y}) \in \mathbb{R}^{N \times d}, \quad i = 0, \dots, L-1 \tag{70} $$ ^eq-70

最后将输出通过线性层再 depatchify，得到与输入同形状的速度场：

$$u = \text{Depatchify}(\tilde{x}_L \tilde{W}) \in \mathbb{R}^{C \times H \times W}, \quad \tilde{W} \in \mathbb{R}^{d \times C'}$$

**Scaled Dot-Product Attention**

Attention 的核心是计算 query-key 的相似度，并用它对 value 做加权平均：

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_h}}\right) V \in \mathbb{R}^{N \times d_h}$$

**推导**：除以 $\sqrt{d_h}$ 是为了防止内积值过大导致 softmax 进入饱和区（梯度极小）。若 $Q, K$ 的每个元素方差为 1，则 $QK^T$ 的每个元素方差为 $d_h$，因此除以 $\sqrt{d_h}$ 可将方差重新归一化为 1，保证梯度流畅。

**Multi-Head Attention（MHA）**

将每个 token 投影到 $h$ 个不同子空间并行计算注意力，再拼接回原始维度。设 $d_h = d/h$，对每个 head $j$：

$$\text{head}_j(x, z) = \text{Attn}(x W_Q^{(j)}, z W_K^{(j)}, z W_V^{(j)})$$

其中 $z = x$ 时为自注意力（self-attention），$z = y$ 时为交叉注意力（cross-attention）。最终：

$$\text{MHA}(x, z) = \text{Concat}\left(\text{head}_1, \dots, \text{head}_h\right) W_O \in \mathbb{R}^{N \times d}$$

> **直观**：不同 head 可以关注不同的特征子空间（如一个头关注颜色，另一个关注纹理），从而丰富模型的表达能力。

**Adaptive Normalization（AdaNorm / AdaLN）**

扩散模型中，时间 $t$ 不直接拼接到 token，而是通过预测逐通道的缩放/偏移参数来调制网络激活：

$$(\gamma, \beta) = g(\tilde{t}), \quad \gamma, \beta \in \mathbb{R}^{d}$$

$$\text{AdaNorm}_{\tilde{t}}(x) = (1 + \gamma) \odot \text{Norm}(x) + \beta$$

> **直观**：$\gamma$ 控制该层特征的重要性（放大/抑制），$\beta$ 控制特征偏移。由于 $\gamma, \beta$ 由时间 $t$ 决定，网络可以随着去噪/流进程动态调整内部表示的风格与强度。

**完整的 DiTBlock 更新**

$$x \leftarrow x + g_{\text{self}}(\tilde{t}) \odot \text{MHA}\big(\text{AdaNorm}_{\tilde{t}}(x), \text{AdaNorm}_{\tilde{t}}(x)\big)$$
$$x \leftarrow x + g_{\text{cross}}(\tilde{t}) \, \text{MHA}\big(\text{AdaNorm}_{\tilde{t}}(x), y\big)$$
$$x \leftarrow x + g_{\text{MLP}}(\tilde{t}) \, \text{MLP}\big(\text{AdaNorm}_{\tilde{t}}(x)\big)$$

其中 $g_{\cdot}(\tilde{t})$ 为可学习的门控参数，MLP 是逐位置的 feed-forward 网络。

> **使用场景**：对于纯类别条件（class-conditional）的 DiT，通常省略 cross-attention，仅用 AdaNorm 注入时间和类别信息。

---

#### 6.1.3 U-Net

U-Net 是扩散模型早期最主流的架构，本质是一种卷积神经网络，其输入与输出均为图像张量。它由对称的 encoder（下采样）和 decoder（上采样）组成，中间夹有 mid-coder，并且 encoder 与 decoder 之间通过 skip connections 相连：

$$x \xrightarrow{\mathcal{E}_1} x_1 \xrightarrow{\mathcal{E}_2} \dots \xrightarrow{\text{mid}} \dots \xrightarrow{\mathcal{D}_2} \hat{x}_1 \xrightarrow{\mathcal{D}_1} u$$

随着下采样，空间分辨率降低，通道数增加；上采样则相反。Skip connections 将高层语义与低层细节相融合，避免信息在下采样过程中丢失。现代 U-Net 变体还会在分辨率较低处插入 attention 层，以捕捉长距离依赖。

> **使用场景**：DDPM、 Stable Diffusion 1/2 的 backbone 均为 U-Net；而 Stable Diffusion 3 及后续模型已全面转向 DiT，因为 DiT 更易于在更大规模数据上扩展（scaling）。

---

### 6.2 隐空间建模：Variational Autoencoder (VAE)

直接在像素空间训练扩散/流模型对高分辨率图像/视频而言计算开销巨大。例如一张 $1024 \times 1024$ 的 RGB 图像维度 $d = 3 \times 1024^2 \approx 3 \times 10^6$，而视频还乘以帧数 $T$。因此，现代生成器无一例外地先在**低维隐空间**（latent space）中训练，再通过 decoder 映射回原始数据空间。

#### 6.2.1 标准自编码器

设 encoder 为 $\mu_\phi: \mathbb{R}^d \to \mathbb{R}^k$，decoder 为 $\mu_\theta: \mathbb{R}^k \to \mathbb{R}^d$，其中 $k \ll d$。核心目标是最小化重构误差：

$$\mathcal{L}_{\text{Recon}}(\phi, \theta) = \mathbb{E}_{x \sim p_{\text{data}}} \left[ \|\mu_\theta(\mu_\phi(x)) - x\|^2 \right]$$

**问题**：标准自编码器没有任何约束来保证隐分布 $p_{\text{latent}}(z)$ 是"良好"的（例如高斯状），因此我们无法保证在其上训练的生成模型能够稳定采样。

#### 6.2.2 变分自编码器（VAE）

VAE 将 encoder 和 decoder 从确定性函数放松为概率分布：
$$q_\phi(z \mid x) = \mathcal{N}\left(z; \mu_\phi(x), \text{diag}(\sigma_\phi^2(x))\right), \quad p_\theta(x \mid z) = \mathcal{N}\left(x; \mu_\theta(z), \sigma_\theta^2(z) I_d\right) \tag{71} $$ ^eq-71

**VAE 重构损失**

自然的重构目标是负对数似然：
$$\mathcal{L}_{\text{VAE-Recon}}(\phi, \theta) = -\mathbb{E}_{x \sim p_{\text{data}},\, z \sim q_\phi(\cdot \mid x)} \left[ \log p_\theta(x \mid z) \right] \tag{72} $$ ^eq-72

**从高斯密度推导 MSE 形式**：

对高斯解码器，展开 $\log p_\theta(x \mid z)$ 得：

$$\log p_\theta(x \mid z) = -\frac{d}{2}\log(2\pi) - \frac{d}{2}\log\sigma_\theta^2(z) - \frac{1}{2\sigma_\theta^2(z)}\|x - \mu_\theta(z)\|^2$$

去掉与参数无关的常数项，得到：
$$\mathcal{L}_{\text{VAE-Recon}} = \mathbb{E}_{x,z}\left[ \frac{1}{2\sigma_\theta^2(z)}\|x - \mu_\theta(z)\|^2 + \frac{d}{2}\log\sigma_\theta^2(z) \right] + \text{const} \tag{73} $$ ^eq-73

实践中，为避免学习方差带来的数值不稳定，常固定 $\sigma_\theta^2(z) = \tilde{\sigma}^2$（常数），于是：
$$\mathcal{L}_{\text{VAE-Recon}} = \mathbb{E}_{x,z}\left[ \frac{1}{2\tilde{\sigma}^2}\|x - \mu_\theta(z)\|^2 \right] + \text{const} \tag{74} $$ ^eq-74

这与标准自编码器的 MSE 损失等价，区别在于 $z$ 是从 encoder 分布中采样的。

**先验正则项（KL 散度）**

为了让隐分布接近容易采样的标准高斯，引入先验 $p_{\text{prior}} = \mathcal{N}(0, I_k)$，并用 KL 散度约束 encoder：
$$\mathcal{L}_{\text{VAE-Prior}}(\phi) = \mathbb{E}_{x \sim p_{\text{data}}} \left[ D_{\text{KL}}\big(q_\phi(\cdot \mid x) \;\|\; p_{\text{prior}}\big) \right] \tag{75} $$ ^eq-75

**KL 散度的定义与性质**：对两个概率密度 $q, p$，

$$D_{\text{KL}}(q \parallel p) = \int q(x) \log\frac{q(x)}{p(x)}\,\mathrm{d}x = \mathbb{E}_{X \sim q}\left[ \log\frac{q(X)}{p(X)} \right]$$

它满足：
$$D_{\text{KL}}(q \parallel p) \ge 0, \quad D_{\text{KL}}(q \parallel p) = 0 \;\Leftrightarrow\; q = p \tag{76-77} $$ ^eq-76-77

**例子 31：对角高斯间的 KL 散度**

设 $q = \mathcal{N}(\mu_q, \text{diag}(\sigma_q^2))$，$p = \mathcal{N}(\mu_p, \text{diag}(\sigma_p^2))$，则
$$D_{\text{KL}}(q \parallel p) = \frac{1}{2}\left( \mathcal{K}\left(\frac{\sigma_q^2}{\sigma_p^2}\right) + \frac{\|\mu_q - \mu_p\|^2}{\sigma_p^2} \right), \quad \mathcal{K}(\alpha) = \sum_{i=1}^d \left(\alpha_i - \log\alpha_i - 1\right) \tag{80} $$ ^eq-80

**推导**（以一维为例，多维求和即可）：

$$D_{\text{KL}}(q \parallel p) = \mathbb{E}_{x \sim q}\big[\log q(x) - \log p(x)\big]$$

代入高斯密度并展开：
$$= \frac{1}{2}\log\frac{\sigma_p^2}{\sigma_q^2} + \frac{1}{2\sigma_p^2}\mathbb{E}_q\left[\|x - \mu_p\|^2\right] - \frac{1}{2\sigma_q^2}\mathbb{E}_q\left[\|x - \mu_q\|^2\right] \tag{81} $$ ^eq-81

利用 $\mathbb{E}_q[\|x - \mu_q\|^2] = \sigma_q^2$ 以及 $\mathbb{E}_q[\|x - \mu_p\|^2] = \sigma_q^2 + \|\mu_q - \mu_p\|^2$，回代后整理即得 [(80)](#^eq-80)。

**VAE 的 KL 项具体形式**

当 $p_{\text{prior}} = \mathcal{N}(0, I_k)$ 时，$\mu_p = 0$，$\sigma_p^2 = 1$，于是：
$$D_{\text{KL}}\big(q_\phi(\cdot \mid x) \parallel \mathcal{N}(0, I_k)\big) = \frac{1}{2}\sum_{j=1}^k \left( \mu_{\phi,j}^2(x) + \sigma_{\phi,j}^2(x) - \log\sigma_{\phi,j}^2(x) - 1 \right) \tag{82} $$ ^eq-82

> **直观**：第一项 $\|\mu_\phi(x)\|^2$ 将隐变量均值拉向 0；第二三项 $\sum(\sigma^2 - \log\sigma^2 - 1)$ 在 $\sigma^2 = 1$ 处取最小值 0，鼓励方差为 1。

**完整的 VAE 训练目标**

将重构项与先验项加权（$\beta \ge 0$ 控制 trade-off）：
$$\mathcal{L}_{\text{VAE}}(\phi, \theta) = \mathcal{L}_{\text{VAE-Recon}}(\phi, \theta) + \beta \, \mathcal{L}_{\text{VAE-Prior}}(\phi) \tag{78} $$ ^eq-78

代入具体形式：
$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{x, z \sim q_\phi}\left[ \underbrace{\frac{1}{2\sigma_\theta^2(z)}\|x - \mu_\theta(z)\|^2}_{\text{重构误差}} + \underbrace{\frac{d}{2}\log\sigma_\theta^2(z)}_{\text{decoder 置信度}} + \underbrace{\frac{\beta}{2}\mathcal{K}(\sigma_\phi^2(x))}_{\text{方差约束}} + \underbrace{\frac{\beta}{2}\|\mu_\phi(x)\|^2}_{\text{均值约束}} \right] + \text{const} \tag{83} $$ ^eq-83

**重参数化技巧（Reparameterization Trick）**

损失 [(83)](#^eq-83) 需要对 $z \sim q_\phi(z \mid x)$ 求期望，而 $q_\phi$ 的分布本身依赖于参数 $\phi$，直接用梯度下降会有困难。重参数化技巧将随机性从参数中分离出来：

$$\epsilon \sim \mathcal{N}(0, I_k), \quad z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon \quad \Rightarrow \quad z \sim q_\phi(\cdot \mid x)$$

此时，对 $\phi$ 的梯度可以畅通无阻地反向传播（因为 $\epsilon$ 不依赖于 $\phi$），而对 $\epsilon$ 的期望可用 Monte Carlo 采样近似。

> **直观解释**：我们可以将 encoder 的均值 $\mu_\phi(x)$ 看作是对 $x$ 的“最佳隐坐标估计”，而 $\sigma_\phi(x) \odot \epsilon$ 则是在该估计附近引入可控的随机扰动。重参数化保证了我们可以同时优化编码的准确性（均值接近真实隐变量）和编码的不确定性（方差适度）。

**实践备注**
1. **$\beta$ 的选择与 KL warm-up**：过大的 $\beta$ 会导致 encoder 忽略输入 $x$ 而直接输出 $\mathcal{N}(0, I_k)$（posterior collapse）。常见做法是 KL warm-up：前几个 epoch 让 $\beta$ 从 0 逐渐增大到目标值。现代 latent diffusion 中通常取 $\beta \ll 1$。
2. **Decoder 方差**：学习解码器方差 $\sigma_\theta^2(z)$ 容易数值不稳定，大多数实现将其固定为常数，使重构项退化为 MSE。
3. **感知损失**：仅用像素级 MSE 会导致重建图像过于平滑，实际系统常加入 perceptual loss（用预训练网络的特征空间距离）来提升锐度和语义保真度。
4. **对抗与混合目标**：为进一步增强视觉真实感，可将 VAE 与对抗判别器结合（VAE-GAN），但会引入额外的优化不稳定性。

**隐空间生成流程**

训练时，从 $q_\phi(z \mid x)$ 采样隐变量，在其上训练 flow/diffusion 模型；推理时，先从隐空间的生成模型中采样 $z$，再通过 decoder 的均值输出 $\mu_\theta(z)$ 映射回数据空间（取均值而非采样，可避免 decoder 噪声带来的伪影）。

> **使用场景**：Stable Diffusion、Movie Gen 等几乎所有 SOTA 图像/视频 diffusion 模型均采用 **Latent Diffusion** 范式——先在 VAE 隐空间训练 flow/diffusion 模型，再通过 decoder 映射回像素/帧空间。

---

### 6.3 案例研究：Stable Diffusion 3 与 Meta Movie Gen

**Stable Diffusion 3** 使用条件 Flow Matching（Algorithm 3/5 的目标）和 Classifier-Free Guidance，在预训练 VAE 的隐空间中进行训练。其 backbone 为 Multi-Modal DiT（MM-DiT），除了图像 patch 的自注意力外，还通过 cross-attention 同时 attending 到 CLIP 全局嵌入和 T5-XXL 序列嵌入，从而在 8B 参数规模下实现高质量的文本到图像生成。采样使用 50 步 Euler 模拟， guidance scale $w$ 通常取 2.0–5.0。

**Meta Movie Gen Video** 将上述技术扩展到视频领域。视频数据 $x \in \mathbb{R}^{T \times C \times H \times W}$ 先通过 Temporal Autoencoder（TAE）压缩到隐空间（时间和空间维度均压缩 8 倍）。其 backbone 同样是 DiT，patchify 操作同时覆盖时空维度，并通过 cross-attention 注入 UL2（推理）、ByT5（字符级细节）和 MetaCLIP（图像-文本对齐）三种文本嵌入。最大模型参数量达 30B。

---

## 7. 离散扩散模型（CTMC / 语言模型）

上述内容针对连续空间 $\mathbb{R}^d$。对于离散数据（如文本），使用 **Continuous-Time Markov Chain (CTMC)** 模型。

### 7.1 CTMC 模型

状态空间 $S = \mathcal{V}^d$（$d$ 为序列长度，$\mathcal{V}$ 为词表），$X_t \in S$。

**Rate Matrix** $Q_t(y \mid x)$ 满足：
- $Q_t(y \mid x) \ge 0$ for $y \neq x$
- $Q_t(x \mid x) = -\sum_{y \neq x} Q_t(y \mid x)$

小步长近似转移概率：
$$p_{t+h \mid t}(y \mid x) \approx 1_{y=x} + h Q_t(y \mid x) =: \tilde{p}_{t+h \mid t}(y \mid x) \tag{88} $$ ^eq-88

#### Factorized CTMC

由于 $|S| = V^d$ 指数级大，实际使用 **factorized** 约束：$Q_t(y \mid x) = 0$ 除非 $y$ 与 $x$ 仅在一个位置不同。此时网络输出形状为 $d \times V$（线性可处理）。

### 7.2 训练 CTMC 模型

#### 离散条件概率路径

$$p_0(\cdot \mid z) = p_{\text{init}}, \quad p_1(\cdot \mid z) = \delta_z$$

边缘路径：$p_t(x) = \sum_{z \in S} p_t(x \mid z) p_{\text{data}}(z)$

#### 例子 35：Factorized Mixture Path

设 $p_{\text{init}}(x) = \prod_{j=1}^d p_{\text{init}}^{(j)}(x_j)$，调度器 $0 \le \kappa_t \le 1$（$\kappa_0=0, \kappa_1=1$）。定义：

$$p_t(x \mid z) = \prod_{j=1}^d \left[ (1-\kappa_t) p_{\text{init}}^{(j)}(x_j) + \kappa_t \delta_{z_j}(x_j) \right]$$

等价采样：对每个位置独立采样 mask $m_j \sim \text{Bernoulli}(\kappa_t)$，噪声 token $\xi_j \sim p_{\text{init}}^{(j)}$，然后

$$x_j = m_j z_j + (1 - m_j) \xi_j$$

> **直观**：以概率 $1-\kappa_t$ “破坏”该位置 token，以概率 $\kappa_t$ 保留原始 token。与 Gaussian 路径不同，这里没有“移动”概率质量，只是对两个分布做淡入淡出。

#### 命题 2：Kolmogorov Forward Equation (KFE)

对 CTMC，$X_t \sim p_t$ 当且仅当：

$$\frac{\mathrm{d}}{\mathrm{d}t} p_t(x) = \sum_{y \in S} Q_t(x \mid y) p_t(y)$$

#### 定理 36：离散 Marginalization Trick

边缘 rate matrix：
$$Q_t(y \mid x) = \sum_{z \in S} Q_t^z(y \mid x) \frac{p_t(x \mid z) p_{\text{data}}(z)}{p_t(x)} = \sum_{z \in S} Q_t^z(y \mid x) p_{1 \mid t}(z \mid x) \tag{90} $$ ^eq-90

满足 $X_0 \sim p_{\text{init}}, \; X_t \text{ CTMC of } Q_t \Rightarrow X_t \sim p_t$，特别地 $X_1 \sim p_{\text{data}}$。

#### 例子 37：Factorized Mixture Path 的条件 Rate Matrix

$$Q_t^z(v_i, j \mid x_j) = \frac{\dot{\kappa}_t}{1 - \kappa_t} \left( \delta_{z_j}(v_i) - \delta_{x_j}(v_i) \right)$$

> 简单理解：若位置 $j$ 的当前值 $x_j \neq z_j$，则它只能“跳”到目标值 $z_j$，速率为 $\frac{\dot{\kappa}_t}{1-\kappa_t}$；若已在 $z_j$，则不再跳转。

#### 定理 38：Factorized Mixture Path 的边缘 Rate Matrix
$$Q_t(v_i, j \mid x) = \frac{\dot{\kappa}_t}{1 - \kappa_t} \left( p_{1 \mid t}(z_j = v_i \mid x) - \delta_{x_j}(v_i) \right) \tag{95} $$ ^eq-95

> **关键洞察**：边缘 rate matrix 完全由后验概率 $p_{1 \mid t}(z_j = v_i \mid x)$ 决定。这意味着训练 CTMC 生成模型等价于训练一个**逐位置的分类器**！

#### 离散 Flow Matching Loss

用神经网络 $f_\theta(x, t)$ 输出每位置 logits，经 softmax 得 $p_{1 \mid t}^\theta(z_j \mid x)$。Loss 为逐 token 的交叉熵：

$$\mathcal{L}_{\text{DFM}}(\theta) = \mathbb{E}_{z, t, x \sim p_t(\cdot \mid z)} \left[ \sum_{j=1}^d -\log p_{1 \mid t}^\theta(z_j \mid x) \right]$$

#### 例子 39：Masked Diffusion Language Model (MDLM)

在词表中增加 `[mask]` token，设 $p_{\text{init}} = \delta_{[\text{mask}]^d}$（全 mask 序列）。这是当前离散扩散语言模型（如 LLaDA）的基础。

> **使用场景**：LLaDA2.0 等百亿参数级离散扩散语言模型均采用上述训练范式。

---

## 8. 算法汇总

以下将 `lecture_notes.md` 中出现的全部 8 个核心算法用数学步骤形式整理，便于在 LaTeX 或 markdown 中直接引用公式。

### Algorithm 1：Flow Model 采样（Euler 方法）

**输入**：神经网络向量场 $u_t^\theta$，步数 $n$。  
**输出**：生成样本 $X_1$。

1. 初始化 $t = 0$，步长 $h = 1/n$。
2. 采样初始状态 $X_0 \sim p_{\text{init}}$。
3. **for** $i = 1, \dots, n$ **do**
   $$X_{t+h} = X_t + h \, u_t^\theta(X_t)$$
4. 更新 $t \leftarrow t + h$。
5. **end for**
6. **return** $X_1$。

> **说明**：这是最简单的 ODE 数值积分，每一步沿当前向量场方向走一个固定步长。

---

### Algorithm 2：Diffusion Model 采样（Euler–Maruyama）

**输入**：神经网络 $u_t^\theta$，步数 $n$，扩散系数 $\sigma_t$。  
**输出**：生成样本 $X_1$。

1. 初始化 $t = 0$，步长 $h = 1/n$。
2. 采样初始状态 $X_0 \sim p_{\text{init}}$。
3. **for** $i = 1, \dots, n$ **do**
   - 采样噪声 $\epsilon \sim \mathcal{N}(0, I_d)$。
   - 更新状态：
     $$X_{t+h} = X_t + h \, u_t^\theta(X_t) + \sigma_t \sqrt{h} \, \epsilon$$
   - 更新 $t \leftarrow t + h$。
4. **end for**
5. **return** $X_1$。

> **说明**：在 Euler 步的基础上注入方差为 $h\sigma_t^2$ 的高斯噪声，对应 SDE 的离散化。

---

### Algorithm 3：Flow Matching 训练（Gaussian CondOT 路径）

**输入**：数据集 $z \sim p_{\text{data}}$，神经网络 $u_t^\theta$，优化器。  
**输出**：训练后的参数 $\theta$。

对每一个 mini-batch 重复：
1. 采样数据 $z$。
2. 采样时间 $t \sim \text{Unif}[0,1]$。
3. 采样噪声 $\epsilon \sim \mathcal{N}(0, I_d)$。
4. 构造含噪样本：
   $$x = t z + (1 - t) \epsilon$$
5. 计算损失：
   $$\mathcal{L}(\theta) = \big\| u_t^\theta(x) - (z - \epsilon) \big\|^2$$
6. 梯度更新 $\theta$。

> **说明**：CondOT 路径下 $\alpha_t = t$，$\beta_t = 1 - t$，条件向量场恰好是 $z - \epsilon$。该算法实现了 simulation-free training。

---

### Algorithm 4：Score Matching 训练（Gaussian 路径）

**输入**：数据集 $z \sim p_{\text{data}}$，score 网络 $s_t^\theta$ 或 noise predictor $\epsilon_t^\theta$。  
**输出**：训练后的参数 $\theta$。

对每一个 mini-batch 重复：
1. 采样数据 $z$。
2. 采样时间 $t \sim \text{Unif}[0,1]$。
3. 采样噪声 $\epsilon \sim \mathcal{N}(0, I_d)$。
4. 构造含噪样本：
   $$x_t = \alpha_t z + \beta_t \epsilon$$
5. 计算损失（二选一）：
   - 若使用 score 网络：
     $$\mathcal{L}(\theta) = \big\| s_t^\theta(x_t) + \frac{\epsilon}{\beta_t} \big\|^2$$
   - 若使用 noise predictor（DDPM 形式）：
     $$\mathcal{L}(\theta) = \big\| \epsilon_t^\theta(x_t) - \epsilon \big\|^2$$
6. 梯度更新 $\theta$。

> **说明**：noise predictor 形式在 $\beta_t \to 0$ 时更数值稳定，是 DDPM 的经典实现。

---

### Algorithm 5：Classifier-Free Guidance 训练（Gaussian 路径）

**输入**：配对数据集 $(z, y) \sim p_{\text{data}}$，神经网络 $u_t^\theta$，丢弃概率 $p$。  
**输出**：训练后的参数 $\theta$。

对每一个 mini-batch 重复：
1. 采样 $(z, y)$。
2. 采样时间 $t \sim \text{Unif}[0,1]$。
3. 采样噪声 $\epsilon \sim \mathcal{N}(0, I_d)$。
4. 构造含噪样本：
   $$x = \alpha_t z + \beta_t \epsilon$$
5. 以概率 $p$ 将条件 $y$ 替换为空标签 $\varnothing$。
6. 计算损失：
   $$\mathcal{L}(\theta) = \big\| u_t^\theta(x \mid y) - (\dot{\alpha}_t z + \dot{\beta}_t \epsilon) \big\|^2$$
7. 梯度更新 $\theta$。

**推理时**：使用 guidance scale $w \ge 1$ 对条件与无条件输出做外推：
$$\tilde{u}_t(x \mid y) = (1 - w) \, u_t^\theta(x \mid \varnothing) + w \, u_t^\theta(x \mid y)$$

> **说明**：训练时随机丢弃标签使网络同时学会无条件与有条件生成；推理时通过 $w > 1$ 显著增强 prompt adherence。

---

### Algorithm 6：$\boldsymbol{\beta}$-VAE 训练

**输入**：数据集 $x \sim p_{\text{data}}$，encoder $(\mu_\phi, \log \sigma_\phi^2)$，decoder $\mu_\theta$，隐变量维度 $k$，常数 $\beta \ge 0$，$\tilde{\sigma}^2 > 0$。  
**输出**：训练后的参数 $(\phi, \theta)$。

对每一个 mini-batch $\{x_i\}_{i=1}^B$ 重复：
1. Encoder 前向：
   $$\mu_i \leftarrow \mu_\phi(x_i), \quad \log \sigma_i^2 \leftarrow \log \sigma_\phi^2(x_i)$$
2. 采样噪声 $\epsilon_i \sim \mathcal{N}(0, I_k)$。
3. 重参数化采样隐变量：
   $$z_i = \mu_i + \exp\!\left(\frac{1}{2} \log \sigma_i^2\right) \odot \epsilon_i$$
4. Decoder 前向：
   $$\hat{x}_i \leftarrow \mu_\theta(z_i)$$
5. 计算重构损失：
   $$\mathcal{L}_{\text{recon}} \leftarrow \frac{1}{B} \sum_{i=1}^B \frac{1}{2\tilde{\sigma}^2} \big\| x_i - \hat{x}_i \big\|^2$$
6. 计算 KL 损失（先验为 $\mathcal{N}(0, I_k)$）：
   $$\mathcal{L}_{\text{KL}} \leftarrow \frac{1}{B} \sum_{i=1}^B \frac{1}{2} \sum_{j=1}^k \left( \mu_{i,j}^2 + \sigma_{i,j}^2 - \log \sigma_{i,j}^2 - 1 \right)$$
7. 总损失与梯度更新：
   $$\mathcal{L} \leftarrow \mathcal{L}_{\text{recon}} + \beta \, \mathcal{L}_{\text{KL}}, \qquad (\phi, \theta) \leftarrow \text{grad\_update}(\mathcal{L})$$

> **说明**：$\beta$ 控制隐空间正则化的强度；$\beta = 1$ 即为标准 VAE，$\beta \ll 1$ 常见于 latent diffusion 的预训练 autoencoder。

---

### Algorithm 7：Factorized CTMC 采样

**输入**：factorized rate 网络 $Q_t^\theta$，初始分布 $p_{\text{init}}$，步数 $n$，词表 $\mathcal{V}$，序列长度 $d$。  
**输出**：离散样本 $X_1 \in \mathcal{V}^d$。

1. 初始化 $t = 0$，步长 $h = 1/n$。
2. 采样初始状态 $X_0 \sim p_{\text{init}}$，记 $X_0 = (X_0^{(1)}, \dots, X_0^{(d)})$。
3. **for** $i = 1, \dots, n$ **do**
   - 由 $Q_t^\theta(\cdot \mid X_t)$ 计算逐位置 jump rates $\{q_j(v)\}_{v \in \mathcal{V}}$。
   - **for** $j = 1, \dots, d$（可并行）**do**
     - 令 $x = X_t^{(j)}$。
     - 定义离散转移概率：
       $$\tilde{p}_{j,t}(v \mid x) = \begin{cases}
       h \, q_j(v), & v \neq x, \\[6pt]
       1 - h \sum_{v' \neq x} q_j(v'), & v = x.
       \end{cases}$$
     - 采样 $X_{t+h}^{(j)} \sim \text{Categorical}\big(\{\tilde{p}_{j,t}(v \mid x)\}_{v \in \mathcal{V}}\big)$。
   - **end for**
   - 更新 $t \leftarrow t + h$。
4. **end for**
5. **return** $X_1$。

> **说明**：factorized 假设保证每一步只改变一个位置，从而使转移概率从指数级 $|\mathcal{V}|^d$ 降到线性级 $d \times |\mathcal{V}|$。

---

### Algorithm 8：Factorized CTMC 训练（离散扩散）

**输入**：数据集 $z = (z_1, \dots, z_d) \in \mathcal{V}^d \sim p_{\text{data}}$，初始边际 $p_{\text{init}}^{(j)}$，调度器 $\kappa_t \in [0,1]$，posterior 网络 $f_\theta$ 输出每位置 logits。  
**输出**：训练后的参数 $\theta$。

对每一个训练迭代重复：
1. 采样数据点 $z$。
2. 采样时间 $t \sim \text{Unif}[0,1]$，并令 $\kappa \leftarrow \kappa_t$。
3. 按 factorized mixture path 采样噪声状态 $x$：
   - **for** $j = 1, \dots, d$（可并行）**do**
     - 采样 mask $m_j \sim \text{Bernoulli}(\kappa)$。
     - 采样噪声 token $\xi_j \sim p_{\text{init}}^{(j)}$。
     - 构造位置 $j$ 的状态：
       $$x_j = m_j z_j + (1 - m_j) \xi_j$$
   - **end for**
   - 组装 $x \leftarrow (x_1, \dots, x_d)$。
4. 用网络预测终点 token 的后验分布：
   $$\ell_j(\cdot) \leftarrow f_\theta(x, t)_j, \qquad p_{1|t}^\theta(v \mid x)_j = \text{Softmax}\big(\ell_j\big)(v)$$
5. 计算离散 Flow Matching 损失（逐 token 交叉熵）：
   $$\mathcal{L}_{\text{DFM}}(\theta) \leftarrow \sum_{j=1}^d \Big[ -\log p_{1|t}^\theta(z_j \mid x)_j \Big]$$
6. 梯度更新 $\theta$。

> **说明**：该损失等价于在每个位置训练一个分类器，目标是根据当前含噪/含 mask 的序列 $x$ 预测原始干净 token $z_j$。LLaDA 等离散扩散语言模型即采用此范式。

