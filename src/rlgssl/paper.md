Of course. Here is the text from the PDF converted into a clean, LLM-friendly format.

***

# Reinforcement Learning-Guided Semi-Supervised Learning

**Authors:** Marzi Heidari, Hanping Zhang, Yuhong Guo
**Affiliation:** School of Computer Science, Carleton University, Ottawa, Canada
**arXiv ID:** arXiv:2405.01760v1 [cs.LG] 2 May 2024

## Abstract

In recent years, semi-supervised learning (SSL) has gained significant attention due to its ability to leverage both labeled and unlabeled data to improve model performance, especially when labeled data is scarce. However, most current SSL methods rely on heuristics or predefined rules for generating pseudo-labels and leveraging unlabeled data. They are limited to exploiting loss functions and regularization methods within the standard norm. In this paper, we propose a novel Reinforcement Learning (RL) Guided SSL method, RLGSSL, that formulates SSL as a one-armed bandit problem and deploys an innovative RL loss based on weighted reward to adaptively guide the learning process of the prediction model. RLGSSL incorporates a carefully designed reward function that balances the use of labeled and unlabeled data to enhance generalization performance. A semi-supervised teacher-student framework is further deployed to increase the learning stability. We demonstrate the effectiveness of RLGSSL through extensive experiments on several benchmark datasets and show that our approach achieves consistent superior performance compared to state-of-the-art SSL methods.

## 1. Introduction

Semi-supervised learning (SSL) is a significant research area in the field of machine learning, addressing the challenge of effectively utilizing limited labeled data alongside abundant unlabeled data. SSL techniques bridge the gap between supervised and unsupervised learning, offering a practical solution when labeling large amounts of data is prohibitively expensive or time-consuming. The primary goal of SSL is to leverage the structure and patterns present within the unlabeled data to improve the learning process, generalization capabilities, and overall performance of the prediction model.

Over the past few years, there has been considerable interest in developing various SSL methods, and these approaches have found success in a wide range of applications, from computer vision [1] to natural language processing [2] and beyond [3, 4].

Within the SSL domain, a range of strategies has been devised to effectively utilize the information available in both labeled and unlabeled data. Broadly, SSL approaches can be categorized into three key paradigms: regularization-based, mean-teacher-based, and pseudo-labeling methodologies.

* **Regularization-based approaches** form a fundamental pillar of SSL [5, 6, 7]. These methods revolve around the core idea of promoting model robustness against minor perturbations in the input data. A quintessential example in this category is Virtual Adversarial Training (VAT) [5]. VAT capitalizes on the introduction of adversarial perturbations to the input space, thereby ensuring the model's predictions maintain consistency.
* The second category, **Mean-teacher based methods**, encapsulates a distinct class of SSL strategies that leverage the concept of temporal ensembling. This technique aids in the stabilization of the learning process by maintaining an exponential moving average of model parameters over training iterations. Mean Teacher [8] notably pioneered this paradigm with their Mean Teacher model, illustrating its efficacy across numerous benchmark tasks.
* Lastly, the category of **Pseudo-labeling approaches** has attracted attention due to its simplicity and effectiveness. These methods employ the model's own predictions on unlabeled data as "pseudo-labels" to augment the training process. The MixMatch [1] framework stands as one of the leading representatives of this category, demonstrating the potential of these methods in the low-data regime.

Despite these advancements, achieving high performance with limited labeled data continues to be a significant challenge in SSL, often requiring intricate design decisions and the careful coordination of multiple loss functions. In this paper, we propose to approach SSL outside the conventional design norms by developing a Reinforcement Learning Guided Semi-Supervised Learning (RLGSSL) method. RL has emerged as a promising direction for addressing learning problems, with the potential to bring a fresh perspective to SSL. It offers a powerful framework for decision-making and optimization, which can be harnessed to discover novel and effective strategies for utilizing the information present in both labeled and unlabeled data.

In RLGSSL, we formulate SSL as a bandit problem, where the prediction model serves as the policy function, and pseudo-labeling acts as the actions. We define a simple reward function that balances the use of labeled and unlabeled data and improves generalization capacity by leveraging linear data interpolation, while the prediction model is trained under the standard RL framework to maximize the empirical expected reward. Formulating the SSL problem as such an RL task allows our approach to dynamically adapt and respond to the data. Moreover, we further deploy a teacher-student learning framework to enhance the stability of learning. Additionally, we integrate a supervised learning loss to improve and accelerate the learning process. This new SSL framework has the potential to pave the way for more robust, flexible, and adaptive SSL methods. We evaluate the proposed method through extensive experiments on benchmark datasets.

The contribution of this work can be summarized as follows:
* We propose RLGSSL, a novel Reinforcement Learning-based approach that effectively tackles SSL by leveraging RL's power to learn effective strategies for generating pseudo-labels and guiding the learning process.
* We design a prediction assessment reward function that encourages the learning of accurate and reliable pseudo-labels while maintaining a balance between the usage of labeled and unlabeled data, thus promoting better generalization performance.
* We introduce a novel integration framework that combines the power of both RL loss and standard semi-supervised loss for SSL, providing a more adaptive and data-driven approach that has the potential to lead to more accurate and robust SSL models.

Extensive experiments demonstrate that our proposed method outperforms state-of-the-art approaches in SSL.

## 2. Related Work

### 2.1 Semi-Supervised Learning

Existing SSL approaches can be broadly classified into three primary categories: regularization-based methods, teacher-student-based methods, and pseudo-labeling techniques.

* **Regularization-Based Methods:** A prevalent research direction in SSL focuses on regularization-based methods, which introduce additional terms to the loss function to promote specific properties of the model. For instance, the II-model [6] and Temporal-Ensemble [6] incorporate consistency regularization into the loss function. Virtual Adversarial Training (VAT) [5] is yet another regularization-based technique that aims to make deep neural networks robust to adversarial perturbations.
* **Teacher-Student-Based Methods:** These techniques train a student network to align its predictions with those of a teacher network on unlabeled data. Mean Teacher (MT) [8], a prominent example, leverages an exponential moving average (EMA) on the teacher model. Other methods include MT + Fast SWA [9], Smooth Neighbors on Teacher Graphs (SNTG) [10], and Interpolation Consistency Training (ICT) [11].
* **Pseudo-Labeling Methods:** Pseudo-labeling is an effective way to extend the labeled set when the number of labels is limited. Methods include Pseudo-Label [12], MixMatch [1], ReMixMatch [13], FixMatch [15], and Meta Pseudo-Labels [21]. Others use label propagation [16, 17] or dynamic confidence thresholds [19, 20]. Early examples include Co-Training [22] and Tri-Training [23].

### 2.2 Reinforcement Learning

Reinforcement Learning (RL) is a field that focuses on optimizing an agent's decision-making by maximizing cumulative reward through interactions with its environment [24]. RL has been applied to various problems like network architecture search [25], text generation [26, 27], and online planning [28]. Recently, RL from Human Feedback (RLHF) [29, 30, 31] has been used to fine-tune Large Language Models (LLMs) like ChatGPT [32].

This approach often frames the problem as a one-armed bandit problem [24, 33], where the objective is to determine the optimal action for a given state in a single step. The bandit problem has applications in economics for optimizing decisions to maximize profits [34, 35, 36]. Modern RL techniques can be applied to bandit problems by formulating them as a Single-Step Markov Decision Process (SSMDP) [37].

## 3. The Proposed Method

We consider a semi-supervised learning setting with a small set of labeled samples, $D_l = (X^l, Y^l) = \{(x_i^l, y_i^l)\}_{i=1}^{N^l}$, and a large set of unlabeled samples, $D_u = X^u = \{x_i^u\}_{i=1}^{N^u}$, where $N^u \gg N^l$. The goal is to train a C-class classifier $f_{\theta}: \mathcal{X} \rightarrow \mathcal{Y}$ that generalizes well.

We present RLGSSL, which formulates SSL as a one-armed bandit problem and uses an RL loss to guide the process. A teacher-student framework is incorporated to enhance stability.

**(Description of Figure 1)**: The framework shows unlabeled data $X^u$ feeding into a "Teacher Policy" ($\theta_T$), which generates pseudo-labels $Y^u$. Labeled data $X^l$ feeds into a "Student Policy" ($\theta_S$). The student's predictions on labeled data are compared with the true labels $Y^l$ to compute a supervised loss ($\mathcal{L}_{sup}$). The student and teacher policies are compared to compute a consistency loss ($\mathcal{L}_{cons}$). The pseudo-labeled data and labeled data are combined into "Mixup Data," which is used by a "Reward Function" to calculate a reward. This reward informs the RL loss ($\mathcal{L}_{rl}$), which updates the student policy.

### 3.1 Reinforcement Learning Formulation for SSL

We treat SSL as a one-armed bandit problem, which is a single-step Markov Decision Process (MDP) [37]. The key components are:

* **State (s):** The state is the observed data from the SSL problem: $s = (X^l, Y^l, X^u)$.
* **Action (a) and Policy Function ($\pi_{\theta}$):** The classifier $f_{\theta}$ serves as the policy function $\pi_{\theta}$. Taking an action is equivalent to making probabilistic predictions on the unlabeled data: $Y^u = P_{\theta}(X^u) = \pi_{\theta}(\cdot|s)$. Each action is a probability vector $y_i^u = P_{\theta}(x_i^u)$, which acts as a soft pseudo-label.

#### 3.1.1 Reward Function

The reward function evaluates the action (prediction) to guide the learning process. We use a data mixup strategy [38] to create new data points by linearly interpolating between labeled and pseudo-labeled data.

To handle the size discrepancy between labeled ($N^l$) and unlabeled ($N^u$) data, we replicate the labeled dataset $r = \lceil\frac{N^u}{N^l}\rceil$ times. A mixup data point is generated as:
$x_i^m = \mu x_i^u + (1-\mu)x_i^l$
$y_i^m = \mu y_i^u + (1-\mu)y_i^l$

The reward function measures the negative mean squared error (MSE) between the model's prediction on the mixup data and the mixup label:
$\mathcal{R}(s,a;sg[\theta]) = -\frac{1}{C \cdot N^m} \sum_{i=1}^{N^m} ||P_{\theta}(x_i^m) - y_i^m||_2^2$

The stop gradient operator $sg[\cdot]$ ensures the reward function is used for assessment only, not direct model updating, adhering to RL principles.

#### 3.1.2 Reinforcement Learning Loss

We use a KL-divergence weighted negative reward as the RL loss:
$\mathcal{L}_{rl} = -\mathbb{E}_{y_i^u \sim \pi_{\theta}} KL(e, y_i^u) \mathcal{R}(s,a;sg[\theta])$
where $e$ is a uniform distribution vector. This encourages the model's predictions to be more discriminative (less uniform).

### 3.2 Teacher-Student Framework for RL-Guided SSL

We use a teacher-student model to improve stability. The teacher model's parameters ($\theta_T$) are an exponential moving average (EMA) of the student model's parameters ($\theta_S$):
$\theta_T = \beta \theta_T + (1-\beta)\theta_S$

The stable teacher model is used to generate pseudo-labels ($Y^u = P_{\theta_T}(X^u)$). We augment the RL loss with a supervised loss on labeled data and a consistency loss on unlabeled data.

* **Supervised Loss:**
    $\mathcal{L}^{sup} = \mathbb{E}_{(x^l, y^l) \in D^l} [l_{CE}(P_{\theta_s}(x^l), y^l)]$
* **Consistency Loss:**
    $\mathcal{L}^{cons} = \mathbb{E}_{x^u \in D^u} [l_{KL}(P_{\theta_s}(x^u), P_{\theta_T}(x^u))]$

### 3.3 Training Algorithm for RL-Guided SSL

The final learning objective combines the three losses:
$\mathcal{L}(\theta_S) = \mathcal{L}_{rl} + \lambda_1 \mathcal{L}_{sup} + \lambda_2 \mathcal{L}_{cons}$

A stochastic batch-wise gradient descent algorithm is used to minimize this objective.

#### Algorithm 1: Pseudo-Label Based Policy Gradient Descent

1.  **Input:** Labeled data $D^l$, unlabeled data $D^u$, initialized $\theta_S, \theta_T$.
2.  **For** each iteration:
3.  Compute soft pseudo-labels for all $x_i^u \in D^u$ using the teacher model: $y_i^u = P_{\theta_T}(x_i^u)$.
4.  Generate mixup data $D^m$ by combining labeled and pseudo-labeled data.
5.  **For** each step within the iteration:
6.  Draw a batch of data from $D^m$.
7.  Calculate the reward function $\mathcal{R}$.
8.  Compute the total loss objective $\mathcal{L}(\theta_S)$.
9.  Update student parameters $\theta_S$ via gradient descent.
10. Update teacher parameters $\theta_T$ via EMA.

## 4. Experiments

### 4.1 Experimental Setup

* **Datasets:** CIFAR-10, CIFAR-100, and SVHN.
* **Architectures:** 13-layer CNN (CNN-13) and Wide-Residual Network (WRN-28).
* **Implementation Details:** For RLGSSL, we set batch size to 128, $\lambda_1$ to 0.1, and $\lambda_2$ to 0.1. We pre-train the model for 50 epochs with the Mean-Teacher algorithm, then train RLGSSL for 400 epochs.

### 4.2 Comparison Results

RLGSSL was compared against numerous state-of-the-art SSL algorithms.

**Table 1: Performance on CIFAR-10 and CIFAR-100 (CNN-13, Test Error %)**
| Dataset | Labels | ICT [11] | RLGSSL (Ours) |
| :--- | :--- | :--- | :--- |
| CIFAR-10 | 1000 | 12.44 (0.57) | **9.15 (0.57)** |
| CIFAR-10 | 2000 | 9.05 | **7.18 (0.24)** |
| CIFAR-10 | 4000 | 6.90 (0.11) | **6.11 (0.10)** |
| CIFAR-100 | 4000 | 36.92 (0.45) | **33.62 (0.54)** |
| CIFAR-100 | 10000 | 32.24 (0.16) | **29.12 (0.20)** |

**Table 2: Performance on SVHN (CNN-13, Test Error %)**
| Dataset | Labels | SNTG [10] | RLGSSL (Ours) |
| :--- | :--- | :--- | :--- |
| SVHN | 500 | 3.99 (0.24) | **3.12 (0.07)** |
| SVHN | 1000 | 3.86 (0.27) | **3.05 (0.04)** |

**Table 3: Performance with WRN-28 Network (Test Error %)**
| Dataset | Labels | MixMatch [1] | RLGSSL (Ours) |
| :--- | :--- | :--- | :--- |
| CIFAR-10 | 1000 | 7.75 (0.32) | **4.92 (0.25)** |
| CIFAR-10 | 2000 | 7.03 (0.15) | **4.24 (0.10)** |
| CIFAR-10 | 4000 | 6.24 (0.06) | **3.52 (0.06)** |
| CIFAR-100 | 10000 | 30.84 (0.29) | **28.82 (0.22)** |
| SVHN | 1000 | 5.35 (0.19) | **1.92 (0.05)** |
*(Note: For SVHN/1000, the best competitor was Meta Pseudo-Labels [21] with 1.99 (0.07))*

Across all datasets, label counts, and network architectures, RLGSSL consistently achieved superior performance compared to state-of-the-art methods.

### 4.3 Ablation Study

An ablation study on CIFAR-100 showed that removing any component of RLGSSL (RL loss, supervised loss, consistency loss, EMA update, or mixup in reward) increased the test error. The most significant performance drop occurred when the RL loss ($\mathcal{L}_{rl}$) was removed, highlighting its critical role. Altering the reward function (e.g., using a constant reward or a different distance metric) also degraded performance.

### 4.4 Hyperparameter Analysis

**(Description of Figure 2)**: Sensitivity analysis for $\lambda_1$ (supervised loss weight) and $\lambda_2$ (consistency loss weight) showed that the best performance was achieved when both values were around 0.1. Very low values failed to leverage labeled data or enforce consistency effectively, while very high values led to overfitting. This confirms that the RL loss is the main component, with the other two acting as augmenting terms.

## 5. Conclusion

In this paper, we presented RLGSSL, a novel approach that integrates RL principles into SSL to address the limitations of conventional methods. RLGSSL uses an RL-optimized reward function to adaptively guide learning and a teacher-student framework to enhance stability. Extensive evaluations demonstrated that RLGSSL consistently outperforms existing state-of-the-art techniques across multiple benchmark datasets, confirming the effectiveness and generalizability of our approach. The results highlight the significant potential of integrating RL into SSL.

## References

[A full list of 42 references is provided in the original paper, including works by Berthelot et al. (MixMatch), Miyato et al. (VAT), Tarvainen & Valpola (Mean Teacher), Sutton & Barto (Reinforcement learning: An introduction), and others.]