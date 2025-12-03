# Reinforcement Learning for Pseudo-Labeling

## Overview  

We have implemented a reinforcement learning framework for pseudo-labeling that learns to assign pseudo-labels to unlabeled data in a way that improves downstream classifier accuracy. 

**Core idea**:  
The RL agent decides which unlabeled samples should receive labels. The classifier retrains using these pseudo-labels. If validation error improves, the agent receives a positive reward.


## Key Features  
- Semi-supervised learning with RL  
- REINFORCE policy gradient optimization  
- Accuracy-based reward: `old_error − new_error`  
- Pseudo-label batching 
- Custom Gym RL environment  
- CNN-based downstream classifier  
- TensorBoard for visualization  
- Modular and extensible code design  

---

## Project Structure
```
capstone
├── demo/                           # Presentation Recording
├── presentation/                   # Preliminary and final presentation
├── reports/                        # Final report, progress reports
├── research_paper/                 # Final research paper
├── src/
│   ├── components/                 # Main project source code
│   │   ├── main.py                 # RL + classifier training loop
│   │   ├── env.py                  # Custom Gym environment
│   │   ├── model.py                # downstream model + policy network
│   │   ├── utils.py                # Data loading & preprocessing
│   │   ├── config.py               # Hyperparameters
│   │   └── requirements.txt        # Python dependencies
│   ├── tests/
│   │   └── trained_models_<model_name>_<data_source>.zip  # Saved weights (zip)
│   └── Readme.md                   # Instructions for running code
├── .gitignore
└── Readme.md                       # Main project README (high-level)
```

---

## Method Summary

### 1. Data Loading & Splitting  
Dataset is divided into three sets:
- **Labeled set** – used to train the initial classifier  
- **Unlabeled set** – used by the RL agent for pseudo-labeling  
- **Validation set** – used to compute error and reward  

Splitting sizes are configured in `config.py`.

---

### 2. Initial Classifier Training  
The CNN classifier is trained on the labeled set.  
Validation accuracy is computed → error rate = `1 - accuracy`.  
This establishes the baseline used to compute rewards.

---

### 3. RL Episode (One Full Pass Over Unlabeled Data)  
For each unlabeled image:
1. Extract features, probability vector, and entropy  
2. Build a state vector  
3. RL agent chooses an action:  
   - **0–9** → assign that pseudo-label  
   - **10** → skip sample  
4. Pseudo-labels accumulate in a buffer  
5. Once the buffer reaches `batch_size`, classifier retrains  
6. New validation error is computed  
7. Reward = `old_error − new_error`  
8. Policy updates once using REINFORCE  
9. Buffer clears, next batch begins  

One full pass over the unlabeled dataset = one RL episode.

---

## Full Pipeline Diagram

```mermaid
flowchart TD

A[Load Dataset] --> B[Split into Labeled / Unlabeled / Validation]
B --> C[Train Initial Classifier]
C --> D[Compute Baseline Error]

subgraph RL_Episode [RL Episode: Full Pass Over Unlabeled]
    D --> E[Init Pseudo-label Buffer]
    E --> F{For Each Unlabeled Sample}

    F --> G[Build State from Features]
    G --> H[Policy Action: Label or Skip]

    H -->|Skip| I[Apply Skip Penalty]
    H -->|Label| J[Store Pseudo-label]

    I --> K{Buffer Full?}
    J --> K

    K -->|No| F
    K -->|Yes| L[Retrain Classifier]

    L --> M[Compute New Error]
    M --> N[Reward = Old - New Error]
    N --> O[Update Policy]
    O --> P[Clear Buffer and Update Old Error]
end

P --> Q{More Unlabeled?}
Q -->|Yes| F
Q -->|No| R[End Episode and Save Models]
```

---

### 4. Research Background

This approach is inspired by concepts from:

- [A review of Pseudo-Labeling For Computer Vision] (https://arxiv.org/pdf/2408.07221)
- [RLGSSL (Reinforcement Learning for Graph-Based SSL)](https://arxiv.org/pdf/2405.01760)
- [Selective Pseudo-Labeling with Reinforcement Learning for Semi-Supervised Domain Adaptation](https://arxiv.org/pdf/2012.03438)
