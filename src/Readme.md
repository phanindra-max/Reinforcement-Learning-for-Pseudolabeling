# How to Run This Project

### Step 1: Install Dependencies

Navigate to `src/component/` folder and install the required Python libraries/dependencies:

```bash
cd src/component/
pip install -r requirements.txt
```

### Step 2: Configure the Experiment

All experiment settings are defined in `src/component/config.py`.

Before running the project, you can adjust:

- **Dataset:**  
  `data_source = "mnist"` (or `"cifar10"`, `"cifar100"`)

- **Model Architecture:**  
  `model_name = "cnn"` (or `"mlp"`, `"resnet18"`)

- **RL Training Settings:**  
  number of episodes, batch size, learning rates

- **Downstream Classifier Hyperparameters**  
  epochs, optimizer settings

- **To train the RL agent for pseudo-labeling, set:**

```python
use_true_labels = False
```
- **To run the baseline experiment with ground truth labels, set:**
```python
use_true_labels = True
```

### Step 3: Run the Training Script

To start the experiment, execute the main script in `src/component/` directory:
```bash
python main.py
```
This will:

- load and split the dataset  
- initialize the classifier and policy network  
- run the RL pseudo-labeling loop  
- log metrics (accuracy, error rate, reward, etc.)  

### Step 4: Monitor Results with TensorBoard

Training logs are saved in the `src/component/runs/` directory as tensorflow event files. To launch TensorBoard:

- Identify the latest folder inside `runs/`(e.g., `pseudo_labeling_20251112-141908`)
- Run the following command in `src/component/` directory:
```bash
tensorboard --logdir=runs/<add_folder_name_here>
```
- Once executed, a localhost link will be generated like `http://localhost:6006/`; you can copy this link and paste in your browser to view the dashboard.