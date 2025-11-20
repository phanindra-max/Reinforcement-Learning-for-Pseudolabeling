# How to Run This Project

### Step 1: Install Dependencies

Navigate to src/component/ folder and install the required Python libraries/dependencies:

```bash
cd src/component/
pip install -r requirements.txt
```

### Step 2: Configure the Experiment

All experiment settings are managed in the `src/component/config.py` file.

Before running, you can adjust hyperparameters in this file. 

-   **To train the RL agent for pseudo-labeling:**
    ```python
    use_true_labels = False
    ```

-   **To run the baseline experiment with ground truth labels:**
    ```python
    use_true_labels = True
    ```

### Step 3: Run the Training Script

To start the experiment, navigate to the `src/component/` directory and execute the main script:

```bash
python main.py
```

### Step 4: Monitor Results with TensorBoard

The results, including accuracy, error rate, and rewards, are logged in the `src/component/runs/` directory to visualize them in TensorBoard. Following are the steps to visualize the results:

- Check the latest folder under `runs/` that contains events.out.tfevents.* and note the folder name(e.g.pseudo_labeling_20251112-141908)
- Navigate to `runs/` folder and run the following command.
```bash
tensorboard --logdir=runs/<add_folder_name_here>
```
- Once executed, a localhost link will be generated like `http://localhost:6006/`; you can copy this link and paste in your browser to view the dashboard.
