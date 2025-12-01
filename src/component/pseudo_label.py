import os, zipfile
import json
import torch
import numpy as np

from config import *
from model import DownstreamModel, PolicyGradient
from utils import load_source_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_state(model, img):
    """
        Build the RL state representation for a single image.
        The state is constructed by concatenating:
          1. Feature embeddings from the downstream model
          2. Class probabilities predicted by the downstream model
          3. Entropy (uncertainty) of the prediction distribution

        Args:
            model: Trained DownstreamModel instance
            img (Tensor): Input image tensor (C×H×W)

        Returns:
            np.ndarray: 1D state vector used by the RL agent
        """
    with torch.no_grad():
        feats = model.extract_features(img.unsqueeze(0)).squeeze(0)
        probs = model.get_predictions(img.unsqueeze(0))
        entropy = -torch.sum(probs * torch.log(probs + 1e-9))
        state = torch.cat([feats, probs, entropy.unsqueeze(0)])
    return state.cpu().numpy()

def unzip_model(model_name, data_source):
    """
        Unzip pretrained model weights from the tests directory into the current folder.

        Args:
            model_name (str): Name of the base model (e.g., "resnet18").
            data_source (str): Dataset identifier (e.g., "cifar10").

        Returns:
            str: Path where the weights were extracted.
        """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dest_dir = os.path.join(current_dir, f"inference_results_{model_name}_{data_source}")
    models_path = os.path.abspath(os.path.join
                                  (current_dir, "..", "tests", f"trained_models_{model_name}_{data_source}",
                                  f"{model_name}_{data_source}_best_models.zip"))
    if not os.path.exists(models_path):
        raise FileNotFoundError(f"Weights zip not found in directory: {models_path}")
    with zipfile.ZipFile(models_path, "r") as zf:
        zf.extractall(dest_dir)
    return dest_dir

def main():
    """
        Run inference to generate pseudo-labels using:
          - A trained downstream classifier
          - A trained RL PolicyGradient agent

        Steps:
          1. Unzip weights into inference directory
          2. Load test data as unlabaled set
          3. Load trained downstream classifier
          4. Load trained RL agent
          5. For each test image:
               - Build RL state
               - Agent selects action (label or skip)
               - If action is a label -> save pseudo-label with confidence
               - If action is skip -> record index
          6. Save pseudo-labels and skipped indices into JSON files
        """
    weights_dir = unzip_model(model_name, data_source)
    policy_name = "best_policy_gradient"

    _, _, _, _, _, _, test_x, _, dataset_info = load_source_data(data_source=data_source,num_labeled=num_labeled, num_unlabeled=num_unlabeled, num_validation=num_validation)
    unlabeled_x = test_x

    input_shape = dataset_info["input_shape"]
    num_classes = dataset_info["num_classes"]
    feature_dim = int(np.prod(input_shape))

    # loading downstream model
    downstream = DownstreamModel(model_name, input_shape, num_classes, feature_dim, DEVICE)
    downstream.load_model(name="best_downstream", save_dir=weights_dir)

    # loading RL agent
    feat_dim = downstream.extract_features(unlabeled_x[:1].to(DEVICE)).shape[1]
    state_dim = feat_dim + num_classes + 1
    agent = PolicyGradient(state_dim, num_classes + 1, gamma=gamma, alpha_pi=alpha_pi, alpha_v=alpha_v, device=DEVICE, policy=policy_type, hidden_dim=hidden_dim)
    agent.load_model(name=policy_name, save_dir=weights_dir)

    pseudo_labels = []
    skipped = []

    for idx in range(len(unlabeled_x)):
        img = unlabeled_x[idx].to(DEVICE)
        state = build_state(downstream, img)
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, device = DEVICE).unsqueeze(0)
            logits = agent.pi(state_tensor)
            action = int(torch.argmax(logits, dim=1).item()) # choosing max probability as action
        if action < num_classes:
            probs = downstream.get_predictions(img.unsqueeze(0)).detach().cpu()
            confidence = float(torch.max(probs))
            if confidence >= 0.15:
                pseudo_labels.append({"index": idx, "pseudo_label": int(action), "confidence": confidence})
            else: # skipping less confident samples
                skipped.append(int(idx))
        else:
            skipped.append(int(idx))

    # saving pseudo-labels and skipped indices
    os.makedirs(weights_dir, exist_ok=True)
    with open(os.path.join(weights_dir, "pseudo_labels.json"), "w") as f:
        json.dump(pseudo_labels, f, indent=2)
    with open(os.path.join(weights_dir, "skipped.json"), "w") as f:
        json.dump(skipped, f, indent=2)

    print(f"Saved {len(pseudo_labels)} pseudo-labels and {len(skipped)} skipped samples to {weights_dir}")


if __name__ == "__main__":
    main()
