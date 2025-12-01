import os
import json
import torch
import numpy as np

from config import *
from model import DownstreamModel, PolicyGradient
from utils import load_source_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_state(model, img):
    with torch.no_grad():
        feats = model.extract_features(img.unsqueeze(0)).squeeze(0)
        probs = model.get_predictions(img.unsqueeze(0))
        entropy = -torch.sum(probs * torch.log(probs + 1e-9))
        state = torch.cat([feats, probs, entropy.unsqueeze(0)])
    return state.cpu().numpy()


def main():
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"trained_models_{model_name}_{data_source}")
    policy_name = "best_policy_gradient"

    # Load data using existing util; we only need unlabeled_x
    _, _, _, _, _, _, test_x, _, dataset_info = load_source_data(data_source=data_source,num_labeled=num_labeled, num_unlabeled=num_unlabeled, num_validation=num_validation)
    unlabeled_x = test_x

    input_shape = dataset_info["input_shape"]
    num_classes = dataset_info["num_classes"]
    feature_dim = int(np.prod(input_shape))

    downstream = DownstreamModel(model_name, input_shape, num_classes, feature_dim, DEVICE)
    downstream.load_model(name="best_downstream", save_dir=weights_dir)

    feat_dim = downstream.extract_features(unlabeled_x[:1].to(DEVICE)).shape[1]
    state_dim = feat_dim + num_classes + 1
    agent = PolicyGradient(state_dim, num_classes + 1, gamma=gamma, alpha_pi=alpha_pi, alpha_v=alpha_v, device=DEVICE, policy=policy_type, hidden_dim=hidden_dim)
    agent.load_model(name=policy_name, save_dir=weights_dir)

    pseudo_labels = []
    skipped = []

    for idx in range(len(unlabeled_x)):
        img = unlabeled_x[idx].to(DEVICE)
        state = build_state(downstream, img)
        action = agent.policy(state)
        if action < num_classes:
            probs = downstream.get_predictions(img.unsqueeze(0)).detach().cpu()
            confidence = float(torch.max(probs))
            pseudo_labels.append({"index": idx, "pseudo_label": int(action), "confidence": confidence})
        else:
            skipped.append(int(idx))

    os.makedirs(weights_dir, exist_ok=True)
    with open(os.path.join(weights_dir, "pseudo_labels.json"), "w") as f:
        json.dump(pseudo_labels, f, indent=2)
    with open(os.path.join(weights_dir, "skipped.json"), "w") as f:
        json.dump(skipped, f, indent=2)

    print(f"Saved {len(pseudo_labels)} pseudo-labels and {len(skipped)} skipped samples to {weights_dir}")


if __name__ == "__main__":
    main()
