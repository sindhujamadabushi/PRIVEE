import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import argparse
sys.path.append('defenses')
from rounding import round_confidences
from noising import add_Gaussian_noise_raw
from fh_ope import FH_OPE, encrypt_confidence_batch
from torch_generate_noise1 import add_Gaussian_noise_privee
from torch_generate_noise2 import add_Gaussian_noise_priveeplus

from utils import load_dat, batch_split
from models import APModel, PPModel, LearningCoordinator, APModelkParties,LearningCoordinatorkParties

parser = argparse.ArgumentParser(
    description="Run GRNA_ADULT_NN with selectable defense and (ε) parameters"
)
parser.add_argument(
    "--defense",
    type=str,
    choices=["rounding", "noising", "fh-ope", "privee", "priveeplus"],
    required=True,
    help="Which defense to apply"
)

# If defense == 'rounding', we need 'decimals'
parser.add_argument(
    "--decimals",
    type=int,
    default=1,
    help="Number of decimal places to round to (only for rounding)."
)

# If defense == 'noising', we need exactly two ε values:
parser.add_argument(
    "--epsilon",
    type=float,
    default=0.1,
    help="Two ε values for noising (only for defense=noising)."
)

parser.add_argument(
    "--attack_strength",
    type=float,
    default=0.5,
    help="nothing"
)

parser.add_argument(
    "--organization_num",
    type=int,
    default=2,
    help="nothing"
)

parser.add_argument(
    "--attack_type",
    type=str,
    default='grna',
    help="nothing"
)

args = parser.parse_args()



batch_size = 128
random_seed = 1992
epochs = 6
num_classes = 10
epochs_grna = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
defense = args.defense
decimals = args.decimals
attack_str = args.attack_strength
epsilon = args.epsilon
organization_num = args.organization_num
attack_type = args.attack_type
delta = 1e-5
sensitivity = 0.1

def gradient_inversion_attack_k(
    org_models,            # nn.ModuleList length K (one encoder per party)
    top_model,             # coordinator head taking concatenated embeddings
    x_known,               # [B, d_known] tensor for the known party
    true_conf,             # [B, C] target confidence/softmax (defended or raw)
    known_idx=0,           # which party is "known"
    lr=1e-3,
    iters=500,
    clamp_min=0.0,
    clamp_max=1.0,
):
    """
    Reconstructs *all unknown parties'* inputs jointly by matching confidences.
    Returns: (final_loss, list_of_x_hats) where list_of_x_hats[i] is reconstructed
    tensor for party i (None for known_idx).
    """
    K = len(org_models)
    x_known = x_known.to(device)
    true_conf = true_conf.to(device)

    # Create learnable tensors for unknown parties
    x_hats = []
    params = []
    B = x_known.shape[0]
    for i, enc in enumerate(org_models):
        if i == known_idx:
            x_hats.append(None)
        else:
            # d_i = enc.net[0].in_features  # assumes first layer is Linear(in_features, ...)
            if hasattr(enc, "fc1") and isinstance(enc.fc1, nn.Linear):
                d_i = enc.fc1.in_features
            elif hasattr(enc, "net") and isinstance(enc.net[0], nn.Linear):
                d_i = enc.net[0].in_features
            else:
                # fallback: find first Linear in the module
                d_i = None
                for m in enc.modules():
                    if isinstance(m, nn.Linear):
                        d_i = m.in_features
                        break
                if d_i is None:
                    raise AttributeError(f"Cannot infer in_features for encoder {type(enc).__name__}")
            
            
            x_hat = torch.zeros((B, d_i), device=device, requires_grad=True)
            x_hats.append(x_hat)
            params.append(x_hat)

    optimizer = torch.optim.Adam(params, lr=lr)

    for it in range(iters):
        optimizer.zero_grad()

        # Forward: build embeddings for each party
        party_embs = []
        for i, enc in enumerate(org_models):
            if i == known_idx:
                party_embs.append(enc(x_known))
            else:
                party_embs.append(enc(x_hats[i]))

        logits = top_model(party_embs)
        pred_conf = F.softmax(logits, dim=1)

        loss = F.mse_loss(pred_conf, true_conf)
        loss.backward()
        optimizer.step()

        # keep reconstructions in valid range
        for i in range(K):
            if i != known_idx:
                with torch.no_grad():
                    x_hats[i].clamp_(clamp_min, clamp_max)

        if it % 100 == 0:
            print(f"[GIA-K] Iter {it}/{iters}  MSE: {loss.item():.6f}")

    return loss, x_hats

def grna_k(
    x_known_all,           # [N, d_known] tensor of known party features (entire training set)
    conf_scores_tensor,    # [N, C] target confidences (raw or defended)
    party_encoders,        # nn.ModuleList length K
    coordinator,           # top model
    known_idx,             # which party is known
    epochs_grna,
    batch_idxs_list,
    lr=0.1,
    clamp_min=0.0,
    clamp_max=1.0,
):
    """
    Optimizes *all unknown parties'* full-dataset inputs to match confidence vectors.
    Returns: (final_loss, list_of_x_hats_full) — x_hats_full[i] is [N, d_i] or None for known.
    """
    # Freeze encoder & head
    for enc in party_encoders: enc.eval()
    coordinator.eval()
    for p in party_encoders.parameters(): p.requires_grad_(False)
    for p in coordinator.parameters():    p.requires_grad_(False)

    x_known_all = x_known_all.to(device)
    conf_scores_tensor = conf_scores_tensor.to(device)

    # Initialize unknown parties' inputs for all N samples
    N = x_known_all.shape[0]
    x_hats_full = []
    params = []
    for i, enc in enumerate(party_encoders):
        if i == known_idx:
            x_hats_full.append(None)
        else:
            # d_i = enc.net[0].in_features
            if hasattr(enc, "fc1") and isinstance(enc.fc1, nn.Linear):
                d_i = enc.fc1.in_features
            elif hasattr(enc, "net") and isinstance(enc.net[0], nn.Linear):
                d_i = enc.net[0].in_features
            else:
                d_i = None
                for m in enc.modules():
                    if isinstance(m, nn.Linear):
                        d_i = m.in_features
                        break
                if d_i is None:
                    raise AttributeError(f"Cannot infer in_features for encoder {type(enc).__name__}")
            x_hat_full = torch.randn(N, d_i, device=device, requires_grad=True)
            x_hats_full.append(x_hat_full)
            params.append(x_hat_full)

    optimizer = torch.optim.Adam(params, lr=lr)
    mse = nn.MSELoss()

    for epoch in range(epochs_grna):
        batch_losses = []
        optimizer.zero_grad()

        for batch_idxs in batch_idxs_list:
            idx = torch.tensor(batch_idxs, dtype=torch.long, device=device)

            party_embs = []
            for i, enc in enumerate(party_encoders):
                if i == known_idx:
                    party_embs.append(enc(x_known_all.index_select(0, idx)))
                else:
                    party_embs.append(enc(x_hats_full[i].index_select(0, idx)))

            gen_conf = coordinator(party_embs)
            gt_conf  = conf_scores_tensor.index_select(0, idx)
            batch_losses.append(mse(gen_conf, gt_conf))

        loss = torch.stack(batch_losses).mean()
        loss.backward()
        optimizer.step()

        # clamp after each epoch
        for i in range(len(x_hats_full)):
            if i != known_idx:
                with torch.no_grad():
                    x_hats_full[i].clamp_(clamp_min, clamp_max)

        # print(f"[GRNA-K] epoch {epoch+1}/{epochs_grna}  MSE {loss.item():.6f}")

    return loss, x_hats_full



file_path = "/home/msindhuja/PRIVEE-VFL/PFI-VFL/datasets/MNIST/MNIST.csv"
X = pd.read_csv(file_path)
y = X['class']
X = X.drop(['class'], axis=1)

N, dim = X.shape
columns = list(X.columns)

dim = X.shape[1]

# 2) attack_str ∈ (0,1) tells us the fraction for the active party
# w0 = int(round(attack_str * dim))
# # Make sure neither side gets 0 or >dim:
# w0 = max(1, min(dim-1, w0))
# w1 = dim - w0

# attribute_split_array = np.array([w0, w1], dtype=int)
# print(f"Successful attribute split: active party gets {w0} features, passive gets {w1} features")

w0 = int(round(attack_str * dim))
w0 = max(1, min(dim-1, w0))  # ensure at least one feature on each side

if organization_num == 2:
    attribute_split_array = np.array([w0, dim - w0], dtype=int)
else:
    # Distribute the remainder across the remaining K-1 parties
    remain = dim - w0
    w_rest = np.full(organization_num-1, remain // (organization_num-1), dtype=int)
    w_rest[:remain % (organization_num-1)] += 1
    attribute_split_array = np.concatenate([[w0], w_rest]).astype(int)

print("Successful attribute split:", attribute_split_array.tolist(),
      " (sum =", attribute_split_array.sum(), ")")



attribute_groups = []
attribute_start_idx = 0
for organization_idx in range(organization_num):
    attribute_end_idx = attribute_start_idx + attribute_split_array[organization_idx]
    attribute_groups.append(columns[attribute_start_idx : attribute_end_idx])
    attribute_start_idx = attribute_end_idx
    
for organization_idx in range(organization_num):
    print('The number of attributes held by Organization {0}: {1}'.format(organization_idx, len(attribute_groups[organization_idx])))
    
vertical_splitted_data = {}
encoded_vertical_splitted_data = {}

for organization_idx in range(organization_num):
                    
    vertical_splitted_data[organization_idx] = \
        X[attribute_groups[organization_idx]].values
    
    encoded_vertical_splitted_data = vertical_splitted_data
            
X_train_vertical_FL = {}
X_test_vertical_FL = {}

for organization_idx in range(organization_num):
    test_set_size = 10000  

    X_test_vertical_FL[organization_idx] = encoded_vertical_splitted_data[organization_idx][-test_set_size:]

    if organization_idx == 0:
        y_test = y[-test_set_size:]
        
        X_train = encoded_vertical_splitted_data[organization_idx][:-test_set_size]
        y_train = y[:-test_set_size]

        X_train_vertical_FL[organization_idx] = X_train
    else:
        X_train = encoded_vertical_splitted_data[organization_idx][:-test_set_size]
        dummy_labels = np.zeros(len(X_train))  # not used for passive parties

        X_train_vertical_FL[organization_idx] = X_train

train_loader_list, test_loader_list = [], []
for organization_idx in range(organization_num):

    X_train_vertical_FL[organization_idx] = torch.from_numpy(X_train_vertical_FL[organization_idx]).float()
    X_test_vertical_FL[organization_idx] = torch.from_numpy(X_test_vertical_FL[organization_idx]).float()

    train_loader_list.append(DataLoader(X_train_vertical_FL[organization_idx], batch_size=batch_size))
    test_loader_list.append(DataLoader(X_test_vertical_FL[organization_idx], batch_size=len(X_test_vertical_FL[organization_idx]), shuffle=False))
    
y_train = torch.from_numpy(y_train.to_numpy()).long()
y_test = torch.from_numpy(y_test.to_numpy()).long()
train_loader_list.append(DataLoader(y_train, batch_size=batch_size))
test_loader_list.append(DataLoader(y_test, batch_size=batch_size))
batch_idxs_list = batch_split(len(X_train_vertical_FL[0]), batch_size, 'mini-batch')

x_ap_all = X_train_vertical_FL[0].to(device) 
N = x_ap_all.shape[0]
batch_idxs_list = batch_split(N, batch_size, 'mini-batch')

# ap_model = APModel(in_features=len(X_train_vertical_FL[0][0]), hidden_layer=128, out_features=64)
# pp_model = PPModel(in_features=len(X_train_vertical_FL[1][0]), hidden_layer=128, out_features=64)
# coordinator = LearningCoordinator(64, 64, num_classes)

in_features_list = [len(X_train_vertical_FL[i][0]) for i in range(organization_num)]
ap_model = APModelkParties(in_features_list=in_features_list, hidden_layer=128, out_features=64).to(device)
coordinator = LearningCoordinatorkParties(out_features_per_party=[64]*organization_num, num_classes=num_classes).to(device)

# ap_model = ap_model.to(device)
# pp_model = pp_model.to(device)
# coordinator = coordinator.to(device)

# optimizer = optim.Adam(list(ap_model.parameters()) + list(pp_model.parameters()) + list(coordinator.parameters()),lr=0.0001/2)
optimizer = optim.Adam(list(ap_model.parameters()) + list(coordinator.parameters()), lr=0.0001/2)


criterion = nn.CrossEntropyLoss()
conf_scores_tensor = torch.zeros(N, num_classes).to(device)
conf_scores_tensor = torch.zeros(N, num_classes).to(device)

total_correct          = 0
total_correct_defense  = 0
total_samples          = 0

ope = FH_OPE()  # Initialize once at the beginning


for epoch in range(epochs):
    ap_model.train()
    # pp_model.train()
    coordinator.train()

    total_loss = 0

    epoch_correct          = 0
    epoch_correct_defense  = 0
    epoch_samples          = 0

    for batch_num, batch_tuple in enumerate(zip(*train_loader_list)):
    # train_loader_list = [X_party0_loader, X_party1_loader, ..., X_partyK-1_loader, y_loader]
        *x_parts, labels = batch_tuple
        idxs   = torch.tensor(batch_idxs_list[batch_num], dtype=torch.long, device=device)
        x_parts = [x.to(device) for x in x_parts]
        labels  = labels.to(device)

        # Forward: K party embeddings -> single head
        party_embs = ap_model(x_parts)                   # list of [B, 64]
        confid     = coordinator(party_embs)             # [B, num_classes]  (logits)
        confid     = F.softmax(confid, dim=1)

        conf_scores_tensor[idxs] = confid.detach()

        loss = criterion(confid, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_raw = torch.argmax(confid, dim=1)
        epoch_correct += (pred_raw == labels).sum().item()

        if defense == 'rounding':
            rounded_scores  = round_confidences(confid.detach(), decimals)
            pred_defense    = torch.argmax(rounded_scores, dim=1)
            epoch_correct_defense += (pred_defense == labels).sum().item()

        if defense == 'noising':
            noised_scores = add_Gaussian_noise_raw(confid.detach(), epsilon, delta, sensitivity).float()
            pred_defense = torch.argmax(noised_scores, dim=1)
            epoch_correct_defense += (pred_defense == labels).sum().item()

        if defense == 'fh-ope':
            enc_scores = encrypt_confidence_batch(confid.detach(), ope, device=device)
            pred_defense = torch.argmax(enc_scores, dim=1)
            epoch_correct_defense += (pred_defense == labels).sum().item()

        if defense == 'privee':
            enc_scores = add_Gaussian_noise_privee(confid.detach(), epsilon, delta, sensitivity)
            pred_defense = torch.argmax(enc_scores, dim=1)
            epoch_correct_defense += (pred_defense == labels).sum().item()

        if defense == 'priveeplus':
            enc_scores = add_Gaussian_noise_priveeplus(confid.detach(), epsilon, delta, sensitivity)
            pred_defense = torch.argmax(enc_scores, dim=1)
            epoch_correct_defense += (pred_defense == labels).sum().item()

        epoch_samples += labels.size(0)
        total_loss    += loss.item()
    total_correct         = epoch_correct
    total_correct_defense = epoch_correct_defense
    total_samples         = epoch_samples

    print("epoch done ", epoch)

accuracy_raw = 100.0 * total_correct / total_samples
if defense != 'no defense':
    accuracy_defense = 100.0 * total_correct_defense / total_samples
    print(f"Final raw accuracy:      {accuracy_raw:.2f}%")
    print(f"Final defense accuracy:  {accuracy_defense:.2f}%")
    print("difference in accuracy=",accuracy_raw-accuracy_defense)
else:
    print(f"Final raw accuracy:      {accuracy_raw:.2f}%")

if attack_type == 'gia':
    mse_no_defense, _ = gradient_inversion_attack_k(
        org_models=ap_model.models,
        top_model=coordinator,
        x_known=x_ap_all.to(device),          # party 0 known
        true_conf=conf_scores_tensor.to(device),
        known_idx=0, lr=1e-3, iters=500
    )
    print(f"GRNA/GIA MSE without defense: {mse_no_defense.item():.6f}")

if attack_type == 'grna':
    loss_grna_no_defense, xhats_full_no_def = grna_k(
        x_known_all=x_ap_all.to(device),            # party 0 (active) features for all N
        conf_scores_tensor=conf_scores_tensor.to(device),  # target confidences (raw)
        party_encoders=ap_model.models,             # encoders inside APModelkParties
        coordinator=coordinator,                    # LearningCoordinatorkParties
        known_idx=0,                                # which party is known (active)
        epochs_grna=epochs_grna,
        batch_idxs_list=batch_idxs_list,            # already computed mini-batch indices
        lr=0.1
    )
    print(f"[GRNA-K] MSE without defense: {loss_grna_no_defense.item():.6f}")

if defense != 'no defense':
    if defense == 'rounding':
        defended_conf = round_confidences(conf_scores_tensor.detach(), decimals)
    elif defense == 'noising':
        defended_conf = add_Gaussian_noise_raw(
        conf_scores_tensor.detach(), epsilon, delta, sensitivity
    ).float().to(device)   # <— add .float()

    elif defense == 'privee':
        defended_conf = add_Gaussian_noise_privee(
            conf_scores_tensor.detach(),
            epsilon, delta, sensitivity
        )

    elif defense == 'priveeplus':
        defended_conf = add_Gaussian_noise_priveeplus(
            conf_scores_tensor.detach(),
            epsilon, delta, sensitivity
        )
    elif defense == 'fh-ope':
        # Instead of encrypted values, reconstruct ordering and convert to a pseudo-confidence
        enc_conf = encrypt_confidence_batch(conf_scores_tensor.detach(), ope, device='cpu')
        # For each row, rank the ciphertexts, convert rank to float-based pseudo-confidence
        ranks = torch.argsort(enc_conf, dim=1, descending=True)
        pseudo_conf = torch.zeros_like(enc_conf, dtype=torch.float)
        for i in range(enc_conf.shape[0]):
            for j, cls in enumerate(ranks[i]):
                pseudo_conf[i][cls] = float(num_classes - j) / num_classes
        defended_conf = pseudo_conf.to(device)

if attack_type == 'gia':
    mse_with_defense, _ = gradient_inversion_attack_k(
    org_models=ap_model.models,
    top_model=coordinator,
    x_known=x_ap_all.to(device),
    true_conf=defended_conf.to(device),
    known_idx=0, lr=1e-3, iters=500
    )
    print(f"GRNA/GIA MSE with defense:   {mse_with_defense.item():.6f}")


if attack_type == 'grna':
    loss_grna_with_defense, xhats_full_def = grna_k(
        x_known_all=x_ap_all.to(device),
        conf_scores_tensor=defended_conf.to(device),   # defended confidences
        party_encoders=ap_model.models,
        coordinator=coordinator,
        known_idx=0,
        epochs_grna=epochs_grna,
        batch_idxs_list=batch_idxs_list,
        lr=0.1
    )
    print(f"[GRNA-K] MSE with defense:   {loss_grna_with_defense.item():.6f}")

