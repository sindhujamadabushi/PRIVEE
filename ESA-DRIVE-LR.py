import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch_generate_noise1 import add_Gaussian_noise
from models import VFLDataset, VFLLogistic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sensitivity = 0.1
delta = 1e-5
epsilon = 0.01
attack = True
defense = False

# -----------------------------
# Defense Implementation 
# -----------------------------
def make_template(K):
    """Construct the distance-preserving template matrix A_T."""
    return (2.0 / K) * np.ones((K, K)) - np.eye(K)
 

def obfuscate_confidence_scores(preds_without_defense):
    c_sorted, sort_idx = torch.sort(preds_without_defense, axis=1)       # sorted values
    A = make_template(preds_without_defense.shape[1])

    # add Gaussian noise
    A_perturb_np = add_Gaussian_noise(A, epsilon, delta, sensitivity)

    A_perturb  = torch.tensor(A_perturb_np, dtype=torch.float32).to(device)
    c_sorted = c_sorted.to(device)
    

    U_sorted = - c_sorted @ A_perturb.T          # [n, C]
    U        = torch.zeros_like(U_sorted)
    U.scatter_(1, sort_idx, U_sorted)
    return U

# -----------------------------
# ESA attack implementation
# -----------------------------

@torch.no_grad()
def esa_attack(model, x_act, yhat, num_active, reg=1e-4, eps=1e-3):
    """
    ESA without any renormalization of yhat:
      - we only clamp yhat >= eps to prevent log(0)
      - we do NOT divide by sum(yhat)
    """
    # 1) build the difference matrices
    W_act  = model.linear_act.weight.data  # [C, A]
    W_pas  = model.linear_pas.weight.data  # [C, P]
    Phi_act = W_act[1:] - W_act[:-1]       # [C-1, A]
    Phi_pas = W_pas[1:] - W_pas[:-1]       # [C-1, P]

    # 2) clamp away zeros, but DO NOT renormalize
    # print("yhat: ", yhat)
    y_safe = yhat.clamp(min=eps)           # all entries < eps → eps
    # print("y_safe: ", y_safe)
    
    # 3) form log‐ratios psi = ln(c_{m+1})−ln(c_m)
    ln_c = torch.log(y_safe.squeeze(0))    # [C]
    psi  = ln_c[1:] - ln_c[:-1]  
    # print("psi: ", psi)
    # 4) subtract active contribution
    a_val = Phi_act @ x_act.squeeze(0)     # [C-1]
    b     = psi - a_val                    # [C-1]

    # 5) regularized least‐squares solve: (Phi_pas^T Phi_pas + reg·I)⁻¹ Phi_pas^T b
    A    = Phi_pas                         # [C-1, P]
    AtA  = A.T @ A                         # [P, P]
    P    = AtA.size(0)
    I    = torch.eye(P, device=AtA.device)
    inv  = torch.inverse(AtA + reg * I)    # [P, P]
    x_hat = inv @ (A.T @ b)                # [P]
    # print("x_hat: ", x_hat[:5])
    # print("Reconstructed x_pas[0]:", x_hat[0].item())
    return x_hat

# -----------------------------
# Training and evaluation
# -----------------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for x_act, x_pas, y in loader:
        x_act, x_pas, y = x_act.to(device), x_pas.to(device), y.to(device)
        optimizer.zero_grad()
        yhat, logits = model(x_act, x_pas)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)

def eval_model(model, loader, device, attack=attack, defense=defense):
    model.eval()

    total          = 0
    correct_no_def = 0
    correct_def    = 0
    mse_sum        = 0.0

    for x_act, x_pas, y in loader:
        x_act, x_pas, y = x_act.to(device), x_pas.to(device), y.to(device)
        batch_size = y.size(0)

        with torch.no_grad():
            # raw confidences + logits
            yhat, logits = model(x_act, x_pas)

        # optionally defend the *confidences* (not logits!)
        if defense:
            # should return a [batch_size, C] tensor of probs
            # print("yhat: ", yhat)
            defended_yhat = obfuscate_confidence_scores(yhat)
            cur_yhat = defended_yhat
            # cur_yhat = defended_yhat / defended_yhat.sum(dim=1, keepdim=True)
        else:
            cur_yhat = yhat

        # --- accuracy without defense (always from raw logits) ---
        preds_no_def = logits.argmax(dim=1)
        correct_no_def += (preds_no_def == y).sum().item()

        # --- accuracy with defense (if toggled) ---
        if defense:
            # defended_yhat → class preds
            preds_def = cur_yhat.argmax(dim=1)
            correct_def += (preds_def == y).sum().item()

        diff = (cur_yhat - yhat).cpu().numpy()
        # print(f"DP ε = {epsilon:.4g} →  "
        # f"│Δc│ max={diff.max():.4g}, mean={diff.mean():.4g}, std={diff.std():.4g}")

        # 3) Clamp both so we never take log(0)
        y_safe_true  = yhat .clamp(min=1e-9)
        y_safe_noisy = cur_yhat.clamp(min=1e-9)

        # 4) Compute log‐ratio vectors
        ln_true   = torch.log(y_safe_true .squeeze(0))  # [C]
        ln_noisy  = torch.log(y_safe_noisy.squeeze(0))  # [C]
        psi_true  = ln_true[1:]  - ln_true[:-1]         # [C-1]
        psi_noisy = ln_noisy[1:] - ln_noisy[:-1]        # [C-1]

        # 5) Measure the maximum absolute perturbation
        max_delta = (psi_noisy - psi_true).abs().max().item()
        # print(f"max |ψ_noisy - ψ_true| = {max_delta:.6f}")


        # --- ESA attack MSE on *cur_yhat* ---
        if attack:
            for i in range(batch_size):
                # use defended yhat if defense=True, else raw yhat
                recon = esa_attack(
                    model,
                    x_act[i:i+1],
                    cur_yhat[i:i+1],
                    loader.dataset.num_active
                )
                mse_sum += ((recon - x_pas[i])**2).mean().item()

        total += batch_size

    # final metrics
    acc_no_def = correct_no_def / total
    acc_def    = (correct_def / total) if defense else None
    mse        = (mse_sum / total)       if attack  else None

    # reporting
    print(f"Accuracy w/o defense: {acc_no_def*100:.2f}%")
    if defense:
        print(f"Accuracy w/  defense: {acc_def*100:.2f}%, drop = {(acc_no_def-acc_def)*100:.2f}%")
    if attack:
        pass
        # print(f"ESA MSE ({'defended' if defense else 'raw'}): {mse:.6f}")

    # return raw‐accuracy, defended‐accuracy (or None), and the mse
    return acc_no_def, mse

# -----------------------------
# Main script
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='VFL Logistic Regression with ESA Attack')
    parser.add_argument('--percent_active', type=float, default=50.0,
                        help='Percent of features for active party (e.g. 20 for 20%%)')
    parser.add_argument('--attack', action='store_true', default=True,
                        help='Enable ESA attack evaluation')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Test set fraction')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load and normalize data
    # df = pd.read_csv(args.csv)
    df = pd.read_csv('/home/msindhuja/PFI-VFL/datasets/DRIVE-DIAG/drive_cleaned.csv')
    X  = df.iloc[:, :-1].values
    y  = df.iloc[:,  -1].astype(int).values
    X  = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-8)

    # split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_ratio,
        random_state=args.seed, stratify=y)

    # create datasets & loaders
    train_ds = VFLDataset(X_train, y_train, args.percent_active)
    test_ds  = VFLDataset(X_test,  y_test,  args.percent_active)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # get the actual split size
    num_active = train_ds.num_active
    num_pas    = X.shape[1] - num_active

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = VFLLogistic(
        in_act=num_active,
        in_pas =num_pas,
        num_classes=len(np.unique(y))
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch}/{args.epochs}, Loss: {loss:.4f}")

    # evaluation
    acc, mse = eval_model(model, test_loader, device, attack=attack)
    if attack:
        print(f"ESA Reconstruction MSE: {mse:.6f}", "DP epsilon: ", epsilon)