import torch
import argparse
from utils import batch_split
from models import APModel, PPModel, LearningCoordinator, APModelADULT, PPModelADULT, LearningCoordinatorADULT, LearningCoordinatorCIFAR,BottomModel
from train import trainVFL, VFLTrainCIFAR
from split_data import vfl_split_data
import yaml
import numpy as np
import random

# RESOLVE ADULT DATASET PROBLEM
# DO GRNA, GIA FOR CIFAR10, CIFAR100
# ESA 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--attack", type=str, choices=["grna", "gia", "esa"], default='grna')
parser.add_argument("--defense", type=str, choices=["rounding", "noising", "fh-ope", "privee", "priveeplus"],required=True)
parser.add_argument("--decimals",type=int, default=1, help="Number of decimal places to round to (only for rounding).")
parser.add_argument("--rho", type=float, default=0.1,help="Two eps values for noising (only for defense=noising).")
parser.add_argument("--attack_strength", type=float, default=0.5, help="nothing")
parser.add_argument("--dataset", type=str, default='DRIVE', help="nothing")
parser.add_argument("--seed", type=int, default='1', help="seed")
parser.add_argument("--epsilon", type=float, default=0.5,help="Two eps values for noising (only for defense=noising).")

args = parser.parse_args()

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
dname = args.dataset
dataset_config = config.get(dname)

organization_num = config['organization_num']
delta = float(config['delta'])
sensitivity = config['sensitivity']

batch_size = dataset_config['batch_size']
epochs = dataset_config['epochs']
num_classes = dataset_config['num_classes']
epochs_grna = dataset_config['epochs_grna']
lr_vfl= float(dataset_config['lr_vfl'])
lr_grna = float(dataset_config['lr_grna'])

attack = args.attack
defense = args.defense
decimals = args.decimals
attack_str = args.attack_strength
rho = args.rho
dname = args.dataset
epsilon = args.epsilon

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


set_seed(args.seed)
print(f"Seed: {args.seed}")

if dname == 'CIFAR10' or dname == 'CIFAR100':
    X_train_vertical_FL, train_loader_list, y_train = vfl_split_data(dname, attack_str, organization_num, batch_size)
else:
    X_train_vertical_FL, train_loader_list = vfl_split_data(dname, attack_str, organization_num, batch_size)
     

x_ap_all = X_train_vertical_FL[0].to(device) 
N = x_ap_all.shape[0]
batch_idxs_list = batch_split(N, batch_size, 'mini-batch')

if dname == 'ADULT':
    ap_model = APModelADULT(in_features=len(X_train_vertical_FL[0][0]), hidden_layer=128, out_features=64)
    pp_model = PPModelADULT(in_features=len(X_train_vertical_FL[1][0]), hidden_layer=128, out_features=64)
    coordinator = LearningCoordinatorADULT(64, 64, num_classes)


if dname == 'MNIST' or dname == 'DRIVE':
    ap_model = APModel(in_features=len(X_train_vertical_FL[0][0]), hidden_layer=128, out_features=64)
    pp_model = PPModel(in_features=len(X_train_vertical_FL[1][0]), hidden_layer=128, out_features=64)
    coordinator = LearningCoordinator(64, 64, num_classes)


    ap_model = ap_model.to(device)
    pp_model = pp_model.to(device)
    coordinator = coordinator.to(device)


    # accuracy_diff, mse_no_defense, mse_with_defense \
    #         = trainVFL(ap_model, pp_model, coordinator, dname, 
    #                 X_train_vertical_FL,train_loader_list, 
    #                     batch_idxs_list,
    #                     num_classes, N, x_ap_all, epochs, 
    #                     defense, lr_vfl, epochs_grna, attack,
    #                     decimals, epsilon,delta, sensitivity
    #                     )

    accuracy_diff, mse_no_defense, mse_with_defense = trainVFL(
    ap_model, pp_model, coordinator, dname,
    X_train_vertical_FL, train_loader_list,
    batch_idxs_list,
    num_classes, N, x_ap_all, epochs,
    defense, lr_vfl, lr_grna, epochs_grna, attack,attack_str,
    decimals, rho
)



if dname == 'CIFAR10' or dname == 'CIFAR100':
    organization_output_dim = np.array([64 for i in range(organization_num)])
    top_output_dim = 10

    if dname == 'CIFAR10':
         top_output_dim = 10
         num_classes = 10

    if dname == 'CIFAR100':
             top_output_dim = 100
             num_classes = 100
        

    organization_models = {}            
    for organization_idx in range(organization_num):
                organization_models[organization_idx] = \
                    BottomModel(out_dim=organization_output_dim[organization_idx]).to(device)

    top_model = LearningCoordinatorCIFAR(np.sum(organization_output_dim),num_classes=num_classes)
    top_model = top_model.to(device).float()

    acc_diff, mse_no_def, mse_with_def \
        = VFLTrainCIFAR(top_model,organization_models,organization_num,N, dname,
                        num_classes,attack,epochs,X_train_vertical_FL, batch_size, 
                        y_train, lr_vfl, lr_grna, defense, epochs_grna, attack_str, decimals, rho, epsilon, delta, sensitivity)



