import torch
import sys
sys.path.append('defenses')
from apply_defenses import apply_defense_before, apply_defense_after, apply_defense_after_cifar
from attacks import gradient_inversion_attack, grna, grna_cifar, gradient_inversion_attack_cifar
from utils import load_dat, batch_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy as np, torch
from torch import nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def trainVFL(ap_model, pp_model, coordinator, X_train_vertical_FL, train_loader_list, batch_idxs_list, num_classes, N, x_ap_all, epochs, defense, lr_vfl, lr_grna,epochs_grna, attack, decimals, epsilon,delta, sensitivity):
    
    ap_model = ap_model.to(device)
    pp_model = pp_model.to(device)
    coordinator = coordinator.to(device)

    optimizer = optim.Adam(list(ap_model.parameters()) + list(pp_model.parameters()) + list(coordinator.parameters()),lr=float(lr_vfl))
    criterion = nn.CrossEntropyLoss()

    conf_scores_tensor = torch.zeros(N, num_classes).to(device)
    
    total_correct          = 0
    total_correct_defense  = 0
    total_samples          = 0 


    for epoch in range(epochs):
        ap_model.train()
        pp_model.train()
        coordinator.train()

        total_loss = 0

        epoch_correct          = 0
        epoch_correct_defense  = 0
        epoch_samples          = 0

        for batch_num, (x_ap, x_pp, labels) in enumerate(
                zip(train_loader_list[0],
                    train_loader_list[1],
                    train_loader_list[2])
            ):
            start = batch_num * 128
            idxs  = torch.arange(start,
                         start + x_ap.size(0),  # works for the short last batch
                         device=device)
            idxs   = torch.tensor(batch_idxs_list[batch_num], dtype=torch.long, device=device)
            x_ap   = x_ap.to(device)
            x_pp   = x_pp.to(device)
            labels = labels.to(device)

            apout  = ap_model(x_ap)
            ppout  = pp_model(x_pp)
            confid = coordinator(apout, ppout)   

            conf_scores_tensor[idxs] = confid.detach()

            loss = criterion(confid, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_raw = torch.argmax(confid, dim=1)
            epoch_correct += (pred_raw == labels).sum().item()

            epoch_correct_defense += apply_defense_before(defense, confid.detach(), labels, decimals, epsilon,delta, sensitivity)
            

            epoch_samples   += labels.size(0)
            total_loss      += loss.item()

        total_correct         = epoch_correct
        total_correct_defense = epoch_correct_defense
        total_samples         = epoch_samples

        print("epoch done with accuracy:", 100.0 * total_correct / total_samples)

    accuracy_raw = 100.0 * total_correct / total_samples
    accuracy_defense = 100.0 * total_correct_defense / total_samples
    print(f"Final raw accuracy:      {accuracy_raw:.2f}%")
    print(f"Final defense accuracy:  {accuracy_defense:.2f}%")
    accuracy_diff = accuracy_raw-accuracy_defense
    print("difference in accuracy=",accuracy_diff)
    print(f"Final raw accuracy:      {accuracy_raw:.2f}%")
    
    defended_conf = apply_defense_after(defense, conf_scores_tensor, num_classes, decimals, epsilon,delta, sensitivity)

    if attack == 'gia':
        mse_no_defense = gradient_inversion_attack(
                [ap_model, pp_model],
                coordinator,
                x_ap_all.to(device),                  
                conf_scores_tensor.to(device),        
                lr=1e-3, iters=500)
        print(f"GIA MSE without defense: {mse_no_defense.item():.6f}")

        mse_with_defense = gradient_inversion_attack(
            [ap_model, pp_model],
            coordinator,
            x_ap_all.to(device),                  
            defended_conf,        
            lr=1e-3, iters=500)
        print(f"GIA MSE with defense:   {mse_with_defense.item():.6f}")
    
    if attack == 'grna':
        
        mse_no_defense = grna(
            x_ap_all,
            conf_scores_tensor,       
            ap_model, pp_model, coordinator,
            epochs_grna,
            batch_idxs_list,
            X_train_vertical_FL
        )
        print(f"GRNA MSE without defense: {mse_no_defense.item():.6f}")

        mse_with_defense = grna(
        x_ap_all,
        defended_conf,         
        ap_model, pp_model, coordinator,
        epochs_grna,
        batch_idxs_list,
        X_train_vertical_FL
        )
        print(f"GRNA MSE with defense:   {mse_with_defense.item():.6f}")
    
    return accuracy_diff, mse_no_defense.item(), mse_with_defense.item()


def VFLTrainCIFAR(top_model,organization_models,organization_num,N,num_classes,attack,epochs,X_train_vertical_FL,batch_size,y_train,lr_vfl,
                  defense,epochs_grna,decimals,epsilon,delta,sensitivity):

    device = next(top_model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(top_model.parameters(), lr_vfl, momentum=0.9, weight_decay=5e-4)

    optim_org = [torch.optim.SGD(m.parameters(), lr_vfl, momentum=0.9, weight_decay=1e-3)
                 for m in organization_models.values()]
    sched_org = [torch.optim.lr_scheduler.MultiStepLR(opt, [15,30,40,60,80], gamma=0.925)
                 for opt in optim_org]

    conf_scores_tensor = torch.zeros(N, num_classes, device=device)

    for ep in range(epochs):
        batches = batch_split(len(X_train_vertical_FL[0]), batch_size, "mini-batch")
        total_correct = total_def = tot = 0
        for idxs in batches:
            idxs = torch.tensor(idxs, dtype=torch.long)
            optimizer.zero_grad()
            for opt in optim_org: opt.zero_grad()

            outs = [organization_models[i](X_train_vertical_FL[i][idxs].to(device))
                    for i in range(organization_num)]
            logits = top_model(torch.cat(outs,1))
            labels = y_train[idxs].to(device)
            loss = criterion(logits, labels)
            loss.backward(); optimizer.step()
            for opt in optim_org: opt.step()

            probs = torch.softmax(logits.detach(),1)
            conf_scores_tensor[idxs] = probs
            pred = logits.argmax(1)
            total_correct += (pred==labels).sum().item()
            total_def    += apply_defense_before(defense, logits, labels,
                                                 decimals, epsilon, delta, sensitivity)
            tot += labels.size(0)

        for s in sched_org: s.step()
        print(f"Epoch {ep+1}/{epochs} loss {loss.item():.4f} accuracy {100*total_correct/tot:.2f}%")

    acc_raw = 100*total_correct/tot
    acc_def = 100*total_def   /tot
    print(f"Final raw accuracy: {acc_raw:.2f}%, with defense accuracy:{acc_def:.2f}%, difference in accuracy: {acc_raw-acc_def:.2f}%")

    if attack=="gia":
        idxs = torch.tensor(batches[0])
        x_act = X_train_vertical_FL[0][idxs].cpu()
        conf  = conf_scores_tensor[idxs]
        mse_no_def = gradient_inversion_attack_cifar(organization_models,
                                               top_model, x_act, conf,
                                               lr=1e-3, iters=10)
        conf_def = apply_defense_after(defense, conf_scores_tensor,
                                       num_classes, decimals, epsilon, delta, sensitivity)[idxs]
        mse_def  = gradient_inversion_attack_cifar(organization_models,
                                             top_model, x_act, conf_def,
                                             lr=1e-3, iters=10)
    print("MSE without defense: ", mse_no_def.item())
    print("MSE with defense: ", mse_def.item())
    
    return acc_raw-acc_def, mse_no_def.item(), mse_def.item()
 


