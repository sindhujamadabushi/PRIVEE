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
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def trainVFL(ap_model, pp_model, coordinator, dname, X_train_vertical_FL, train_loader_list, batch_idxs_list,
    num_classes, N, x_ap_all, epochs, defense, lr_vfl, lr_grna, epochs_grna, attack, attack_strength, decimals, rho, epsilon, delta, sensitivity):    
    ap_model = ap_model.to(device)
    pp_model = pp_model.to(device)
    coordinator = coordinator.to(device)

    optimizer = optim.Adam(list(ap_model.parameters()) + list(pp_model.parameters()) + list(coordinator.parameters()),lr=lr_vfl)
    criterion = nn.CrossEntropyLoss()

        # ---------------------------------------------------------
    # Checkpoint path: one model per dataset and attack strength
    # ---------------------------------------------------------
    os.makedirs("checkpoints", exist_ok=True)

    checkpoint_path = os.path.join(
        "checkpoints",
        f"{dname}_{attack_strength}.pt"
    )

    # ---------------------------------------------------------
    # Load checkpoint if it already exists
    # ---------------------------------------------------------
    if os.path.exists(checkpoint_path):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        ap_model.load_state_dict(
            checkpoint["ap_model_state_dict"]
        )
        pp_model.load_state_dict(
            checkpoint["pp_model_state_dict"]
        )
        coordinator.load_state_dict(
            checkpoint["coordinator_state_dict"]
        )

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        print(f"Loaded model checkpoint from: {checkpoint_path}")
        print("Skipping VFL model training.")

    # ---------------------------------------------------------
    # Train only if checkpoint does not exist
    # ---------------------------------------------------------
    else:

        print(
            f"No checkpoint found for dataset={dname}, "
            f"attack_strength={attack_strength}"
        )
        print("Training VFL model.")

        for epoch in range(epochs):

            ap_model.train()
            pp_model.train()
            coordinator.train()

            total_loss = 0
            epoch_correct = 0
            epoch_samples = 0

            for batch_num, (x_ap, x_pp, labels) in enumerate(
                zip(
                    train_loader_list[0],
                    train_loader_list[1],
                    train_loader_list[2]
                )
            ):

                idxs = torch.tensor(
                    batch_idxs_list[batch_num],
                    dtype=torch.long,
                    device=device
                )

                x_ap = x_ap.to(device)
                x_pp = x_pp.to(device)
                labels = labels.to(device)

                apout = ap_model(x_ap)
                ppout = pp_model(x_pp)
                confid = coordinator(apout, ppout)

                loss = criterion(confid, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pred_raw = torch.argmax(confid, dim=1)

                epoch_correct += (
                    pred_raw == labels
                ).sum().item()

                epoch_samples += labels.size(0)
                total_loss += loss.item()

            print(
                f"Epoch {epoch + 1}/{epochs}, "
                f"loss: {total_loss:.4f}, "
                f"accuracy: "
                f"{100.0 * epoch_correct / epoch_samples:.2f}%"
            )

        # Save the trained model
        torch.save(
            {
                "dataset": dname,
                "attack_strength": attack_strength,
                "ap_model_state_dict": ap_model.state_dict(),
                "pp_model_state_dict": pp_model.state_dict(),
                "coordinator_state_dict": coordinator.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path
        )

        print(f"Model checkpoint saved to: {checkpoint_path}")

    # ---------------------------------------------------------
    # Fresh forward pass after either loading or training
    # ---------------------------------------------------------
    ap_model.eval()
    pp_model.eval()
    coordinator.eval()

    conf_scores_tensor = torch.zeros(
        N,
        num_classes,
        device=device
    )

    all_labels = torch.zeros(
        N,
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        for batch_num, (x_ap, x_pp, labels) in enumerate(
            zip(
                train_loader_list[0],
                train_loader_list[1],
                train_loader_list[2]
            )
        ):

            idxs = torch.tensor(
                batch_idxs_list[batch_num],
                dtype=torch.long,
                device=device
            )

            x_ap = x_ap.to(device)
            x_pp = x_pp.to(device)
            labels = labels.to(device)

            apout = ap_model(x_ap)
            ppout = pp_model(x_pp)
            confid = coordinator(apout, ppout)

            conf_scores_tensor[idxs] = confid
            all_labels[idxs] = labels

    # ---------------------------------------------------------
    # Apply the selected defense to the fresh confidence scores
    # ---------------------------------------------------------
    defended_conf = apply_defense_after(
        defense,
        conf_scores_tensor,
        num_classes,
        decimals,
        rho, epsilon, delta, sensitivity
    )

    accuracy_raw = 100.0 * (
        conf_scores_tensor.argmax(dim=1) == all_labels
    ).sum().item() / N

    accuracy_defense = 100.0 * (
        defended_conf.argmax(dim=1) == all_labels
    ).sum().item() / N

    accuracy_diff = accuracy_raw - accuracy_defense

    print(f"Final raw accuracy:     {accuracy_raw:.2f}%")
    print(f"Final defense accuracy: {accuracy_defense:.2f}%")
    print(f"Difference in accuracy: {accuracy_diff:.2f}%")

    
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


def VFLTrainCIFAR(top_model, organization_models, organization_num, N,dname,
                  num_classes, attack, epochs, X_train_vertical_FL,
                  batch_size, y_train, lr_vfl, lr_grna, defense,
                  epochs_grna, attack_strength, decimals, rho, epsilon, delta, sensitivity):

    device = next(top_model.parameters()).device

    # X_train_vertical_FL = [x.to(device) for x in X_train_vertical_FL]
    for i in range(organization_num):
        X_train_vertical_FL[i] = X_train_vertical_FL[i].to(device)

    y_train = y_train.to(device)

    batches = [
        torch.as_tensor(batch, dtype=torch.long, device=device)
        for batch in batch_split(
            len(X_train_vertical_FL[0]),
            batch_size,
            "mini-batch"
        )
    ]


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(top_model.parameters(), lr_vfl, momentum=0.9, weight_decay=5e-4)

    optim_org = [torch.optim.SGD(m.parameters(), lr_vfl, momentum=0.9, weight_decay=1e-3)
                 for m in organization_models.values()]
    sched_org = [torch.optim.lr_scheduler.MultiStepLR(opt, [15,30,40,60,80], gamma=0.925)
                 for opt in optim_org]

        # ---------------------------------------------------------
    # Checkpoint path
    # num_classes=10  -> CIFAR10
    # num_classes=100 -> CIFAR100
    # ---------------------------------------------------------
    dataset_name = dname

    os.makedirs("checkpoints", exist_ok=True)

    checkpoint_path = os.path.join(
        "checkpoints",
        f"{dataset_name}_{attack_strength}.pt"
    )

    # ---------------------------------------------------------
    # Load existing checkpoint
    # ---------------------------------------------------------
    if os.path.exists(checkpoint_path):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        top_model.load_state_dict(
            checkpoint["top_model_state_dict"]
        )

        for i, model in organization_models.items():
            model.load_state_dict(
                checkpoint[
                    "organization_model_state_dicts"
                ][i]
            )

        if "top_optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["top_optimizer_state_dict"]
            )

        if "organization_optimizer_state_dicts" in checkpoint:
            for opt, saved_state in zip(
                optim_org,
                checkpoint[
                    "organization_optimizer_state_dicts"
                ]
            ):
                opt.load_state_dict(saved_state)

        print(f"Loaded model checkpoint from: {checkpoint_path}")
        print("Skipping CIFAR VFL model training.")

    # ---------------------------------------------------------
    # Train only when checkpoint is missing
    # ---------------------------------------------------------
    else:

        print(
            f"No checkpoint found for dataset={dataset_name}, "
            f"attack_strength={attack_strength}"
        )
        print("Training CIFAR VFL model.")

        for ep in range(epochs):

            top_model.train()

            for model in organization_models.values():
                model.train()

            total_correct = torch.zeros(
                (),
                device=device
            )
            tot = 0
            total_loss = 0.0

            for idxs in batches:

                optimizer.zero_grad(set_to_none=True)

                for opt in optim_org:
                    opt.zero_grad(set_to_none=True)

                outs = [
                    organization_models[i](
                        X_train_vertical_FL[i][idxs]
                    )
                    for i in range(organization_num)
                ]

                logits = top_model(
                    torch.cat(outs, dim=1)
                )

                labels = y_train[idxs]

                loss = criterion(logits, labels)
                loss.backward()

                optimizer.step()

                for opt in optim_org:
                    opt.step()

                total_correct += (
                    logits.argmax(dim=1) == labels
                ).sum()

                tot += labels.size(0)
                total_loss += loss.item()

            for scheduler in sched_org:
                scheduler.step()

            print(
                f"Epoch {ep + 1}/{epochs}, "
                f"loss: {total_loss:.4f}, "
                f"accuracy: "
                f"{100.0 * total_correct.item() / tot:.2f}%"
            )

        # Save trained CIFAR model
        torch.save(
            {
                "dataset": dataset_name,
                "attack_strength": attack_strength,
                "top_model_state_dict": top_model.state_dict(),
                "organization_model_state_dicts": {
                    i: model.state_dict()
                    for i, model in organization_models.items()
                },
                "top_optimizer_state_dict": optimizer.state_dict(),
                "organization_optimizer_state_dicts": [
                    opt.state_dict()
                    for opt in optim_org
                ],
            },
            checkpoint_path
        )

        print(f"Model checkpoint saved to: {checkpoint_path}")

    # ---------------------------------------------------------
    # Fresh forward pass after either loading or training
    # ---------------------------------------------------------
    top_model.eval()

    for model in organization_models.values():
        model.eval()

    conf_scores_tensor = torch.zeros(
        N,
        num_classes,
        device=device
    )

    total_correct = 0
    tot = 0

    with torch.no_grad():

        for idxs in batches:

            outs = [
                organization_models[i](
                    X_train_vertical_FL[i][idxs]
                )
                for i in range(organization_num)
            ]

            logits = top_model(
                torch.cat(outs, dim=1)
            )

            labels = y_train[idxs]

            probs = torch.softmax(
                logits,
                dim=1
            )

            conf_scores_tensor[idxs] = probs

            total_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()

            tot += labels.size(0)

    acc_raw = 100.0 * total_correct / tot

    # Apply the requested defense
    defended_conf = apply_defense_after(
        defense,
        conf_scores_tensor,
        num_classes,
        decimals,
        rho, epsilon, delta, sensitivity)

    acc_def = 100.0 * (
        defended_conf.argmax(dim=1) == y_train
    ).sum().item() / N

    print(
        f"Final raw accuracy: {acc_raw:.2f}%, "
        f"with defense accuracy: {acc_def:.2f}%, "
        f"difference in accuracy: {acc_raw - acc_def:.2f}%"
    )

    
    if attack=="gia":
        idxs = torch.tensor(batches[0])
        x_act = X_train_vertical_FL[0][idxs].cpu()
        conf  = conf_scores_tensor[idxs]
        mse_no_def = gradient_inversion_attack_cifar(organization_models,
                                               top_model, x_act, conf,
                                               lr=1e-3, iters=10)
        conf_def = apply_defense_after(defense, conf_scores_tensor,
                                       num_classes, decimals, rho, epsilon, delta, sensitivity)[idxs]
        mse_def  = gradient_inversion_attack_cifar(organization_models,
                                             top_model, x_act, conf_def,
                                             lr=1e-3, iters=10)


    elif attack == "grna":
        # defended_conf = apply_defense_after_cifar(
        #     defense,
        #     conf_scores_tensor,
        #     num_classes,
        #     decimals,
        #     rho
        # )

        defended_conf = apply_defense_after(
            defense,
            conf_scores_tensor,
            num_classes,
            decimals,
            rho, epsilon, delta, sensitivity
        )

        mse_no_def = grna_cifar(
            X_train_vertical_FL,
            X_train_vertical_FL[0],
            conf_scores_tensor,
            organization_models[0],
            organization_models[1],
            top_model,
            epochs_grna,
            batches,
            lr=lr_grna
        )

        print(f"GRNA MSE without defense: {mse_no_def:.6f}")

        mse_def = grna_cifar(
            X_train_vertical_FL,
            X_train_vertical_FL[0],
            defended_conf,
            organization_models[0],
            organization_models[1],
            top_model,
            epochs_grna,
            batches,
            lr=lr_grna
        )

    print("MSE without defense: ", mse_no_def)
    print("MSE with defense: ", mse_def)
    
    return acc_raw-acc_def, mse_no_def, mse_def
    # print("MSE without defense: ", mse_no_def.item())
    # print("MSE with defense: ", mse_def.item())
    
    # return acc_raw-acc_def, mse_no_def.item(), mse_def.item()
 


