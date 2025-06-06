import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.amp import autocast
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gradient_inversion_attack(org_models, top_model,
                              x_act,      
                              true_conf,  
                              lr=1e-3, iters=500):

    x_pas_hat = torch.zeros_like(x_act, device=device, requires_grad=True)

    x_act = x_act.to(device)               

    optimizer = torch.optim.Adam([x_pas_hat], lr=lr)

    for it in range(iters):
        optimizer.zero_grad()

        h_act = org_models[0](x_act)       
        h_pas = org_models[1](x_pas_hat)

        logits = top_model(h_act, h_pas)    
        pred_conf = F.softmax(logits, dim=1)

        loss = F.mse_loss(pred_conf, true_conf)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x_pas_hat.clamp_(0.0, 1.0)

        if it % 100 == 0:
            print(f"GIA[Epoch {it}/{iters}] MSE: {loss.item():.6f}")

    return loss


def grna(x_ap_all, conf_scores_tensor,
         ap_model, pp_model, coordinator,
         epochs_grna, batch_idxs_list, X_train_vertical_FL, lr=0.1):

    ap_model.eval()  
    pp_model.eval()
    coordinator.eval()
    
    for p in ap_model.parameters():        
        p.requires_grad_(False)
    for p in pp_model.parameters():        
        p.requires_grad_(False)
    for p in coordinator.parameters():     
        p.requires_grad_(False)

    x_pp_init = torch.randn_like(X_train_vertical_FL[1], device=device, requires_grad=True)

    optimizer = torch.optim.Adam([x_pp_init], lr=lr)
    mse = nn.MSELoss()

    for epoch in range(epochs_grna):
        optimizer.zero_grad()

        batch_losses = []                                    
        for batch_idxs in batch_idxs_list:
            idx = torch.tensor(batch_idxs, dtype=torch.long, device=device)

            with torch.no_grad():
                apout = ap_model(x_ap_all[idx])
            ppout = pp_model(torch.index_select(x_pp_init, 0, idx))

            generated_confidence_scores = coordinator(apout, ppout)
            ground_truth_confidence_scores = conf_scores_tensor[idx]

            batch_losses.append(mse(generated_confidence_scores, ground_truth_confidence_scores))              

        epoch_loss = torch.stack(batch_losses).mean()         
        epoch_loss.backward()
        optimizer.step()
        if epoch%10==0:
            print(f"[GRNA] epoch {epoch+1}/{epochs_grna}  MSE {epoch_loss.item():.6f}")

    return epoch_loss

def grna_cifar(X_train_vertical_FL, x_ap_all, conf_scores_tensor,
         ap_model, pp_model, coordinator,
         epochs_grna, batch_idxs_list, lr=0.1):

    ap_model.eval();  pp_model.eval();  coordinator.eval()
    
    for m in (ap_model, pp_model, coordinator):
        for p in m.parameters(): p.requires_grad_(False)

    x_pp_init = torch.randn_like(X_train_vertical_FL[1],
                                 device=device, requires_grad=True)

    optimizer = torch.optim.Adam([x_pp_init], lr=lr)
    mse = nn.MSELoss()

    for epoch in range(epochs_grna):
        epoch_loss = 0.0                                            # float

        for batch_idxs in batch_idxs_list:
            idx_cpu = torch.tensor(batch_idxs, dtype=torch.long)
            idx_gpu = idx_cpu.to(device)

            # ----- forward -----
            with torch.no_grad():
                apout = ap_model(x_ap_all[idx_cpu].to(device))      # (B,64)

            ppout  = pp_model(torch.index_select(x_pp_init, 0, idx_gpu))  # (B,64)
            joint  = torch.cat((apout, ppout), dim=1)               # (B,128)
            logits = coordinator(joint)                             # raw
            gen    = F.softmax(logits, dim=1)                       # (B,10)

            true   = conf_scores_tensor[idx_gpu]                    # (B,10)
            loss   = mse(gen, true)

            # ----- backward for *this* mini-batch only -----
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()                               # scalar
            del apout, ppout, joint, logits, gen, true, loss
            torch.cuda.empty_cache()                                # release

        print(f"[GRNA] epoch {epoch+1}/{epochs_grna}  "
              f"avg-MSE {epoch_loss/len(batch_idxs_list):.6f}")

    return epoch_loss/len(batch_idxs_list)


from contextlib import contextmanager

try:                                            # PyTorch ≥ 1.13
    from torch.amp import autocast as _ac
    def autocast_fp16():                        # usage:  with autocast_fp16():
        return _ac("cuda", dtype=torch.float16)
except ImportError:                             # legacy path
    from torch.cuda.amp import autocast as _ac
    def autocast_fp16():
        return _ac(dtype=torch.float16)

def gradient_inversion_attack_cifar(
    org_models,            # dict or list — active idx 0, passive idx 1
    top_model,
    x_act_cpu,             # (B, 3, 32, 32)  active-party images on *CPU*
    true_conf,             # (B, num_classes)  confidence vector
    lr: float = 1e-3,
    iters: int = 100,
    attack_batch_size: int = 1,   # outer chunk size
    micro: int = 1                # inner micro-batch for passive model
):
    """
    Memory-efficient GIA for CIFAR-like data.

    • freezes all model weights (we only optimise x_pas_hat)
    • uses FP16 autocast to halve activation size
    • passive forward done in 'micro' splits to bound peak memory
    """
    device = next(top_model.parameters()).device
    B      = x_act_cpu.shape[0]
    true_conf = true_conf.to(device)

     if isinstance(org_models, dict):
        model_list = list(org_models.values())
    else:
        model_list = list(org_models)

    for m in model_list + [top_model]:
        m.eval()
        for p in m.parameters():       # type: ignore[arg-type]
            p.requires_grad_(False)

    x_pas_hat = torch.zeros_like(x_act_cpu, device=device, requires_grad=True)
    opt       = torch.optim.Adam([x_pas_hat], lr=lr)
    mse       = nn.MSELoss()

    for it in range(iters):
        opt.zero_grad()

        with autocast_fp16():         
            
            h_act_chunks = []
            for s in range(0, B, attack_batch_size):
                e   = min(s + attack_batch_size, B)
                sub = x_act_cpu[s:e].to(device)
                with torch.no_grad():
                    h_act_chunks.append(org_models[0](sub))
            h_act = torch.cat(h_act_chunks, dim=0)           # (B, d)

            
            h_pas_chunks = []
            for s in range(0, B, attack_batch_size):
                e         = min(s + attack_batch_size, B)
                sub_hat   = x_pas_hat[s:e]                   
                micro_out = []
                for msub in sub_hat.split(micro):
                    micro_out.append(org_models[1](msub))
                h_pas_chunks.append(torch.cat(micro_out, dim=0))
            h_pas = torch.cat(h_pas_chunks, dim=0)           # (B, d)

           
            h_joint = torch.cat((h_act, h_pas), dim=1)
            logits  = top_model(h_joint)
            pred    = torch.softmax(logits, dim=1)

        
        loss = mse(pred, true_conf)
        loss.backward()
        opt.step()

        
        with torch.no_grad():
            x_pas_hat.clamp_(0., 1.)

        if it % 100 == 0 or it == iters - 1:
            print(f"[Iter {it+1:4d}/{iters}]  MSE loss: {loss.item():.6f}")

        torch.cuda.empty_cache()

    return loss
