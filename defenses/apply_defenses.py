import torch
from rounding import round_confidences
from noising import add_Gaussian_noise_raw
from fh_ope import FH_OPE, encrypt_confidence_batch
from torch_generate_noise1 import add_Gaussian_noise_privee
from torch_generate_noise2 import add_Gaussian_noise_priveeplus

#import decimals, epsilon, delta, sensitivity from config
ope = FH_OPE() 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def apply_defense_before(defense, confidence_scores, labels, decimals, epsilon,delta, sensitivity):
    epoch_correct_defense = 0
    if defense == 'rounding':
        rounded_scores  = round_confidences(confidence_scores.detach(), decimals)
        pred_defense    = torch.argmax(rounded_scores, dim=1)
        epoch_correct_defense += (pred_defense == labels).sum().item()

    if defense == 'noising':
        noised_scores = add_Gaussian_noise_raw(
            confidence_scores.detach(), epsilon, delta, sensitivity).float()       
        pred_defense = torch.argmax(noised_scores, dim=1)
        epoch_correct_defense += (pred_defense == labels).sum().item()

    if defense == 'fh-ope':
        enc_scores = encrypt_confidence_batch(confidence_scores.detach(), ope, device=device)
        pred_defense = torch.argmax(enc_scores, dim=1)
        epoch_correct_defense += (pred_defense == labels).sum().item()

    if defense == 'privee':
        enc_scores = add_Gaussian_noise_privee(confidence_scores.detach(), epsilon, delta, sensitivity)
        pred_defense = torch.argmax(enc_scores, dim=1)
        epoch_correct_defense += (pred_defense == labels).sum().item()

    if defense == 'priveeplus':
        enc_scores = add_Gaussian_noise_priveeplus(confidence_scores.detach(), epsilon, delta, sensitivity)
        pred_defense = torch.argmax(enc_scores, dim=1)
        epoch_correct_defense += (pred_defense == labels).sum().item()
    
    return epoch_correct_defense

def apply_defense_after(defense, conf_scores_tensor, num_classes, decimals, epsilon,delta, sensitivity):
    if defense == 'rounding':
        defended_conf = round_confidences(conf_scores_tensor.detach(), decimals)
    elif defense == 'noising':
        defended_conf = add_Gaussian_noise_raw(
        conf_scores_tensor.detach(), epsilon, delta, sensitivity
    ).float().to(device)   

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
        enc_conf = encrypt_confidence_batch(conf_scores_tensor.detach(), ope, device='cpu')
        ranks = torch.argsort(enc_conf, dim=1, descending=True)
        pseudo_conf = torch.zeros_like(enc_conf, dtype=torch.float)
        for i in range(enc_conf.shape[0]):
            for j, cls in enumerate(ranks[i]):
                pseudo_conf[i][cls] = float(num_classes - j) / num_classes
        defended_conf = pseudo_conf.to(device)
    return defended_conf


def apply_defense_after_cifar(defense, stored_batch_idxs, epsilon, delta, sensitivity, x_act_all_cpu, conf_scores_tensor, decimals, num_classes):
    if defense == 'priveeplus':
        defended_scores = add_Gaussian_noise_priveeplus(conf_batch_gpu, epsilon, delta, sensitivity)
        for batch_idxs in stored_batch_idxs:
            idxs = torch.tensor(batch_idxs, dtype=torch.long)

            x_act_batch_cpu = x_act_all_cpu[idxs]         # (B, 3, 32, 16) on CPU
            conf_batch_gpu   = conf_scores_tensor[idxs]     # (B, num_classes) on GPU

            # Run GIA on this batch, using attack_batch_size=1
            conf_def_batch = defended_scores  # (B, num_classes)
            
    if defense == 'rounding':
               
        defended_scores = round_confidences(conf_scores_tensor, decimals)
        conf_def_batch = defended_scores
                
    if defense == 'noising':
        defended_scores = add_Gaussian_noise_raw(
            conf_scores_tensor,
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity
        ).float().to(device)
        conf_def_batch = defended_scores

    if defense == 'fh-ope':
        defended_scores = encrypt_confidence_batch(conf_scores_tensor.detach(), ope, device='cpu')
        ranks = torch.argsort(defended_scores, dim=1, descending=True)
        pseudo_conf = torch.zeros_like(defended_scores, dtype=torch.float)
        for i_row in range(defended_scores.shape[0]):
            for rank_idx, cls in enumerate(ranks[i_row]):
                pseudo_conf[i_row][cls] = float(num_classes - rank_idx) / num_classes
        defended_conf = pseudo_conf.to(device)
        conf_def_batch = defended_conf

    if defense == 'privee':
        defended_scores_full = add_Gaussian_noise_privee(
            conf_scores_tensor,
            epsilon, delta, sensitivity
        )


        for batch_idxs in stored_batch_idxs:
            idxs = torch.tensor(batch_idxs, dtype=torch.long)

            x_act_batch_cpu = x_act_all_cpu[idxs]         # (B, 3, 32, 16) on CPU
            conf_batch_gpu   = conf_scores_tensor[idxs]     # (B, num_classes) on GPU

            # Run GIA on this batch, using attack_batch_size=1
            conf_def_batch = defended_scores_full[idxs]  # (B, num_classes)
            

    return conf_def_batch, x_act_batch_cpu
