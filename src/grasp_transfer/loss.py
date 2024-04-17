''' Adapted from DIF-NET repository https://github.com/microsoft/DIF-Net
'''
import torch
import torch.nn.functional as F

def implicit_loss(model_output, gt,sdf_loss=3e3):

    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    pred_sdf = model_output['model_out']

    embeddings = model_output['latent_vec']

    gradient_sdf = model_output['grad_sdf']

    sdf_constraint = torch.clamp(pred_sdf,-0.5,0.5)-torch.clamp(gt_sdf,-0.5,0.5)

    normal_constraint = torch.where(gt_sdf == 0, 1 - F.cosine_similarity(gradient_sdf, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient_sdf[..., :1]))
    embeddings_constraint = torch.mean(embeddings ** 2)

    # -----------------
    return {'sdf': torch.abs(sdf_constraint).mean() * sdf_loss,  
            'normal_constraint': normal_constraint.mean() * 3e1,
            'embeddings_constraint': embeddings_constraint.mean() * 1e6}
            
def implicit_loss_no_normal(model_output, gt,sdf_loss=3e3):

    gt_sdf = gt['sdf']
    pred_sdf = model_output['model_out']
    sdf_constraint = torch.clamp(pred_sdf,-0.5,0.5)-torch.clamp(gt_sdf,-0.5,0.5)

    embeddings = model_output['latent_vec']
    embeddings_constraint = torch.mean(embeddings ** 2)

    # -----------------
    return {'sdf': torch.abs(sdf_constraint).mean() * sdf_loss, 
            'embeddings_constraint': embeddings_constraint.mean() * 1e4}
    
def transfer_loss(model_output, gt,sdf_loss=3e3):

    gt_sdf = gt['sdf']
    pred_sdf = model_output['model_out']
    sdf_constraint = torch.clamp(pred_sdf,-0.5,0.5)-torch.clamp(gt_sdf,-0.5,0.5)
    
    embeddings = model_output['latent_vec']
    embeddings_constraint = torch.mean(embeddings ** 2)

    # -----------------
    return {'sdf': torch.abs(sdf_constraint).mean() * sdf_loss,  
            'embeddings_constraint': embeddings_constraint.mean() * 1e5}

def transfer_loss_pcd(model_output, gt):

    gt_sdf = gt['sdf']
    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    embeddings = model_output['latent_vec']

    sdf_constraint = torch.clamp(torch.abs(pred_sdf),-0.5,0.5)-torch.clamp(torch.abs(gt_sdf),-0.5,0.5)
    embeddings_constraint = torch.mean(embeddings ** 2)
    
    
    # -----------------
    return {'sdf': torch.abs(sdf_constraint).mean() * 3e3,  # 1e4      # 3e3
            'embeddings_constraint': embeddings_constraint.mean() * 1e5
           }

def transfer_loss_pcd_no_reduction(model_output, gt):

    gt_sdf = gt['sdf']
    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    embeddings = model_output['latent_vec']

    sdf_constraint = torch.clamp(torch.abs(pred_sdf),-0.5,0.5)-torch.clamp(torch.abs(gt_sdf),-0.5,0.5)
    embeddings_constraint = torch.mean(embeddings ** 2,axis=-1)

    # -----------------
    return {'sdf': torch.abs(sdf_constraint).mean(axis=(1,2)) * 3e3,  # 1e4      # 3e3
            'embeddings_constraint': embeddings_constraint * 1e5
           }
