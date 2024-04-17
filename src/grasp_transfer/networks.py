import torch
from torch import nn
from modules import *
from meta_modules import HyperNetwork
from loss import *
from camera import *
from warp import *

class DeepSDF(nn.Module):
    def __init__(self, num_instances,latent_dim=256,**kwargs):
        super().__init__()

        
        self.latent_dim = latent_dim
        self.latent_codes = nn.Embedding(num_instances, self.latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

        self.sdf_net=SingleBVPNetWithEmbedding(in_features=3,out_features=1,pos_encoding=True)

    def get_latent_code(self,instance_idx):

        embedding = self.latent_codes(instance_idx)

        return embedding

    # for generation
    def inference(self,coords,embedding,epoch = 1000):

        with torch.no_grad():
            model_in = {'coords': coords,'z': embedding}
            model_output = self.sdf_net(model_in,epoch)

            return model_output
    
    def forward(self, model_input,gt,epoch,**kwargs):

        instance_idx = model_input['instance_idx']
        coords = model_input['coords'] # 3 dimensional input coordinates

        embedding = self.latent_codes(instance_idx)
        model_input_with_z = dict()
        model_input_with_z['coords']= coords
        model_input_with_z['z']= embedding
        
        model_output = self.sdf_net(model_input_with_z,epoch)

        x = model_output['model_in'] # input coordinates

        sdf = model_output['model_out']
        
        model_out = {'model_in':model_output['model_in'], 'model_out':sdf, 'latent_vec':embedding}
        losses = implicit_loss_no_normal(model_out, gt)
        return losses
    
class Siren(nn.Module):
    def __init__(self, num_instances,latent_dim=256, hyper_hidden_layers=1,hyper_hidden_features=256,**kwargs):
        super().__init__()

        
        self.latent_dim = latent_dim
        self.latent_codes = nn.Embedding(num_instances, self.latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

        self.sdf_net=SingleBVPNet(in_features=3,out_features=1)

        # Hyper-Net
        self.hyper_net = HyperNetwork(hyper_in_features=self.latent_dim, hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features,
                                      hypo_module=self.sdf_net)

    def get_hypo_net_weights(self, model_input):
        instance_idx = model_input['instance_idx']
        embedding = self.latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def get_latent_code(self,instance_idx):

        embedding = self.latent_codes(instance_idx)

        return embedding

    # for generation
    def inference(self,coords,embedding,epoch = 1000):

        with torch.no_grad():
            model_in = {'coords': coords}
            hypo_params = self.hyper_net(embedding)
            model_output = self.sdf_net(model_in,epoch, params=hypo_params)

            return model_output

    def forward(self, model_input,gt,epoch,**kwargs):

        instance_idx = model_input['instance_idx']
        coords = model_input['coords'] # 3 dimensional input coordinates

        embedding = self.latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)

        model_output = self.sdf_net(model_input,epoch, params=hypo_params)

        x = model_output['model_in'] # input coordinates

        sdf = model_output['model_out']
        grad_sdf = torch.autograd.grad(sdf, [x], grad_outputs=torch.ones_like(sdf), create_graph=True)[0]
        
        model_out = {'model_in':model_output['model_in'], 'model_out':sdf, 'latent_vec':embedding, 'grad_sdf':grad_sdf}
        losses = implicit_loss(model_out, gt,loss_grad_deform=5,loss_grad_temp=1e2,loss_correct=1e2)

        return losses
    
class MyNet(nn.Module):
    def __init__(self, num_instances,latent_dim=256, hyper_hidden_layers=1,hyper_hidden_features=256, affine=True, **kwargs):
        super().__init__()

        
        self.latent_dim = latent_dim
        self.latent_codes = nn.Embedding(num_instances, self.latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)
        self.se3_refine = torch.nn.Embedding(num_instances,6)
        self.affine_tr = torch.nn.Embedding(num_instances,3)
        nn.init.zeros_(self.se3_refine.weight)
        nn.init.ones_(self.affine_tr.weight)
        if affine == False:
            self.affine_tr = self.affine_tr.requires_grad_(False)
        self.sdf_net = SingleBVPNet(in_features=3,out_features=1,pos_encoding=True) #
        self.hyper_net = HyperNetwork(hyper_in_features=self.latent_dim, hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features,
                                      hypo_module=self.sdf_net)    

    def get_hypo_net_weights(self, model_input):
        instance_idx = model_input['instance_idx']
        embedding = self.latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def get_latent_code(self,instance_idx):

        embedding = self.latent_codes(instance_idx)

        return embedding
    
    def get_refine_poses(self,instance_idx,affine=True):
            
        tr_embeds = self.affine_tr(instance_idx)

        affine_transform = torch.diag_embed(torch.hstack([tr_embeds,torch.ones(tr_embeds.shape[0],1,device = tr_embeds.device)]))
        
        refine_pose = lie.se3_to_SE3(self.se3_refine(instance_idx))
        if affine:
            return (affine_transform @ refine_pose)[:,:3,:4], tr_embeds
        else:
            return refine_pose[:,:3,:4]
    def inference(self,coords,embedding,epoch = 1000):

        with torch.no_grad():
            model_in = {'coords': coords}
            hypo_params = self.hyper_net(embedding)
            
            model_output = self.sdf_net(model_in,epoch, params=hypo_params)
            
            return model_output

    def forward(self, model_input,gt,epoch,**kwargs):

        instance_idx = model_input['instance_idx']
        coords = model_input['coords'] # 3 dimensional input coordinates
        
        embedding = self.latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)
        
        refine_poses, _ = self.get_refine_poses(instance_idx)
        transformed_coords = (refine_poses@to_hom(coords).permute(0,2,1)).permute(0,2,1)
        
        gt_aligned = dict()
        gt_aligned['sdf'] = gt['sdf']
        refine_poses = self.get_refine_poses(instance_idx, False)
        
        model_in = {'coords': transformed_coords}
        model_output = self.sdf_net(model_in,epoch, params=hypo_params)
        model_output['model_out'] = model_output['model_out'] 
        
        x = model_output['model_in'] # input coordinates

        sdf = model_output['model_out']        
        model_out = {'model_in':model_output['model_in'], 'model_out':sdf, 'latent_vec':embedding}
        losses = implicit_loss_no_normal(model_out, gt_aligned)
        translations = refine_poses[:,:3,3].norm(dim=-1)
        translation_constraint = torch.where(translations < 0.5, translations,translations*100)
        losses['translation_constraint'] = translation_constraint.mean()
        
        return losses
    
class MyNet2D(nn.Module):
    def __init__(self, num_instances,latent_dim=256, hyper_hidden_layers=1,hyper_hidden_features=256, affine=True, **kwargs):
        super().__init__()

        self.latent_dim = latent_dim
        self.latent_codes = nn.Embedding(num_instances, self.latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)
        self.se2_refine = torch.nn.Embedding(num_instances-1,3)
        self.affine_tr = torch.nn.Embedding(num_instances-1,2)
        nn.init.zeros_(self.se2_refine.weight)
        nn.init.ones_(self.affine_tr.weight)
        if affine == False:
            self.affine_tr = self.affine_tr.requires_grad_(False)
        self.sdf_net=SingleBVPNet(in_features=2,out_features=1,pos_encoding=True)
        
        # Hyper-Net
        self.hyper_net = HyperNetwork(hyper_in_features=self.latent_dim, hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features,
                                      hypo_module=self.sdf_net)    

    def get_hypo_net_weights(self, model_input):
        instance_idx = model_input['instance_idx']
        embedding = self.latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def get_latent_code(self,instance_idx):

        embedding = self.latent_codes(instance_idx)

        return embedding
    
    def get_refine_poses(self,instance_idx,affine=True):
        if torch.all(instance_idx !=0):
            zero_index = len(instance_idx)
            zero_affine = torch.ones(0,2).cuda()
            zero_tr = torch.zeros(0,3,3).cuda()
        else:
            zero_index =torch.where(instance_idx==0)[0]
            zero_affine = torch.ones(1,2).cuda()
            zero_tr = torch.zeros(1,3,3).cuda()
            zero_tr[0,:3,:3]= torch.eye(3).cuda()
        tr_embeds = torch.vstack([self.affine_tr(instance_idx[:zero_index]-1),
                          zero_affine,
                          self.affine_tr(instance_idx[zero_index+1:]-1)])


        affine_transform = torch.diag_embed(torch.hstack([tr_embeds,torch.ones(tr_embeds.shape[0],1,device = tr_embeds.device)]))
        
        refine_pose = torch.vstack([lie2d.se2_to_SE2(self.se2_refine(instance_idx[:zero_index]-1)),
                  zero_tr,
                  lie2d.se2_to_SE2(self.se2_refine(instance_idx[zero_index+1:]-1))])
        if affine:
            return (affine_transform @ refine_pose)[:,:2,:3]
        else:
            return refine_pose[:,:2,:3]
    def inference(self,coords,embedding,epoch = 1000):

        with torch.no_grad():
            model_in = {'coords': coords}
            hypo_params = self.hyper_net(embedding)
            
            model_output = self.sdf_net(model_in,epoch, params=hypo_params)

            return model_output
        
    def forward(self, model_input,gt,epoch,**kwargs):

        instance_idx = model_input['instance_idx']
        coords = model_input['coords'] # 2 dimensional input coordinates
        
        embedding = self.latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)
        
        refine_poses = self.get_refine_poses(instance_idx)
        transformed_coords = (refine_poses@to_hom(coords).permute(0,2,1)).permute(0,2,1)
        
        gt_aligned = dict()
        gt_aligned['sdf'] = gt['sdf']
        refine_poses = self.get_refine_poses(instance_idx, False)
        gt_aligned['normals'] = (refine_poses @ to_hom_with0(gt['normals']).permute(0,2,1)).permute(0,2,1)
        
        model_in = {'coords': transformed_coords}
        model_output = self.sdf_net(model_in,epoch, params=hypo_params)

        x = model_output['model_in'] # input coordinates

        sdf = model_output['model_out']
        grad_sdf = torch.autograd.grad(sdf, [x], grad_outputs=torch.ones_like(sdf), create_graph=True)[0] # normal direction in original shape space
        
        model_out = {'model_in':model_output['model_in'], 'model_out':sdf, 'latent_vec':embedding, 'grad_sdf':grad_sdf}
        losses = implicit_loss(model_out, gt_aligned,loss_grad_deform=5,loss_grad_temp=1e2,loss_correct=1e2)
        
        return losses
    
class MyNetTransfer(nn.Module):
    def __init__(self, num_instances, transfer_model, latent_dim=256,embedding=None):
        super().__init__()

        self.latent_dim = latent_dim
        self.latent_codes = nn.Embedding(num_instances, self.latent_dim) # should this be hidden_num?
        if embedding is None:
            nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)
        else:
            self.latent_codes.weight.data = torch.ones(num_instances,1).cuda()@embedding.reshape(1,latent_dim)  
        self.se3_refine = torch.nn.Embedding(num_instances,6)
        self.affine_tr = torch.nn.Embedding(num_instances,1)
        nn.init.zeros_(self.se3_refine.weight)
        nn.init.ones_(self.affine_tr.weight)

        self.sdf_net=transfer_model.sdf_net
        self.sdf_net.eval()
        self.hyper_net = transfer_model.hyper_net
        self.hyper_net.eval()
                
    def get_hypo_net_weights(self, model_input):
        instance_idx = model_input['instance_idx']
        embedding = self.latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def get_latent_code(self,instance_idx):

        embedding = self.latent_codes(instance_idx)

        return embedding
    
    def get_refine_poses(self,instance_idx,affine=True):
        tr_embeds = self.affine_tr(instance_idx)@torch.ones(1,3).cuda()

        affine_transform = torch.diag_embed(torch.hstack([tr_embeds,torch.ones(tr_embeds.shape[0],1,device = tr_embeds.device)]))
        refine_pose = lie.se3_to_SE3(self.se3_refine(instance_idx))
        if affine:
            return (affine_transform @ refine_pose)[:,:3,:4], tr_embeds
        else:
            return refine_pose[:,:3,:4]
    def inference(self,coords,embedding,epoch = 1000):

        with torch.no_grad():
            model_in = {'coords': coords}
            hypo_params = self.hyper_net(embedding)
            
            model_output = self.sdf_net(model_in,epoch, params=hypo_params)

            return model_output

    def forward(self, model_input,gt,epoch,**kwargs):

        instance_idx = model_input['instance_idx']
        coords = model_input['coords'] # 3 dimensional input coordinates
        
        embedding = self.latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)
        
        refine_poses, _ = self.get_refine_poses(instance_idx)
        transformed_coords = (refine_poses@to_hom(coords).permute(0,2,1)).permute(0,2,1)
        
        gt_aligned = dict()
        gt_aligned['sdf'] = gt['sdf']
        
        model_in = {'coords': transformed_coords}
        model_output = self.sdf_net(model_in,epoch, params=hypo_params)
        model_output['model_out'] = model_output['model_out']

        sdf = model_output['model_out']
        
        model_out = {'model_in':model_output['model_in'], 'model_out':sdf, 'latent_vec':embedding}
        losses = transfer_loss(model_out, gt_aligned,sdf_loss=3e3)

        return losses
    
class MyNetTransferPCD(nn.Module):
    def __init__(self, num_instances, transfer_model, latent_dim=256,embedding=None):
        super().__init__()

        
        self.latent_dim = latent_dim
        self.latent_codes = nn.Embedding(num_instances, self.latent_dim) # should this be hidden_num?
        if embedding is None:
            nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)
        else:
            self.latent_codes.weight.data = torch.ones(num_instances,1).cuda()@embedding.reshape(1,latent_dim)        
        self.se3_refine = torch.nn.Embedding(num_instances,6)
        self.affine_tr = torch.nn.Embedding(num_instances,1)
        nn.init.zeros_(self.se3_refine.weight)
        nn.init.ones_(self.affine_tr.weight)

        self.sdf_net=transfer_model.sdf_net
        self.sdf_net.eval()
        # Hyper-Net
        self.hyper_net = transfer_model.hyper_net
        self.hyper_net.eval()
                
    def get_hypo_net_weights(self, model_input):
        instance_idx = model_input['instance_idx']
        embedding = self.latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def get_latent_code(self,instance_idx):

        embedding = self.latent_codes(instance_idx)

        return embedding
    
    def get_refine_poses(self,instance_idx,affine=True):
        tr_embeds = self.affine_tr(instance_idx)@torch.ones(1,3).cuda()

        affine_transform = torch.diag_embed(torch.hstack([tr_embeds,torch.ones(tr_embeds.shape[0],1,device = tr_embeds.device)]))
        refine_pose = lie.se3_to_SE3(self.se3_refine(instance_idx))
        if affine:
            return (affine_transform @ refine_pose)[:,:3,:4], tr_embeds
        else:
            return refine_pose[:,:3,:4]
            # for generation
    def inference(self,coords,embedding,epoch = 2000):

        with torch.no_grad():
            model_in = {'coords': coords}
            hypo_params = self.hyper_net(embedding)
            
            model_output = self.sdf_net(model_in,epoch, params=hypo_params)

            return model_output

    def forward(self, model_input,gt,epoch,reduction=True,**kwargs):

        instance_idx = model_input['instance_idx']
        coords = model_input['coords'] # 3 dimensional input coordinates
        
        embedding = self.latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)
        
        refine_poses, affine_codes = self.get_refine_poses(instance_idx)
        translation = refine_poses[:,:3,3]
        transformed_coords = (refine_poses@to_hom(coords).permute(0,2,1)).permute(0,2,1)
        
        gt_aligned = dict()
        gt_aligned['sdf'] = gt['sdf']
        
        model_in = {'coords': transformed_coords}
        model_output = self.sdf_net(model_in,epoch, params=hypo_params)
        model_output['model_out'] = model_output['model_out'] 
        x = model_output['model_in'] # input coordinates

        sdf = model_output['model_out']

        model_out = {'model_in':model_output['model_in'], 'model_out':sdf, 'latent_vec':embedding,}
        if reduction:
            losses = transfer_loss_pcd(model_out, gt_aligned)
            affine_constraint = torch.mean((affine_codes-1)**2,axis=1)*250
            losses['affine'] = affine_constraint
 
        else:
            losses = transfer_loss_pcd_no_reduction(model_out, gt_aligned)
        return losses
    