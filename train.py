import sys
import os
sys.path.append(os.getcwd())
import time
from tqdm import tqdm
from hyper_mlp import Hyper_mlp
import numpy as np
import torch
from tools.scalarization_function import CS_functions,EPOSolver
from tools.hv import HvMaximization
from dataset import get_ray
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def train_epoch(device, cfg, criterion, pb):
    name = cfg['NAME']
    mode = cfg['MODE']
    ray_hidden_dim = cfg['TRAIN']['Ray_hidden_dim']
    out_dim = cfg['TRAIN']['Out_dim']
    n_tasks = cfg['TRAIN']['N_task']
    num_hidden_layer = cfg['TRAIN']['Solver'][criterion]['Num_hidden_layer']
    last_activation = cfg['TRAIN']['Solver'][criterion]['Last_activation']
    ref_point = tuple(map(int, cfg['TRAIN']['Ref_point'].split(',')))
    lr = cfg['TRAIN']['OPTIMIZER']['Lr']
    wd = cfg['TRAIN']['OPTIMIZER']['WEIGHT_DECAY']
    type_opt = cfg['TRAIN']['OPTIMIZER']['TYPE']
    epochs = cfg['TRAIN']['Epoch']
    alpha_r = cfg['TRAIN']['Alpha']
    start = time.time()
    
    if criterion == 'HVI':
        sol = []
        hnet = Hyper_mlp(ray_hidden_dim = ray_hidden_dim, out_dim = out_dim, n_tasks = n_tasks,num_hidden_layer=num_hidden_layer,last_activation=last_activation)
        hnet = hnet.to(device)
        if type_opt == 'adam':
            optimizer = torch.optim.Adam(hnet.parameters(), lr = lr, weight_decay=wd)
        elif type_opt == 'adamw':
            optimizer = torch.optim.AdamW(hnet.parameters(), lr = lr, weight_decay=wd)

        net_list = []
        n_mo_obj = cfg['TRAIN']['N_task']
        hesophat = cfg['TRAIN']['Solver'][criterion]['Rho']
        bs = 8 #B
        
        end = np.pi/2
        mo_opt = HvMaximization(bs, n_mo_obj, ref_point)
        
        dem = 0
        train_dt = get_ray(alpha_r)
        train_loader = torch.utils.data.DataLoader(
                        dataset=train_dt,
                        batch_size=bs,num_workers=4,
                        shuffle=True)
        num_epoch = 4
        for epoch in tqdm(range(num_epoch)):
            for i, batch in enumerate(train_loader):
                rays_batch = batch[0] # (B,ray_dim)
                output = hnet(rays_batch)[0] # (B,output_dim)
                objectives = pb.get_values(output) # (m,B)
                obj_values = []
                for i in range(len(objectives)):
                    obj_values.append(objectives[i])
                loss_per_sample = torch.stack(obj_values).transpose(1,0) # (B,ray_dim)
                tmp1 = torch.sum((loss_per_sample*rays_batch),1)/(torch.norm(loss_per_sample, dim=1)*torch.norm(rays_batch, dim=1))
                penalty = hesophat * torch.mean(tmp1) # (1)
                dynamic_weights_per_sample = torch.ones(rays_batch.shape[0], n_mo_obj, 1) # (B,m,1)
                loss_numpy_per_sample = loss_per_sample.unsqueeze(0).cpu().detach().numpy().transpose(0,2,1)
                weights_task = mo_opt.compute_weights(loss_numpy_per_sample[0,:,:])
                dynamic_weights_per_sample[:, :, 0] = weights_task.permute(1,0) # (B,m,1)
                tmp2 = torch.sum(dynamic_weights_per_sample* loss_per_sample.unsqueeze(2), dim=1)
                total_dynamic_loss = torch.mean(tmp2)
                total_dynamic_loss -= penalty
                total_dynamic_loss /= rays_batch.shape[0]
                if epoch == num_epoch-1:
                    sol.append(output.cpu().detach().numpy().tolist()) # save output in the last epoch
                optimizer.zero_grad()
                total_dynamic_loss.backward()
                optimizer.step()
    end = time.time()
    time_training = end-start
    torch.save(hnet,("./save_weights/best_weight_"+str(criterion)+"_"+str(mode)+"_"+str(name)+"_" + str(cfg['TRAIN']['Ray_hidden_dim'])+"_attention.pt"))
    sol = np.array(sol).reshape(len(train_loader)*bs,2).tolist()
    return sol,time_training
