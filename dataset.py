import torch
import os
import numpy as np

def get_ray(alpha_r):
    rays = []
    for i in range(30000):
        ray = np.random.dirichlet((alpha_r, alpha_r), 1).astype(np.float32)[0].tolist()
        rays.append(ray)
    rays = np.array(rays)
    # ind = np.lexsort((rays[:,1],rays[:,0]))    
    # rays_sort = rays[ind]
    rays_sort = rays
    rays_train = torch.from_numpy(rays_sort).float()
    train_dt = torch.utils.data.TensorDataset(rays_train)
    return train_dt
if __name__ == "__main__":
    train_dt = get_ray(0.2)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dt,
        batch_size=1,num_workers=4,
        shuffle=False)
    for i, batch in enumerate(train_loader):
        print(batch)
        #break