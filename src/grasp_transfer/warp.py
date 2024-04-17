#Adapted from BARF, https://github.com/chenhsuanlin/bundle-adjusting-NeRF/blob/main/warp.py
import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F

import camera

class Lie2D():

    def so2_to_SO2(self,theta): # [...,1]
        thetax = torch.stack([torch.cat([theta.cos(),-theta.sin()],dim=-1),
                              torch.cat([theta.sin(),theta.cos()],dim=-1)],dim=-2)
        R = thetax
        return R

    def SO2_to_so2(self,R): # [...,2,2]
        theta = torch.atan2(R[...,1,0],R[...,0,0])
        return theta[...,None]

    def so2_jacobian(self,X,theta): # [...,N,2],[...,1]
        dR_dtheta = torch.stack([torch.cat([-theta.sin(),-theta.cos()],dim=-1),
                                 torch.cat([theta.cos(),-theta.sin()],dim=-1)],dim=-2) # [...,2,2]
        J = X@dR_dtheta.transpose(-2,-1)
        return J[...,None] # [...,N,2,1]

    def se2_to_SE2(self,delta): # [...,3]
        u,theta = delta.split([2,1],dim=-1)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        V = torch.stack([torch.cat([A,-B],dim=-1),
                         torch.cat([B,A],dim=-1)],dim=-2)
        R = self.so2_to_SO2(theta)
        Rt = torch.cat([R,V@u[...,None]],dim=-1)
        temp = torch.cat([torch.zeros(Rt.shape[0],1,2,device=u.device,dtype=torch.float32),
                          torch.ones(Rt.shape[0],1,1,device=u.device,dtype=torch.float32)],axis=-1)
        H = torch.cat([Rt,temp],axis=1)        
        return H

    def SE2_to_se2(self,Rt,eps=1e-7): # [...,2,3]
        R,t = Rt.split([2,1],dim=-1)
        theta = self.SO2_to_so2(R)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        denom = (A**2+B**2+eps)[...,None]
        invV = torch.stack([torch.cat([A,B],dim=-1),
                            torch.cat([-B,A],dim=-1)],dim=-2)/denom
        u = (invV@t)[...,0]
        delta = torch.cat([u,theta],dim=-1)
        return delta

    def se2_jacobian(self,X,delta): # [...,N,2],[...,3]
        u,theta = delta.split([2,1],dim=-1)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        D = self.taylor_D(theta)
        V = torch.stack([torch.cat([A,-B],dim=-1),
                         torch.cat([B,A],dim=-1)],dim=-2)
        R = self.so2_to_SO2(theta)
        dV_dtheta = torch.stack([torch.cat([C,-D],dim=-1),
                                 torch.cat([D,C],dim=-1)],dim=-2) # [...,2,2]
        dt_dtheta = dV_dtheta@u[...,None] # [...,2,1]
        J_so2 = self.so2_jacobian(X,theta) # [...,N,2,1]
        dX_dtheta = J_so2+dt_dtheta[...,None,:,:] # [...,N,2,1]
        dX_du = V[...,None,:,:].repeat(*[1]*(len(dX_dtheta.shape)-3),dX_dtheta.shape[-3],1,1)
        J = torch.cat([dX_du,dX_dtheta],dim=-1)
        return J # [...,N,2,3]

    def sl3_to_SL3(self,h):
        # homography: directly expand matrix exponential
        # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.61.6151&rep=rep1&type=pdf
        h1,h2,h3,h4,h5,h6,h7,h8 = h.chunk(8,dim=-1)
        A = torch.stack([torch.cat([h5,h3,h1],dim=-1),
                         torch.cat([h4,-h5-h6,h2],dim=-1),
                         torch.cat([h7,h8,h6],dim=-1)],dim=-2)
        H = A.matrix_exp()
        return H

    def taylor_A(self,x,nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            if i>0: denom *= (2*i)*(2*i+1)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    def taylor_B(self,x,nth=10):
        # Taylor expansion of (1-cos(x))/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+1)*(2*i+2)
            ans = ans+(-1)**i*x**(2*i+1)/denom
        return ans

    def taylor_C(self,x,nth=10):
        # Taylor expansion of (x*cos(x)-sin(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+2)*(2*i+3)
            ans = ans+(-1)**(i+1)*x**(2*i+1)*(2*i+2)/denom
        return ans

    def taylor_D(self,x,nth=10):
        # Taylor expansion of (x*sin(x)+cos(x)-1)/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+1)*(2*i+2)
            ans = ans+(-1)**i*x**(2*i)*(2*i+1)/denom
        return ans

lie2d = Lie2D()