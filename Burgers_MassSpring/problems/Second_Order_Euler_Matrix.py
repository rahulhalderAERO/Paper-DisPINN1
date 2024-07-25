import numpy as np
import torch

class MassSpring_Discrete:
     
     def __init__(self,nt,u):
     
         self.T = torch.linspace(0, (nt-1)*(torch.pi/18), nt).reshape(-1, 1)
         self.nt = nt
         self.u = u
         self.dt = self.T[1]-self.T[0]
    
     def Temporal_Derivative(self):
         A =  np.zeros((self.nt, self.nt)) 
         for i in range(self.nt):
              if (i == 0) :
                  A[i,i] = 1.5
              elif (i == 1) :
                  A[i,i] = 1.5
                  A[i,i-1]= -2
              else:
                  A[i,i] = 1.5
                  A[i,i-1]= -2
                  A[i,i-2] = 0.5
              
         A_tensor = torch.from_numpy(A).float()
         return (1/self.dt)*A_tensor
          
     def Vel_Compute(self):
         return torch.matmul(self.Temporal_Derivative(),self.u)
         
     def Acc_Compute(self):
         return torch.matmul(self.Temporal_Derivative(),self.Vel_Compute())

# nt = 10
# u = 0.01*torch.linspace(0, (nt-1)*1, nt).reshape(-1, 1)
# prob = MassSpring_Discrete(nt,u)
# du2dt2 = prob.Acc_Compute()
# print("du2dt2 = ", du2dt2)