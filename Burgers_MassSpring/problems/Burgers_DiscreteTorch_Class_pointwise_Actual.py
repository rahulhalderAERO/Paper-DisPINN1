import numpy as np
import torch


class Burgers_Discrete:
     
     def __init__(self,L,T,j,mu,um1j,um1jp1,um1jm1,up1j):
     
         self.nx = len(L)
         self.nt = len(T)
         self.mu = mu
         self.L = L
         self.T = T
         self.j = j
         self.um1j = um1j
         self.um1jm1 = um1jm1
         self.um1jp1 = um1jp1
         self.up1j = up1j         
         self.dx = self.L[1]-self.L[0]
         self.dt = self.T[1]-self.T[0] 
         self.const = (self.mu/(self.dx*self.dx))
         
     def Compute_Linear(self):
     
     
          if (self.j == 1):          
            return - 2*self.um1j + self.um1jp1
      
          elif (self.j == self.nx-2):
            return - 2*self.um1j + self.um1jm1
          
          else:
            return (self.um1jp1 - 2*self.um1j + self.um1jm1)

          # return (self.um1[self.j+1] - 2*self.um1[self.j] + self.um1[self.j-1])
      
     def Compute_Nonlinear(self):
     
         if (self.j == self.nx-2):         
            return - self.um1j*self.um1j    
         else:         
            return (self.um1jp1 - self.um1j)*self.um1j  
         
          # return (self.um1[self.j+1] - self.um1[self.j])*self.um1[self.j] 
         
         

     def Burgers_Residual_Temp(self):
         return (1/self.dt)*(self.up1j-self.um1j)
     
     def Burgers_Residual_Spatial(self):
         return - self.const*self.Compute_Linear() + (1.0/self.dx)*self.Compute_Nonlinear()
         
      
     def Burgers_Residual(self):
         return (self.dx*(self.Burgers_Residual_Temp() + self.Burgers_Residual_Spatial()))
         # return self.const
         
         
     
        
     
         
         
          
        
