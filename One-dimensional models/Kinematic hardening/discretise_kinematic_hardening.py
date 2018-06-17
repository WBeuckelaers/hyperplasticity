import numpy as np

def discretise_kinematic_hardening_parallel(sigma, epsilon):
    
    sigma_0 = np.append(0, sigma)
    epsilon_0 = np.append(0, epsilon)
    
    E = (sigma_0[1:]-sigma_0[0:-1])/(epsilon_0[1:]-epsilon_0[0:-1])
    E_0 = np.append(E, 0)
 
    H = E_0[0:-1] - E_0[1:]
    k = H * epsilon
        
    return k, H
    
def discretise_kinematic_hardening_series(sigma, epsilon):
    
    k = sigma
    
    sigma_0 = np.append(0, sigma)
    epsilon_0 = np.append(0, epsilon)
    
    E = (sigma_0[1:]-sigma_0[0:-1])/(epsilon_0[1:]-epsilon_0[0:-1])
            
    H_0 = E[0]
    H = np.array([])
    
    for i in range(0, len(sigma)-1):
        H = np.append(H, (1 / (1 / E[i+1] - (np.sum(1/H) + 1 / H_0))))
        
    H = np.append(H, 0)
         
    return k, H, H_0