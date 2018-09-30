import numpy as np
import numpy.linalg as la

def macaulay(x):
    x[x<0] = 0
    return x

def series_kinematic_hardening(k_n, H_n, E, sigma_x_history, sigma_y_history):
    
    N = len(H_n)

    # Initialise the model state parameters
    epsilon_x = 0
    epsilon_y = 0
    chi_n_x = np.zeros(N)
    chi_n_y = np.zeros(N)
    alpha_n_x = np.zeros(N)
    alpha_n_y = np.zeros(N)
    
    d2_g_d_s2 = -1/E
    d2_g_d_an2 =  H_n
    d2_g_d_san = -np.ones(N)
    d2_g_d_ans = -np.ones(N)

    sigma_0_x = 0
    sigma_0_y = 0
    
    epsilon_x_history = np.zeros(len(sigma_x_history))
    epsilon_y_history = np.zeros(len(sigma_y_history))

    indices = np.arange(0,len(sigma_x_history))

    # Calculate the incremental response
    for index, sigma_x, sigma_y in zip(indices, sigma_x_history, sigma_y_history):
    
        d_sigma_x = sigma_x - sigma_0_x
        d_sigma_y = sigma_y - sigma_0_y
        
        chi_norm = (chi_n_x**2 + chi_n_y**2)**(1/2)
        y_n = chi_norm - k_n
    
        d_y_n_d_chi_n_x = chi_n_x / chi_norm
        d_y_n_d_chi_n_x[np.abs(chi_norm) < 1e-10] = 0
        d_y_n_d_chi_n_y = chi_n_y / chi_norm
        d_y_n_d_chi_n_y[np.abs(chi_norm) < 1e-10] = 0
        
        lambda_n = (d_y_n_d_chi_n_x * d_sigma_x + d_y_n_d_chi_n_y * d_sigma_y) / H_n
        lambda_n[lambda_n < 1e-10] = 0
        lambda_n[y_n < -1e-10] = 0
    
        d_alpha_n_x = lambda_n * d_y_n_d_chi_n_x
        d_alpha_n_y = lambda_n * d_y_n_d_chi_n_y
                    
        d_epsilon_x = - (d2_g_d_s2 * d_sigma_x + np.sum(d2_g_d_san * d_alpha_n_x))
        d_epsilon_y = - (d2_g_d_s2 * d_sigma_y + np.sum(d2_g_d_san * d_alpha_n_y))
    
        d_chi_n_x = - (d2_g_d_ans * d_sigma_x + d2_g_d_an2 * d_alpha_n_x)
        d_chi_n_y = - (d2_g_d_ans * d_sigma_y + d2_g_d_an2 * d_alpha_n_y)
        
        epsilon_x = epsilon_x + d_epsilon_x
        epsilon_y = epsilon_y + d_epsilon_y
    
        chi_n_x = chi_n_x + d_chi_n_x
        chi_n_y = chi_n_y + d_chi_n_y
    
        alpha_n_x = alpha_n_x + d_alpha_n_x
        alpha_n_y = alpha_n_y + d_alpha_n_y
    
        sigma_0_x = sigma_x
        sigma_0_y = sigma_y
            
        epsilon_x_history[index] = epsilon_x  
        epsilon_y_history[index] = epsilon_y
        
    return epsilon_x_history, epsilon_y_history


def parallel_kinematic_hardening(k_n, H_n, sigma_x_history, sigma_y_history):

    E_0 = np.sum(H_n)
    N = len(H_n)

    # Initialise the model state parameters
    epsilon_x = 0
    epsilon_y = 0
    chi_n_x = np.zeros(N)
    chi_n_y = np.zeros(N)
    alpha_n_x = np.zeros(N)
    alpha_n_y = np.zeros(N)
    
    d2_g_d_s2 = -1 / E_0
    d2_g_d_an2 = -np.matmul(np.transpose(np.asmatrix(H_n)), np.asmatrix(H_n)) / E_0 + np.diag(H_n)
    d2_g_d_san = -H_n / E_0
    d2_g_d_ans = -H_n / E_0

    sigma_0_x = 0
    sigma_0_y = 0

    epsilon_x_history = np.zeros(len(sigma_x_history))
    epsilon_y_history = np.zeros(len(sigma_y_history))

    indices = np.arange(0,len(sigma_x_history))

    # Calculate the incremental response
    for index, sigma_x, sigma_y in zip(indices, sigma_x_history, sigma_y_history):
        
        d_sigma_x = sigma_x - sigma_0_x
        d_sigma_y = sigma_y - sigma_0_y
    
        chi_n_x = H_n * (sigma_x + np.sum(H_n*alpha_n_x)) / E_0 - H_n * alpha_n_x
        chi_n_y = H_n * (sigma_y + np.sum(H_n*alpha_n_y)) / E_0 - H_n * alpha_n_y
        
        chi_norm = (chi_n_x**2 + chi_n_y**2)**(1/2)
        y_n = chi_norm - k_n
    
        d_y_n_d_chi_n_x = chi_n_x / chi_norm
        d_y_n_d_chi_n_x[chi_norm==0] = 0
        d_y_n_d_chi_n_y = chi_n_y / chi_norm
        d_y_n_d_chi_n_y[chi_norm==0] = 0
                
        # Solve A * lambda_n = b
        b = np.zeros((len(H_n),1))
        A = np.zeros_like(d2_g_d_an2)
        lambda_n = np.zeros_like(H_n)
        for i_n in range(0,len(H_n)):
            b[i_n,0] = - (d_y_n_d_chi_n_x[i_n] * d_sigma_x + d_y_n_d_chi_n_y[i_n] * d_sigma_y) * d2_g_d_ans[i_n]
            A[i_n,:] = np.asarray(d2_g_d_an2[i_n,:]) * ((d_y_n_d_chi_n_x[i_n] * d_y_n_d_chi_n_x + d_y_n_d_chi_n_y[i_n] * d_y_n_d_chi_n_y))
        
        y_active = ((y_n > 0) * ((d_sigma_x * d_y_n_d_chi_n_x + d_sigma_y * d_y_n_d_chi_n_y) > 0))

        if np.sum(y_active) > 0:
        
            lambda_active = la.solve(A[y_active,:][:,y_active], b[y_active])
            lambda_active = lambda_active[:,0]
            lambda_n[y_active] = lambda_active  
                        
        d_alpha_n_x = lambda_n * d_y_n_d_chi_n_x
        d_alpha_n_y = lambda_n * d_y_n_d_chi_n_y
    
                    
        d_epsilon_x = - (d2_g_d_s2 * d_sigma_x + np.sum(d2_g_d_san * d_alpha_n_x))
        d_epsilon_y = - (d2_g_d_s2 * d_sigma_y + np.sum(d2_g_d_san * d_alpha_n_y))
    
        #d_chi_n_x = - (d2_g_d_ans * d_sigma_x + d2_g_d_an2 * d_alpha_n_x)
        #d_chi_n_y = - (d2_g_d_ans * d_sigma_y + d2_g_d_an2 * d_alpha_n_y)
            
        epsilon_x = epsilon_x + d_epsilon_x
        epsilon_y = epsilon_y + d_epsilon_y
    
        #chi_n_x = chi_n_x + d_chi_n_x
        #chi_n_y = chi_n_y + d_chi_n_y
    
        alpha_n_x = alpha_n_x + d_alpha_n_x
        alpha_n_y = alpha_n_y + d_alpha_n_y
    
        sigma_0_x = sigma_x
        sigma_0_y = sigma_y
            
        epsilon_x_history[index] = epsilon_x  
        epsilon_y_history[index] = epsilon_y
        
    return epsilon_x_history, epsilon_y_history
    
def parallel_kinematic_hardening_viscous(k_n, H_n, mu, dt, sigma_x_history, sigma_y_history):

    E_0 = np.sum(H_n)
    N = len(H_n)

    # Initialise the model state parameters
    epsilon_x = 0
    epsilon_y = 0
    chi_n_x = np.zeros(N)
    chi_n_y = np.zeros(N)
    alpha_n_x = np.zeros(N)
    alpha_n_y = np.zeros(N)

    d2_g_d_s2 = -1/E_0
    d2_g_d_an2 = -np.matmul(np.transpose(np.asmatrix(H_n)), np.asmatrix(H_n)) / E_0 + np.diag(H_n)
    d2_g_d_san = -np.transpose(H_n) / E_0
    d2_g_d_ans = -H_n / E_0

    sigma_0_x = 0
    sigma_0_y = 0
    
    epsilon_x_history = np.zeros(len(sigma_x_history))
    epsilon_y_history = np.zeros(len(sigma_y_history))

    indices = np.arange(0,len(sigma_x_history))

    # Calculate the incremental response
    for index, sigma_x, sigma_y in zip(indices, sigma_x_history, sigma_y_history):
    
        d_sigma_x = sigma_x - sigma_0_x
        d_sigma_y = sigma_y - sigma_0_y
    
        chi_n_x = H_n * (sigma_x + np.sum(H_n*alpha_n_x)) / E_0 - H_n * alpha_n_x
        chi_n_y = H_n * (sigma_y + np.sum(H_n*alpha_n_y)) / E_0 - H_n * alpha_n_y
    
        D = 1/((chi_n_x**2+chi_n_y**2)**(1/2))
        D[chi_n_x**2+chi_n_y**2 == 0] = 0
    
        d_w_d_chi_n_x = 1 / mu * macaulay((chi_n_x**2+chi_n_y**2)**(1/2) - k_n) * chi_n_x * D
        d_w_d_chi_n_y = 1 / mu * macaulay((chi_n_x**2+chi_n_y**2)**(1/2) - k_n) * chi_n_y * D
    
        d_alpha_n_x = d_w_d_chi_n_x * dt
        d_alpha_n_y = d_w_d_chi_n_y * dt
    
        d_epsilon_x = - (d2_g_d_s2 * d_sigma_x + np.matmul(d2_g_d_san, d_alpha_n_x))
        d_epsilon_y = - (d2_g_d_s2 * d_sigma_y + np.matmul(d2_g_d_san, d_alpha_n_y))
    
        #d_chi_n_x = - (d2_g_d_ans * d_sigma_x + np.transpose(np.matmul(d2_g_d_an2, np.transpose(d_alpha_n_x))))
        #d_chi_n_y = - (d2_g_d_ans * d_sigma_y + np.transpose(np.matmul(d2_g_d_an2, np.transpose(d_alpha_n_y))))
    
        epsilon_x = epsilon_x + d_epsilon_x
        epsilon_y = epsilon_y + d_epsilon_y
    
        #chi_n_x = chi_n_x + d_chi_n_x
        #chi_n_y = chi_n_y + d_chi_n_y
    
        alpha_n_x = alpha_n_x + d_alpha_n_x
        alpha_n_y = alpha_n_y + d_alpha_n_y
    
        sigma_0_x = sigma_x
        sigma_0_y = sigma_y
            
        epsilon_x_history[index] = epsilon_x  
        epsilon_y_history[index] = epsilon_y
        
    return epsilon_x_history, epsilon_y_history

def series_kinematic_hardening_viscous(k_n, H_n, E, mu, dt, sigma_x_history, sigma_y_history):
    
    N = len(H_n)

    # Initialise the model state parameters
    epsilon_x = 0
    epsilon_y = 0
    chi_n_x = np.zeros(N)
    chi_n_y = np.zeros(N)
    alpha_n_x = np.zeros(N)
    alpha_n_y = np.zeros(N)
    
    d2_g_d_s2 = -1/E
    d2_g_d_an2 =  H_n
    d2_g_d_san = -np.ones(N)
    d2_g_d_ans = -np.ones(N)

    sigma_0_x = 0
    sigma_0_y = 0

    epsilon_x_history = np.zeros(len(sigma_x_history))
    epsilon_y_history = np.zeros(len(sigma_y_history))
    
    indices = np.arange(0,len(sigma_x_history))

    # Calculate the incremental response
    for index, sigma_x, sigma_y in zip(indices, sigma_x_history, sigma_y_history):
    
        d_sigma_x = sigma_x - sigma_0_x
        d_sigma_y = sigma_y - sigma_0_y
        
        D = 1/((chi_n_x**2+chi_n_y**2)**(1/2))
        D[D==np.inf] = 0
    
        d_w_d_chi_n_x = 1 / mu * macaulay((chi_n_x**2+chi_n_y**2)**(1/2) - k_n) * chi_n_x * D
        d_w_d_chi_n_y = 1 / mu * macaulay((chi_n_x**2+chi_n_y**2)**(1/2) - k_n) * chi_n_y * D
    
        d_alpha_n_x = d_w_d_chi_n_x * dt
        d_alpha_n_y = d_w_d_chi_n_y * dt
        
        d_epsilon_x = - (d2_g_d_s2 * d_sigma_x + np.sum(d2_g_d_san * d_alpha_n_x))
        d_epsilon_y = - (d2_g_d_s2 * d_sigma_y + np.sum(d2_g_d_san * d_alpha_n_y))
    
        d_chi_n_x = - (d2_g_d_ans * d_sigma_x + d2_g_d_an2 * d_alpha_n_x)
        d_chi_n_y = - (d2_g_d_ans * d_sigma_y + d2_g_d_an2 * d_alpha_n_y)
        
        epsilon_x = epsilon_x + d_epsilon_x
        epsilon_y = epsilon_y + d_epsilon_y
    
        chi_n_x = chi_n_x + d_chi_n_x
        chi_n_y = chi_n_y + d_chi_n_y
    
        alpha_n_x = alpha_n_x + d_alpha_n_x
        alpha_n_y = alpha_n_y + d_alpha_n_y
    
        sigma_0_x = sigma_x
        sigma_0_y = sigma_y
            
        epsilon_x_history[index] = epsilon_x  
        epsilon_y_history[index] = epsilon_y
    
    return epsilon_x_history, epsilon_y_history

