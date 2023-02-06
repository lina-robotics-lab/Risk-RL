# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from scipy.special import kl_div
import warnings
warnings.filterwarnings('error')

def DP(lst,epsilon,H):
    l = len(lst)
    V = np.zeros([H, l])
    pi = np.zeros([H-1,l])
    V[H-1,:] = lst
    for h in range(H-2,-1,-1):
        for i in range(l):
            r1 = (i+1)%l
            r2 = (i+2)%l
            l1 = (i-1)%l
            l2 = (i-2)%l
            val3 = (1-2*epsilon)*V[h+1,r1] + epsilon*V[h+1,r2] + epsilon*V[h+1,i]
            val2 = (1-2*epsilon)*V[h+1,i] + epsilon*V[h+1,r1] + epsilon*V[h+1,l1]
            val1 = (1-2*epsilon)*V[h+1,l1] + epsilon*V[h+1,l2] + epsilon*V[h+1,i]
            pi[h,i] = np.argmax([val1, val2, val3]) - 1
            pi = pi.astype(int)
            V[h,i] = lst[i] + np.max([val1, val2, val3])
    return V, pi

def policy_eval(lst, epsilon, H, pi, rho = None):
    l = len(lst)
    V = np.zeros([H, l])
    V[H - 1, :] = lst
    for h in range(H - 2, -1, -1):
        for i in range(l):
            a = pi[h, i]
            r1 = (i + a + 1) % l
            l1 = (i + a - 1) % l
            p = (i+a)%l
            val = (1 - 2 * epsilon) * V[h + 1, p] + epsilon * V[h + 1, r1] + epsilon * V[h + 1, l1]
            V[h, i] = lst[i] + val
    if rho is not None:
        return np.dot(rho, V[0,:])
    return V

def policy_eval_robust(lst, epsilon, H, pi, delta, rho= None):
    l = len(lst)
    V = np.zeros([H, l])
    V[H - 1, :] = lst
    for h in range(H - 2, -1, -1):
        for i in range(l):
            a = pi[h, i]
            r1 = (i + a + 1) % l
            l1 = (i + a - 1) % l
            p = (i + a) % l
            x = np.array([V[h + 1, l1], V[h + 1, p], V[h + 1, r1]])
            mu = np.array([epsilon, 1-2*epsilon, epsilon])
            nu = sup_inf_solver(-x, mu, delta)
            val = np.dot(x,nu) #(1 - 2 * epsilon) * V[h + 1, p] + epsilon * V[h + 1, r1] + epsilon * V[h + 1, l1]
            V[h, i] = lst[i] + val
    if rho is not None:
        return np.dot(rho, V[0, :])
    return V

def DP_exp(lst,epsilon,H,beta):
    l = len(lst)
    V = np.zeros([H, l])
    pi = np.zeros([H - 1, l])
    V[H - 1, :] = lst
    if beta < 0:
        for h in range(H - 2, -1, -1):
            for i in range(l):
                r1 = (i + 1) % l
                r2 = (i + 2) % l
                l1 = (i - 1) % l
                l2 = (i - 2) % l
                val3 = np.log((1 - 2 * epsilon) * np.exp(beta* V[h + 1, r1]) + epsilon * np.exp(beta* V[h + 1, r2]) + epsilon * np.exp(beta* V[h + 1, i]))
                val2 = np.log((1 - 2 * epsilon) * np.exp(beta* V[h + 1, i]) + epsilon * np.exp(beta* V[h + 1, r1]) + epsilon * np.exp(beta* V[h + 1, l1]))
                val1 = np.log((1 - 2 * epsilon) * np.exp(beta* V[h + 1, l1]) + epsilon * np.exp(beta* V[h + 1, l2]) + epsilon * np.exp(beta* V[h + 1, i]))
                pi[h, i] = np.argmin([val1, val2, val3]) - 1
                pi = pi.astype(int)
                V[h, i] = lst[i] + np.min([val1, val2, val3])/beta
        return V, pi
    if beta > 0:
        for h in range(H - 2, -1, -1):
            for i in range(l):
                r1 = (i + 1) % l
                r2 = (i + 2) % l
                l1 = (i - 1) % l
                l2 = (i - 2) % l
                val3 = np.log((1 - 2 * epsilon) * np.exp(beta* V[h + 1, r1]) + epsilon * np.exp(beta* V[h + 1, r2]) + epsilon * np.exp(beta* V[h + 1, i]))
                val2 = np.log((1 - 2 * epsilon) * np.exp(beta* V[h + 1, i]) + epsilon * np.exp(beta* V[h + 1, r1]) + epsilon * np.exp(beta* V[h + 1, l1]))
                val1 = np.log((1 - 2 * epsilon) * np.exp(beta* V[h + 1, l1]) + epsilon * np.exp(beta* V[h + 1, l2]) + epsilon * np.exp(beta* V[h + 1, i]))
                pi[h, i] = np.argmax([val1, val2, val3]) - 1
                pi = pi.astype(int)
                V[h, i] = lst[i] + np.max([val1, val2, val3])/beta
        return V, pi

def sup_inf_solver(x, mu, epsilon):
    x = x-min(x)
    x = x/(np.linalg.norm(x)+1e-5)
    beta0 = 0
    beta = 1
    nu = mu * np.exp(beta*x)
    nu = nu/np.sum(nu)
    if (max(x) - min(x))/(np.linalg.norm(x) + 1e-5) < 1e-5:
        return mu
    while np.sum(kl_div(nu, mu)) < epsilon:# and beta < max(1000/np.linalg.norm(x), 100*np.linalg.norm(x)):
        try:
            beta0 = beta
            beta = beta * 2
            nu = mu * np.exp(beta * x)
            nu = nu / np.sum(nu)
        except:
            nu = mu * np.exp(beta0 * x)
            nu = nu / np.sum(nu)
            return nu
    count = 0
    while True:
        count +=1
        alpha = (beta+beta0)/2
        nu = mu * np.exp(alpha * x)
        nu = nu / np.sum(nu)
        if np.sum(kl_div(nu,mu)) > epsilon:
            beta = alpha
        else:
            beta0 = alpha
        if np.abs(np.sum(kl_div(nu, mu)) - epsilon) < 1e-10:
            break
        if count > 1e5:
            pass
    return nu



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
# # Test sup_inf_solver
#     x = np.random.randn(100)
#     mu = np.random.rand(100)
#     mu = mu/np.sum(mu)
#     epsilon = 0.01
#     nu = sup_inf_solver(x, mu, epsilon)
#     print(nu)
#     print(np.sum(nu))
#     print(np.sum(kl_div(nu,mu))-epsilon)
#############################################################################
    lst = np.array([0,-10,5,-10,0,1,1,0, 0,0,-1,2,-1,0])
    l = len(lst)
    H = 100
    epsilon0 = 0.15
    V,pi = DP(lst,epsilon0,H)
    rho = np.ones(l)/l

    r_lst = []
    delta_lst = [1e-5,1e-3, 1e-2] + [.03*(i+1) for i in range(10)]
    for delta in delta_lst:
        r_lst.append(policy_eval_robust(lst, epsilon0, H, pi, delta, rho))
        print(r_lst)
    from matplotlib import pyplot as plt
    plt.plot(delta_lst, r_lst)

    V,pi = DP_exp(lst,epsilon0,H,beta=-.5)
    r_lst_exp = []
    for delta in delta_lst:
        r_lst_exp.append(policy_eval_robust(lst, epsilon0, H, pi, delta, rho))
        print(r_lst_exp)
    plt.plot(delta_lst, r_lst_exp)

    V,pi = DP_exp(lst,epsilon0,H,beta=.2)
    r_lst_exp = []
    for delta in delta_lst:
        r_lst_exp.append(policy_eval_robust(lst, epsilon0, H, pi, delta, rho))
        print(r_lst_exp)
    plt.plot(delta_lst, r_lst_exp)

    plt.legend(['classic', 'exp-beta<0', 'exp-beta>0'])
    plt.grid(visible=True)

    plt.show()
#############################################################################
    # lst = np.array([0,-10,5,-10,0,1,1,0, 0,0,-1,2,-1,0])
    # l = len(lst)
    # H = 100
    # epsilon0 = 0.15
    # V,pi = DP(lst,epsilon0,H)
    # rho = np.ones(l)/l
    #
    # r_lst = []
    # epsilon_lst = [0.1,0.15,0.2,0.25,0.3]
    # for epsilon in epsilon_lst:
    #     r_lst.append(policy_eval(lst, epsilon, H, pi, rho))
    # from matplotlib import pyplot as plt
    # plt.plot(epsilon_lst, r_lst)
    #
    # V,pi = DP_exp(lst,epsilon0,H,beta=.2)
    # r_lst_exp = []
    # for epsilon in epsilon_lst:
    #     r_lst_exp.append(policy_eval(lst, epsilon, H, pi, rho))
    # plt.plot(epsilon_lst, r_lst_exp)
    #
    # V, pi = DP_exp(lst, epsilon0, H, beta=-.5)
    # r_lst_exp = []
    # for epsilon in epsilon_lst:
    #     r_lst_exp.append(policy_eval(lst, epsilon, H, pi, rho))
    # plt.plot(epsilon_lst, r_lst_exp)
    # plt.legend(['classic', 'exp-beta>0', 'exp-beta<0'])
    # plt.grid(visible=True)
    #
    # plt.show()
