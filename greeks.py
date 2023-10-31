from scipy.stats import norm
import numpy as np

def d1(S, K, T, r, sigma):
    return (np.log(S/K) + (r + sigma**2/2)*T) /\
                     sigma*np.sqrt(T)

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma* np.sqrt(T)

def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T)* norm.cdf(d2)

def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def delta_call(S, K, T, r, sigma):
    return norm.cdf(d1(S, K, T, r, sigma))
    
def delta_fdm_call(S, K, T, r, sigma, ds = 1e-5, method='central'):
    method = method.lower() 
    if method =='central':
        return (BS_CALL(S+ds, K, T, r, sigma) -BS_CALL(S-ds, K, T, r, sigma))/\
                        (2*ds)
    elif method == 'forward':
        return (BS_CALL(S+ds, K, T, r, sigma) - BS_CALL(S, K, T, r, sigma))/ds
    elif method == 'backward':
        return (BS_CALL(S, K, T, r, sigma) - BS_CALL(S-ds, K, T, r, sigma))/ds
    
    
def delta_put(S, K, T, r, sigma):
    return - norm.cdf(-d1(S, K, T, r, sigma))

def delta_fdm_put(S, K, T, r, sigma, ds = 1e-5, method='central'):
    method = method.lower() 
    if method =='central':
        return (BS_PUT(S+ds, K, T, r, sigma) -BS_PUT(S-ds, K, T, r, sigma))/\
                        (2*ds)
    elif method == 'forward':
        return (BS_PUT(S+ds, K, T, r, sigma) - BS_PUT(S, K, T, r, sigma))/ds
    elif method == 'backward':
        return (BS_PUT(S, K, T, r, sigma) - BS_PUT(S-ds, K, T, r, sigma))/ds

def gamma(S, K, T, r, sigma):
    N_prime = norm.pdf
    return N_prime(d1(S,K, T, r, sigma))/(S*sigma*np.sqrt(T))


def gamma_fdm(S, K, T, r, sigma , ds = 1e-5, method='central'):
    method = method.lower() 
    if method =='central':
        return (BS_CALL(S+ds , K, T, r, sigma) -2*BS_CALL(S, K, T, r, sigma) + 
                    BS_CALL(S-ds , K, T, r, sigma) )/ (ds)**2
    elif method == 'forward':
        return (BS_CALL(S+2*ds, K, T, r, sigma) - 2*BS_CALL(S+ds, K, T, r, sigma)+
                   BS_CALL(S, K, T, r, sigma) )/ (ds**2)
    elif method == 'backward':
        return (BS_CALL(S, K, T, r, sigma) - 2* BS_CALL(S-ds, K, T, r, sigma)
                + BS_CALL(S-2*ds, K, T, r, sigma)) /  (ds**2)  

def vega(S, K, T, r, sigma):
    N_prime = norm.pdf
    return S*np.sqrt(T)*N_prime(d1(S,K,T,r,sigma)) 

def vega_fdm(S, K, T, r, sigma, dv=1e-4, method='central'):
    method = method.lower() 
    if method =='central':
        return (BS_CALL(S, K, T, r, sigma+dv) -BS_CALL(S, K, T, r, sigma-dv))/\
                        (2*dv)
    elif method == 'forward':
        return (BS_CALL(S, K, T, r, sigma+dv) - BS_CALL(S, K, T, r, sigma))/dv
    elif method == 'backward':
        return (BS_CALL(S, K, T, r, sigma) - BS_CALL(S, K, T, r, sigma-dv))/dv
 
def theta_call(S, K, T, r, sigma):
    p1 = - S*norm.pdf(d1(S, K, T, r, sigma))*sigma / (2 * np.sqrt(T))
    p2 = r*K*np.exp(-r*T)*norm.cdf(d2(S, K, T, r, sigma)) 
    return p1 - p2

def theta_put(S, K, T, r, sigma):
    p1 = - S*norm.pdf(d1(S, K, T, r, sigma))*sigma / (2 * np.sqrt(T))
    p2 = r*K*np.exp(-r*T)*norm.cdf(-d2(S, K, T, r, sigma)) 
    return p1 + p2

def theta_call_fdm(S, K, T, r, sigma, dt, method='central'):
    method = method.lower() 
    if method =='central':
        return -(BS_CALL(S, K, T+dt, r, sigma) -BS_CALL(S, K, T-dt, r, sigma))/\
                        (2*dt)
    elif method == 'forward':
        return -(BS_CALL(S, K, T+dt, r, sigma) - BS_CALL(S, K, T, r, sigma))/dt
    elif method == 'backward':
        return -(BS_CALL(S, K, T, r, sigma) - BS_CALL(S, K, T-dt, r, sigma))/dt
    
def theta_put_fdm(S, K, T, r, sigma, dt, method='central'):
    method = method.lower() 
    if method =='central':
        return -(BS_PUT(S, K, T+dt, r, sigma) -BS_PUT(S, K, T-dt, r, sigma))/\
                        (2*dt)
    elif method == 'forward':
        return -(BS_PUT(S, K, T+dt, r, sigma) - BS_PUT(S, K, T, r, sigma))/dt
    elif method == 'backward':
        return -(BS_PUT(S, K, T, r, sigma) - BS_PUT(S, K, T-dt, r, sigma))/dt

def rho_call(S, K, T, r, sigma):
    return K*T*np.exp(-r*T)*norm.cdf(d2(S, K, T, r, sigma))

def rho_put(S, K, T, r, sigma):
    return -K*T*np.exp(-r*T)*norm.cdf(-d2(S, K, T, r, sigma))


def rho_call_fdm(S, K, T, r, sigma, dr, method='central'):
    method = method.lower() 
    if method =='central':
        return (BS_CALL(S, K, T, r+dr, sigma) -BS_CALL(S, K, T, r-dr, sigma))/\
                        (2*dr)
    elif method == 'forward':
        return (BS_CALL(S, K, T, r+dr, sigma) - BS_CALL(S, K, T, r, sigma))/dr
    elif method == 'backward':
        return (BS_CALL(S, K, T, r, sigma) - BS_CALL(S, K, T, r-dr, sigma))/dr
  
def rho_put_fdm(S, K, T, r, sigma, dr, method='central'):
    method = method.lower() 
    if method =='central':
        return (BS_PUT(S, K, T, r+dr, sigma) -BS_PUT(S, K, T, r-dr, sigma))/\
                        (2*dr)
    elif method == 'forward':
        return (BS_PUT(S, K, T, r+dr, sigma) - BS_PUT(S, K, T, r, sigma))/dr
    elif method == 'backward':
        return (BS_PUT(S, K, T, r, sigma) - BS_PUT(S, K, T, r-dr, sigma))/dr