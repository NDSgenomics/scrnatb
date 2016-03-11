import numpy as np

def T_classic(data, sigma=1000.):
    ''' Implementation of diffusion transition matrix calculation
    from Haghverdi et al http://biorxiv.org/content/early/2016/03/02/041384
    '''
    n, G = data.shape
    d2 = np.zeros((n, n))

    for g in range(G):
        d2_g = np.subtract.outer(data[:, g], data[:, g]) ** 2
        d2 += d2_g

    W = np.exp(-d2 / (2. * sigma ** 2))

    D = W.sum(0)

    q = np.multiply.outer(D, D)
    np.fill_diagonal(W, 0.)

    H = W / q

    D_ = np.diag(H.sum(1))
    D_inv = np.diag(1. / np.diag(D_))

    T = D_inv.dot(H)

    phi0 = np.diag(D_) / np.diag(D_).sum()
    
    return T, phi0


def dpt_input(T, phi0):
    ''' Implementation of accumulated transition matrix calculation
    from Haghverdi et al http://biorxiv.org/content/early/2016/03/02/041384
    '''

    n = T.shape[0]
    M = np.linalg.inv(np.eye(n) - T + np.dot(np.ones((n, 1)) / n, phi0[None, :])) - np.eye(n)
    
    return M


def dpt_to_root(M, phi0, s):
    ''' Implementation of propagation distance calculation
    from Haghverdi et al http://biorxiv.org/content/early/2016/03/02/041384
    
    s is the index of the root point
    
    The propagation distance to every other point is returned. This should
    estimate the geodesic distance along the manifold defined by the data.
    '''
    n = M.shape[0]
    dpt = np.zeros(n)

    for x in range(n):
        D = (M[s, :] - M[x, :]) ** 2 / np.abs(phi0)
        dpt[x] = np.sqrt(D.sum())
        
    return dpt
