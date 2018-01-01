import numpy as np


def sink(a, b, M, reg, numItermax=1000, stopThr=1e-9, verbose=False, log=False):

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    # init data
    Nini = len(a)
    Nfin = len(b)

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances

    u = np.ones(Nini) / Nini
    v = np.ones(Nfin) / Nfin

    # print(reg)

    K = np.exp(-M / reg)
    # print(np.min(K))

    Kp = (1 / a).reshape(-1, 1) * K
    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v
        KtransposeU = np.dot(K.T, u)
        v = np.divide(b, KtransposeU)
        u = 1. / np.dot(Kp, v)

        if (np.any(KtransposeU == 0) or
                np.any(np.isnan(u)) or np.any(np.isnan(v)) or
                np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            transp = u.reshape(-1, 1) * (K * v)
            err = np.linalg.norm((np.sum(transp, axis=0) - b))**2
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 1
    if log:
        log['u'] = u
        log['v'] = v

    if log:
        return np.sum(u.reshape((-1, 1)) * K * v.reshape((1, -1)) * M), log
    else:
        return np.sum(u.reshape((-1, 1)) * K * v.reshape((1, -1)) * M)




if __name__ == '__main__':
    a = [.5, .5]
    b = [.5, .5]
    M = [[0., 1.], [1., 0.]]
    out = sink(a, b, M, 1)
    print(out)