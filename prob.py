import numpy as np
from itertools import permutations

class ProbProber:
    '''
    This class takes the win odds of each competitor and returns the winning probability of all
    different combinations according to a specific pool.
    The calculation is based on Harville formula and the implementation avoids too many iterations by
    using vectorization as much as possible.

    Parameters
    ----------
    wos : Win OddS of each competitor
    c : Correction coefficient. [1, 1, 1, 1] as default. This indicates the correction coefficient for
        the probabilities of the competitor ranking 1, 2, 3 and 4 respectively (denoted by c_1, c_2, c_3,
        c_4), used in the correction formular p_i'=(p_i^c)/(∑_j (p_j)^c) (note that if c = 1, the
        probability does not change). Based on maximum likelihood estimation, Benter suggests that c_2 =
        0.81 and c_3 = 0.65. 
    pool : Can be win, quinella, tierce, trio, place, place_q, quartet or first_4

    Output
    ------
    An array that stores the winning probability of all combinations.
    
    '''
    def __init__(self, wos, c = [1, 1, 1, 1]):
        self.wos = np.array(wos)
        self.c = c
        #order probabilities for specific competitor(s)
        #p_i means the probability for horse i ranks first
        #p_ij is the probability for horse i ranks first and horse j ranks second (i≠j) and so on. 
        self.p_i, self.p_ij, self.p_ijk, self.p_ijkl  = None, None, None, None 
        if len(c) < 4:
            c.extend([1] * (4 - len(c)))
        self.pools = ["win", "quinella", "tierce", "trio", "place", "place_q", "quartet", "first_4"]
            
    def _check_dim(self, dim, least_dim):
        if dim <= least_dim:
            raise ValueError('Not enough competitors, at least {} competitors for the following operation.'.format(least_dim))
        
    def _check_pool(self, pool):
        if pool not in self.pools:
            raise ValueError('Invalid pool: {}'.format(pool))
        
    def transform(self, pool = 'win'):
        self._check_pool(pool)
        dim = len(self.wos)
        
        if self.p_i is None:
            nppow = np.power(self.wos, self.c[0])
            self.p_i = nppow / nppow.sum()
        
        if pool == 'win':
            return self.p_i
        
        self._check_dim(dim, 2)
        
        if self.p_ij is None:
            nppow = np.power(self.wos, self.c[1])
            p_i2 =  nppow / nppow.sum()
            self.p_ij = p_i2.reshape(-1, 1) * (self.p_i / (1 - p_i2))
            self.p_ij[np.diag_indices(dim, ndim = 2)] = 0
            
        if pool == 'quinella':
            return self.p_ij + self.p_ij.T
        
        self._check_dim(dim, 3)

        if self.p_ijk is None: 
            nppow = np.power(self.wos, self.c[2])
            p_i3 =  nppow / nppow.sum()
            comp = np.tile(p_i3.reshape(-1, 1), dim)
            comp = 1 - comp - comp.T
            comp = np.divide(self.p_ij, comp, where = comp != 0)
            self.p_ijk = comp.reshape(*comp.shape, -1) * p_i3.reshape(1, dim)
            self.p_ijk.transpose([2, 1, 0])[np.diag_indices(dim, ndim = 2)] = 0
            self.p_ijk.transpose([2, 0, 1])[np.diag_indices(dim, ndim = 2)] = 0
        
        if pool == 'tierce':
            return self.p_ijk
        elif pool == 'trio':
            trio = 0
            for perm in permutations(range(3)):
                trio += self.p_ijk.transpose(perm)
            return trio
        elif pool == 'place':
            p_rk2 = self.p_ij.sum(axis = 1)
            p_rk3 = self.p_ijk.sum(axis = 1).sum(axis = 0)
            return self.p_i + p_rk2 + p_rk3
        elif pool == 'place_q':
            p_dot_i_j = self.p_ijk.sum(axis = 0)
            p_i_dot_j = self.p_ijk.sum(axis = 1)
            return self.p_ij + self.p_ij.T + \
                   p_i_dot_j + p_i_dot_j.T + \
                   p_dot_i_j + p_dot_i_j.T
        
        self._check_dim(dim, 4)
        
        if self.p_ijkl is None:
            p_i4 = np.power(self.wos, self.c[3]) / np.power(self.wos, self.c[3]).sum()
            comp = np.tile(p_i4, (dim, dim, 1))
            comp = 1 - comp - comp.T - comp.transpose([0, 2, 1])
            comp = np.divide(self.p_ijk, comp, where = comp != 0)
            self.p_ijkl = comp.reshape(*comp.shape, -1) * p_i4.reshape(1, dim)
            for i in range(dim):
                self.p_ijkl[i, :, :, i] = 0
                self.p_ijkl[:, i, :, i] = 0
                self.p_ijkl[:, :, i, i] = 0
                self.p_ijkl[i, :, i, :] = 0
                self.p_ijkl[:, i, i, :] = 0
                self.p_ijkl[i, i, :, :] = 0
        
        if pool == 'quartet':
            return self.p_ijkl
        elif pool == 'first_4':
            first_4 = 0
            for perm in permutations(range(4)):
                first_4 += self.p_ijkl.transpose(perm)
            return first_4
    
        #unreachable
        return None

if __name__ == '__main__':
    w_p = [0.1, 0.2, 0.2, 0.05, 0.05, 0.3, 0.1] # winning probabilities of horse1, 2, ..., 7                    
    pp = ProbProber(w_p) #winning probabilities and win odds are both ok

    place_p = ProbProber(w_p).transform(pool = 'place')
    print(place_p) # place_p = [0.34850021, 0.59536643, 0.59536643, 0.18543271, 0.18543271, 0.7414013 , 0.34850021]
    tierce_p = ProbProber(w_p).transform(pool = 'tierce')
    tierce_p126 = tierce_p[1, 2, 6]
    print(tierce_p126) #tierce_p126 = 0.008333333333333326, the proba of horse1, 2 and 6 are respectively in the 1st, 2nd, 3rd place