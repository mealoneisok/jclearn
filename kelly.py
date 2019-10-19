import numpy as np

class MultiKellyBettor:
    '''
    A Kelly calcalator that supports multiple exclusive outcomes.

    Parameters
    ----------
    odds : win odds of each competitor
    prob : winning probability of each competitor
    label : names of each competitor (optional)

    Output
    ------
    prop : a dictionary with the key being the competitor label and 
           the value being the proportion of the whole capital to bet on

    reference
    ---------
    1. https://www.sportsbookreview.com/picks/tools/kelly-calculator/
    2. https://www.sportsbookreview.com/forum/handicapper-think-tank/29624-simultaneous-event-kelly-calculator-beta.html
    3. https://www.sportsbookreview.com/forum/handicapper-think-tank/521569-simple-closed-form-solution-unconstrained-simultaneous-bet-kelly-staking.html

    '''
    def __init__(self, odds, prob, label = []):
        self.odds = np.array(odds)
        self.prob = np.array(prob)

        if len(self.odds.shape) != 1 or len(self.prob.shape) != 1:
            raise ValueError('The dimension of odds and prob must be 1.')
        if self.odds.shape != self.prob.shape:
            raise ValueError('The length of odds does not match the length of prob.')
        if np.sum(self.prob) - 1 > 1e6:
            raise ValueError('prob does not sum up to 1.')
        if np.any(self.odds <= 1):
            raise ValueError('odds must be all larger than 1.')

        if not label:
            label = list(range(len(odds)))
        elif len(odds) != len(label):
            raise ValueError('The length of label does not match.')
        self.label = label
        
    def transform(self):
        prop = {self.label[k] : 0 for k in range(len(self.odds))}
        eret = np.array([o * p for o, p in zip(self.odds, self.prob)])
        indices = np.argsort(eret)[::-1]
        odds = self.odds[indices]
        prob = self.prob[indices]
        eret = eret[indices]
        
        if eret[0] <= 1: #it means no edge so no betting
            return prop
        
        rodds = 1 / odds
        csprob = np.cumsum(prob)
        csrodds = np.cumsum(rodds)
        tmp = (1 - csprob) / (1 - csrodds)
        tmp = tmp[tmp > 0]
        v = tmp.min()
        bf = prob - v / odds
        bf = np.where(bf < 0, 0, bf)
        
        for k in range(len(tmp)):
            prop[self.label[indices[k]]] = bf[k]
        return prop

if __name__ == '__main__':
    mkb = MultiKellyBettor(odds = [1.87, 3.4, 3.4], 
                           prob = [0.592, 0.285, 0.123], 
                           label = ['horse1', 'horse2', 'horse3'])
    prop = mkb.transform()
    print(prop) 				#{'horse1': 0.207625, 'horse2': 0.07359374999999999, 'horse3': 0.0}
    print(sum(prop.values())) 	#total_prop : 0.28121874999999996
