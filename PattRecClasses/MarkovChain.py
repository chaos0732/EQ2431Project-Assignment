import numpy as np
from .DiscreteD import DiscreteD

class MarkovChain:
    """
    MarkovChain - class for first-order discrete Markov chain,
    representing discrete random sequence of integer "state" numbers.
    
    A Markov state sequence S(t), t=1..T
    is determined by fixed initial probabilities P[S(1)=j], and
    fixed transition probabilities P[S(t) | S(t-1)]
    
    A Markov chain with FINITE duration has a special END state,
    coded as nStates+1.
    The sequence generation stops at S(T), if S(T+1)=(nStates+1)
    """
    def __init__(self, initial_prob, transition_prob):

        self.q = initial_prob  #InitialProb(i)= P[S(1) = i]
        self.A = transition_prob #TransitionProb(i,j)= P[S(t)=j | S(t-1)=i]


        self.nStates = transition_prob.shape[0]

        self.is_finite = False
        if self.A.shape[0] != self.A.shape[1]:
            self.is_finite = True


    def probDuration(self, tmax):
        """
        Probability mass of durations t=1...tMax, for a Markov Chain.
        Meaningful result only for finite-duration Markov Chain,
        as pD(:)== 0 for infinite-duration Markov Chain.
        
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.8.
        """
        pD = np.zeros(tmax)

        if self.is_finite:
            pSt = (np.eye(self.nStates)-self.A.T)@self.q

            for t in range(tmax):
                pD[t] = np.sum(pSt)
                pSt = self.A.T@pSt

        return pD

    def probStateDuration(self, tmax):
        """
        Probability mass of state durations P[D=t], for t=1...tMax
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.7.
        """
        t = np.arange(tmax).reshape(1, -1)
        aii = np.diag(self.A).reshape(-1, 1)
        
        logpD = np.log(aii)*t+ np.log(1-aii)
        pD = np.exp(logpD)

        return pD

    def meanStateDuration(self):
        """
        Expected value of number of time samples spent in each state
        """
        return 1/(1-np.diag(self.A))
    
    def rand(self, tmax):
        """
        S=rand(self, tmax) returns a random state sequence from given MarkovChain object.
        
        Input:
        tmax= scalar defining maximum length of desired state sequence.
           An infinite-duration MarkovChain always generates sequence of length=tmax
           A finite-duration MarkovChain may return shorter sequence,
           if END state was reached before tmax samples.
        
        Result:
        S= integer row vector with random state sequence,
           NOT INCLUDING the END state,
           even if encountered within tmax samples
        If mc has INFINITE duration,
           length(S) == tmax
        If mc has FINITE duration,
           length(S) <= tmaxs
        """
        
        S = []
        
        randNumGenerator = DiscreteD(self.q)
        currentState = int(randNumGenerator.rand(1)[0])-1
        S.append(currentState)

        for t in range(1, tmax):
            randNumGenerator = DiscreteD(self.A[currentState])
            currentState = int(randNumGenerator.rand(1)[0])-1

            if currentState == self.nStates:
                break
            else:
                S.append(currentState)

        S_array = np.array(S)
        return S_array+1

    def viterbi(self):
        pass
    
    def stationaryProb(self):
        pass
    
    def stateEntropyRate(self):
        pass
    
    def setStationary(self):
        pass

    def logprob(self):
        pass

    def join(self):
        pass

    def initLeftRight(self):
        pass
    
    def initErgodic(self):
        pass

    def forward(self, pX):
        """
        alpha_hat, c = forward(self, pX)
        Forward algorithm for Markov Chain.
        Input:
        pX= matrix of size (nStates, nSamples)
        pX(i,j)= P[X(j) | S(t)=i]

        Result:
        alpha_hat= matrix of size (nStates, nSamples)
        alpha_hat(i,j)= P[S(t)=i | X(1:j)]
        c= vector of size (nSamples)
        c(j)= P[X(1:j)]
        """
        
        # Initialize
        nSamples = pX.shape[1]
        alpha_hat = np.zeros((self.nStates, nSamples))
        c = np.zeros(nSamples)
        if self.is_finite:
            A_square = self.A[:,:-1]  # Exclude the last column (END state)
        else:
            A_square = self.A

        #t =1  
        alpha_temp = self.q * pX[:, 0]
        c[0] = np.sum(alpha_temp)
        alpha_hat[:, 0] = alpha_temp / c[0]

        #t = 2..nSamples
        for t in range(1, nSamples):
            alpha_temp = A_square.T @ alpha_hat[:, t-1] * pX[:, t]
            c[t] = np.sum(alpha_temp)
            alpha_hat[:, t] = alpha_temp / c[t] 
        
        # termination
        if self.is_finite:
            c= np.append(c, alpha_hat[:,-1] @ self.A[:,-1])
        
        #c = c/c[0] #normalize?
        return alpha_hat, c

    def finiteDuration(self):
        pass
    
    def backward(self):
        pass

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass
