import numpy as np
from collections import defaultdict

def vitesse(gamma=0.95):
     
     gamma = gamma  
     R = defaultdict(lambda: defaultdict(lambda: 0.0))
     tSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))        
     Q = defaultdict(lambda: defaultdict(lambda: 0.0))
     for (i,j,k,l,m) in [(a,b,c,d,e) for a in range(5) for b in range(5) for c in range(5) for d in range(5) for e in range(5)]:
         tSAS[i,j][k][l,m]=np.random.random()
         Q[i,j][k]=np.random.random()
         R[i,j][k]=np.random.random()
     for z in range(5):
        for visited_state in [(i,j) for i in range(5) for j in range(5)]:
                 for taken_action in range(5):
                     Q[visited_state][taken_action]=R[visited_state][taken_action]+gamma*np.sum([max(Q[next_state].values())*tSAS[visited_state][taken_action][next_state] for next_state in tSAS[visited_state][taken_action]])
             
vitesse()

"""
def vitesse_2(gamma=0.95):
    R=np.random.random((5,5,5))
    tSAS=np.random.random((5,5,5,5,5))
    Q=np.random.random((5,5,5,5,5))
    for z in range(50):
        Q[:][:][:]=R+gamma*np.sum([max(Q[next_state])*tSAS[:][:][:][next_state] for next_state in tSAS[:][:][:]])

vitesse_2()"""