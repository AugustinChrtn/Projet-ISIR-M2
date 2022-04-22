import numpy as np
from collections import defaultdict

def count_to_dirichlet(dictionnaire):
    keys,values=[],[]
    for key,value in dictionnaire.items():
        keys.append(key)
        values.append(value)
    results=np.random.dirichlet(values)
    return {keys[i]:results[i] for i in range(len(dictionnaire))}



class BEBLP2_Agent:

    def __init__(self,environment, gamma=0.95, beta=1,step_update=10,alpha=0.5,coeff_prior=0.5,optimistic=1):
        
        self.environment=environment
        self.gamma = gamma
        self.beta=beta
        
        self.R = defaultdict(lambda: defaultdict(lambda: 0.0)) #Récompenses
        self.Rsum = defaultdict(lambda: defaultdict(lambda: 0.0)) #Somme récompenses accumulées 
        self.tSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0))) #Transitions
        
        self.nSA = defaultdict(lambda: defaultdict(lambda: 0.0)) #Compteur passage (état,action)
        self.nSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0))) #Compteur (état_1,action,état_2)

        
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0)) #Q-valeur état-action
        self.bonus=defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.counter=self.nSA #Compteur (état,action)
        self.step_counter=0
        
        self.prior= defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.tSAS_old=defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        self.nSAS_old = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.LP=defaultdict(lambda: defaultdict(lambda: 0.0))
        self.CV=defaultdict(lambda: defaultdict(lambda: 0.0))
        self.step_update=step_update
        self.alpha=alpha
        
        self.coeff_prior=coeff_prior
        self.optimistic=optimistic
        self.ajout_states()
        
    def learn(self,old_state,reward,new_state,action):
                    
                    
                    self.nSA[old_state][action] +=1
                    self.Rsum[old_state][action] += reward
                    self.nSAS[old_state][action][new_state] += 1
                    self.R[old_state][action]=self.Rsum[old_state][action]/self.nSA[old_state][action]
                    self.prior[old_state][action][new_state]+=1
                    #Modifier les probabilités de transition selon le compteur de passages avec distribution de dirichlet
                    self.tSAS[old_state][action]=count_to_dirichlet(self.prior[old_state][action])
                    
                    
                    if self.nSA[old_state][action]%self.step_update==0:                    
                        new_CV,new_variance=self.cross_validation(self.prior[old_state][action],self.nSAS[old_state][action])
                        self.LP[old_state][action]=max(self.CV[old_state][action]-new_CV+self.alpha*np.sqrt(new_variance),0.001)
                        self.CV[old_state][action]=new_CV
                        
                    self.bonus[old_state][action]=self.beta/(1+1/np.sqrt(self.LP[old_state][action]))
                    self.Q[old_state][action]=self.R[old_state][action]+self.bonus[old_state][action]+self.gamma*np.sum([max(self.Q[next_state].values())*self.tSAS[old_state][action][next_state] for next_state in self.tSAS[old_state][action].keys()])
                    
    def choose_action(self): #argmax pour choisir l'action
        self.step_counter+=1
        state=self.environment.current_location
        
        q_values = self.Q[state]
        maxValue = max(q_values.values())
        action = np.random.choice([k for k, v in q_values.items() if v == maxValue])
        return action
    
    
    def ajout_states(self):
        self.states=self.environment.states
        number_states=len(self.states)
        for state_1 in self.states:
            for action in self.environment.actions:
                for state_2 in self.states:
                    self.prior[state_1][action][state_2]=self.coeff_prior
                self.bonus[state_1][action]=self.beta
                self.Q[state_1][action]=self.optimistic*self.beta
                self.CV[state_1][action]=np.log(number_states)
                self.LP[state_1][action]=np.log(number_states)
                
    def cross_validation(self,prior,nSAS):
        cv=0
        v=[]
        for key,value in nSAS.items():
            for i in range(int(value)):
                keys,values=[],[]
                for next_state, next_state_count in prior.items():
                        keys.append(next_state)
                        values.append(next_state_count)
                        if next_state==key:
                            values[-1]-=1
                values=np.random.dirichlet(values)
                for j in range(len(keys)):
                    if keys[j]==key:
                        cv+=np.log(values[j])
                        v.append(np.log(values[j]))
        v=np.array(v)
        cardinal=sum(nSAS.values())
        cross_validation =-1/cardinal*cv
        v=(v-cross_validation)**2
        variance_cv=1/cardinal*np.sum(v)
        return cross_validation,variance_cv
        
            