import numpy as np
from collections import defaultdict


class RmaxLP_Agent:

    def __init__(self,environment, gamma=0.95,Rmax=200,step_update=10,alpha=0.1,m=1,VI=50):
        
        self.Rmax=Rmax
        
        self.environment=environment
        self.gamma = gamma  
        
        self.R = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.Rsum = defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.nSA = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.nSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.tSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.counter=self.nSA
        self.step_counter=0
        
        self.LP=defaultdict(lambda: defaultdict(lambda: 0.0))
        self.CV=defaultdict(lambda: defaultdict(lambda: 0.0))
        self.step_update=step_update
        self.alpha=alpha
        self.m=m
        self.VI=VI
        self.known_state_action=[]
        self.ajout_states()
        
    def learn(self,old_state,reward,new_state,action):
                    
                    
                    self.nSA[old_state][action] +=1
                    self.Rsum[old_state][action] += reward
                    self.nSAS[old_state][action][new_state] += 1
                    
                    if self.LP[old_state][action] < self.m :
                        if (old_state,action) not in self.known_state_action :
                            self.R[old_state][action]=self.Rsum[old_state][action]/self.nSA[old_state][action] 
                            self.known_state_action.append((old_state,action))
                            self.tSAS[old_state][action]=defaultdict(lambda:.0)
                            self.R[old_state][action]=self.Rsum[old_state][action]/self.nSA[old_state][action]                         
                            for next_state in self.nSAS[old_state][action].keys():
                                self.tSAS[old_state][action][next_state] = self.nSAS[old_state][action][next_state]/self.nSA[old_state][action]
                            for j in range(self.VI): #cf formule logarithme Strehl 2009 PAC Analysis
                                for state_known,action_known in self.known_state_action:
                                    self.Q[state_known][action_known]=self.R[state_known][action_known]+self.gamma*np.sum([max(self.Q[next_state].values())*self.tSAS[state_known][action_known][next_state] for next_state in self.tSAS[state_known][action_known].keys()])
                    else : 
                        self.R[old_state][action]=self.Rmax      
                        if (old_state,action) in self.known_state_action: self.known_state_action.remove((old_state,action))
                    
                    if self.nSA[old_state][action]%self.step_update==0:                    
                        new_CV,new_variance=self.cross_validation(self.nSAS[old_state][action])
                        self.LP[old_state][action]=max(self.CV[old_state][action]-new_CV+self.alpha*np.sqrt(new_variance),0.001)
                        self.CV[old_state][action]=new_CV
                        
                        
    def choose_action(self):
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
                self.tSAS[state_1][action]=1
                self.R[state_1][action]=self.Rmax
                self.Q[state_1][action]=self.Rmax/(1-self.gamma)
                self.CV[state_1][action]=np.log(number_states)
                self.LP[state_1][action]=np.log(number_states)
    
    def cross_validation(self,nSAS_SA):
        cv,v=0,[]
        for key,value in nSAS_SA.items():
            for i in range(int(value)):
                keys,values=[],[]
                for next_state, next_state_count in nSAS_SA.items():
                        keys.append(next_state)
                        values.append(next_state_count)
                        if next_state==key:
                            values[-1]-=1
                for j in range(len(keys)):
                    if keys[j]==key:
                        if values[j]==0:
                            cv+=1 #np.log(1/sum(nSAS_SA.values())) ?
                            v.append(1) #np.log(1/sum(nSAS_SA.values())) ?
                        else :
                            cv+=np.log(values[j])
                            v.append(np.log(values[j]))
        v=np.array(v)
        cardinal=sum(nSAS_SA.values())
        cross_validation =1/cardinal*cv
        v=(v-cross_validation)**2
        variance_cv=1/cardinal*np.sum(v)
        return cross_validation,variance_cv