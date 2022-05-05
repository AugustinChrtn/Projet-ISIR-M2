import numpy as np
from collections import defaultdict


class RmaxLP_Agent:

    def __init__(self,environment, gamma=0.95,Rmax=200,step_update=10,alpha=0.1,m=1,VI=50):
        
        self.Rmax=Rmax
        
        self.environment=environment
        self.gamma = gamma  
        
        self.R = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.Rsum = defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.nSA = defaultdict(lambda: defaultdict(lambda: 0))
        self.nSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0)))
        self.last_k=defaultdict(lambda: defaultdict(lambda: [(0,0)]*self.step_update))
        
        self.tSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.counter=self.nSA
        self.step_counter=0
        
        self.LP=defaultdict(lambda: defaultdict(lambda: 0.0))

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
                    self.last_k[old_state][action][self.nSA[old_state][action]%self.step_update]=new_state
                    
                    
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
                    
                    
                    if self.nSA[old_state][action]>self.step_update:
                        new_dict={}
                        for k,v in self.nSAS[old_state][action].items():
                            new_dict[k]=v
                        for last_seen_state in self.last_k[old_state][action]:
                            new_dict[last_seen_state]-=1
                            if new_dict[last_seen_state]==0:
                                del new_dict[last_seen_state]
                        new_CV,new_variance=self.cross_validation(self.nSAS[old_state][action])
                        old_CV,old_variance=self.cross_validation(new_dict)
                        self.LP[old_state][action]=max(old_CV-new_CV+self.alpha*np.sqrt(new_variance),0.001)
                        
                        for j in range(3): #cf formule logarithme Strehl 2009 PAC Analysis
                            for state_known in self.nSAS.keys():
                                for action_known in self.nSAS[state_known].keys():
                                    self.Q[state_known][action_known]=self.R[state_known][action_known]+self.gamma*np.sum([max(self.Q[next_state].values())*self.tSAS[state_known][action_known][next_state] for next_state in self.tSAS[state_known][action_known].keys()])
                        
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
                self.LP[state_1][action]=np.log(number_states)
    
    def cross_validation(self,nSAS_SA):
        cv,v=0,[]
        for next_state,next_state_count in nSAS_SA.items():
            value=(next_state_count-1)/sum(nSAS_SA.values())
            if value ==0: log_value=-2
            else: log_value=np.log(value)
            cv-=next_state_count*log_value
            v+=[-log_value]*next_state_count
        v=np.array(v)
        cardinal=sum(nSAS_SA.values())
        cross_validation =cv/cardinal
        var=(v-cross_validation)**2
        variance_cv=np.sum(var)/cardinal
        return cross_validation,variance_cv