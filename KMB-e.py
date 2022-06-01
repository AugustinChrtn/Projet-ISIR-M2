import numpy as np
from collections import defaultdict

class KMBE_Agent:

    def __init__(self,environment, gamma=0.95,H_update=3,gamma_epis=0.5,epis_factor=10,variance_ob=1,variance_tr=1):
        
        self.environment=environment
        self.gamma = gamma
        self.gamma_epis=gamma_epis
        self.variance_ob=variance_ob
        self.variance_tr=variance_tr
        
        
        
        self.R = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.tSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.nSA = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.nSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        self.Rsum = defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.K_var=defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.counter=self.nSA
        self.step_counter=0
        
        self.H=defaultdict(lambda: defaultdict(lambda: 0.0))
        self.tSAS_old=defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        self.H_update=H_update
        
        self.epis=defaultdict(lambda: defaultdict(lambda: 0.0))
        self.epis_factor=epis_factor
        self.ajout_states()
        
        
    def learn(self,old_state,reward,new_state,action):
                    
                    self.uncountered_state(new_state)
                    if self.nSA[old_state][action]==0:self.tSAS_old[old_state][action][new_state]=1
                    
                    self.nSA[old_state][action] +=1
                    self.Rsum[old_state][action] += reward
                    self.nSAS[old_state][action][new_state] += 1
                    self.R[old_state][action]=self.Rsum[old_state][action]/self.nSA[old_state][action]  

                    self.tSAS[old_state][action]=defaultdict(lambda:0.0)   
                    for next_state in self.nSAS[old_state][action].keys():
                        self.tSAS[old_state][action][next_state] = self.nSAS[old_state][action][next_state]/self.nSA[old_state][action]

                    current_mean = self.Q[old_state][action]
                    current_variance = self.K_var[old_state][action]
                    self.Q[old_state][action] = ((current_variance+self.variance_tr)*(self.R[old_state][action]+self.gamma*np.sum([max(self.Q[next_state].values())*self.tSAS[old_state][action][next_state] for next_state in self.tSAS[old_state][action].keys()]))
                                                                                      +(self.variance_ob*current_mean))/ (current_variance+self.variance_tr+self.variance_ob)
                    self.K_var[old_state][action]=((current_variance+self.variance_tr)*self.variance_ob)/(current_variance+self.variance_ob+self.variance_tr)
                    for move in self.environment.actions:
                        if move != action : 
                            self.K_var[old_state][move]+=self.variance_tr
                                    
                            
                    self.H[old_state][action]=self.entropy(self.tSAS[old_state][action])
                    if self.counter[old_state][action]%self.H_update==1:
                        self.KL[old_state][action]=self.KL_div(self.tSAS_old[old_state][action],self.tSAS[old_state][action])
                        self.tSAS_old[old_state][action] = self.tSAS[old_state][action]
                    
                   
                    for i in range(10):
                        for state_known in self.nSA:
                                for action_known in self.nSA[state_known]:
                                    self.Q[state_known][action_known]=self.R[state_known][action_known]+self.gamma*np.sum([max(self.Q[next_state].values())*self.tSAS[state_known][action_known][next_state] for next_state in self.tSAS[state_known][action_known].keys()])
                                    self.epis[state_known][action_known] = self.H[state_known][action_known]+self.gamma_epis *np.sum([max(self.H[next_state].values())*self.tSAS[state_known][action_known][next_state] for next_state in self.tSAS[state_known][action_known].keys()])
    def choose_action(self):
        self.step_counter+=1
        state=self.environment.current_location
         
        self.uncountered_state(state)
        
        q_values = self.Q[state]
        epis_values=self.epis[state]
        var=self.K_var[state]
        total_dict={}
        for key in q_values.keys():
            total_dict[key]=self.epis_factor*(epis_values[key]+var[key])+q_values[key]
        maxValue = max(total_dict.values())
        action = np.random.choice([k for k, v in total_dict.items() if v == maxValue])
        return action
    
                
    def ajout_states(self):
        self.states=self.environment.states
        for state_1 in self.states:
            for action in self.environment.actions:
                self.R[state_1][action]=0
                self.Q[state_1][action]=1/(1-self.gamma)
                self.K_var[state_1][action]=1
                self.H[state_1][action]=0
                self.KL[state_1][action]=0 
                self.epis[state_1][action]=0
                for state_2 in self.states:
                    self.tSAS[state_1][action][state_2]=1/len(self.states)

    def entropy(self,transitions):
        value_entropy=0
        for value in transitions.values():
            if value>0:
                value_entropy+=-value*np.log2(value)
        return value_entropy
    