import numpy as np
from collections import defaultdict

class PEI_KL_Agent:

    def __init__(self,environment, gamma=0.95,gamma_kl=0.1,coeff_kl=1,epsilon=0.1,step_update=10):
        
        self.environment=environment
        self.gamma = gamma
        self.gamma_kl=gamma_kl
        self.step_update=step_update
        
        
        self.R = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.tSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))       
        self.nSA = defaultdict(lambda: defaultdict(lambda: 0))
        self.nSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0)))
        self.prior=defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))

        self.epsilon=epsilon
        
        
        self.counter=self.nSA
        self.step_counter=0
        
        
        self.KL=defaultdict(lambda: defaultdict(lambda: 0.0))
        self.epis=defaultdict(lambda: defaultdict(lambda: 0.0))
        self.coeff_kl=coeff_kl
        self.last_k=defaultdict(lambda: defaultdict(lambda: [(0,0)]*self.step_update))
        
        
        self.ajout_states()
        
        
    def learn(self,old_state,reward,new_state,action):
                    
                    
                    self.nSA[old_state][action] +=1
                    self.nSAS[old_state][action][new_state] += 1
                    self.R[old_state][action]=reward 
                    self.prior[old_state][action][new_state]+=1
                    
                    if self.nSA[old_state][action]==1:self.tSAS[old_state][action]=defaultdict(lambda:0.0)   
                    for next_state in self.prior[old_state][action].keys():
                        self.tSAS[old_state][action][next_state] = self.nSAS[old_state][action][next_state]/self.nSA[old_state][action]
                                                                                
                    
                    self.last_k[old_state][action][self.nSA[old_state][action]%self.step_update]=new_state
                   

                    if self.nSA[old_state][action]>self.step_update:
                        old_prior_dict={k:v for k,v in self.prior[old_state][action].items()}
                        for last_seen_state in self.last_k[old_state][action]:
                            old_prior_dict[last_seen_state]-=1
                        self.KL[old_state][action]=self.KL_div(self.prior[old_state][action],old_prior_dict)
                    
                    
                    delta=1
                    while delta > 1e-3 :
                        delta=0
                        for visited_state in self.nSA:
                            for taken_action in self.nSA[visited_state]:
                                value_action=self.Q[visited_state][taken_action]
                                self.Q[visited_state][taken_action]=self.R[visited_state][taken_action]+self.gamma*np.sum([max(self.Q[next_state].values())*self.tSAS[visited_state][taken_action][next_state] for next_state in self.tSAS[visited_state][taken_action]])
                                delta=max(delta,np.abs(value_action-self.Q[visited_state][taken_action]))   
                    delta=1
                    while delta > 1e-3 :
                        delta=0
                        for visited_state in self.nSA:
                            for taken_action in self.nSA[visited_state]:
                                value_epis=self.epis[visited_state][taken_action]
                                self.epis[visited_state][taken_action]=self.KL[visited_state][taken_action]+self.gamma_kl*np.sum([max(self.Q[next_state].values())*self.tSAS[visited_state][taken_action][next_state] for next_state in self.tSAS[visited_state][taken_action]])
                                delta=max(delta,np.abs(value_epis-self.epis[visited_state][taken_action]))  
                        

    def choose_action(self):
        self.step_counter+=1
        state=self.environment.current_location
        
        if np.random.random() > (1-self.epsilon) :
            action = np.random.choice(self.environment.actions)
        else:      
            q_values, epis_values = self.Q[state], self.epis[state]
            total_dict={k:q_values[k]+self.coeff_kl*epis_values[k] for k in q_values}
            maxValue = max(total_dict.values())
            action = np.random.choice([k for k, v in total_dict.items() if v == maxValue])
        return action
    
                
    def ajout_states(self):
        self.states=self.environment.states
        for state_1 in self.states:
            for action in self.environment.actions:
                self.R[state_1][action]=0
                self.Q[state_1][action]=1/(1-self.gamma)
                for state_2 in self.states:
                    self.tSAS[state_1][action][state_2]=1/len(self.states)
                    self.prior[state_1][action][state_2]=1/len(self.states)
                self.KL[state_1][action]=self.entropy(state_1,action)
                self.epis[state_1][action]=self.KL[state_1][action]/(1-self.gamma_kl)

    def entropy(self,state,action):
        values=np.array(list(self.prior[state][action].values()))/(self.nSA[state][action]+1)
        return np.sum(-values*np.log2(values))
    
    def KL_div(self,new_prior,old_prior):
        sum_new_prior=sum(new_prior.values())
        new_transi={k:v/sum_new_prior for k,v in new_prior.items()}
        old_transi={k:v/(sum_new_prior-self.step_update) for k,v in old_prior.items()}       
        value_KL=0
        for key,value in new_transi.items():
            value_KL+=value*np.log2(value/old_transi[key])
        return value_KL