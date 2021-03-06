import numpy as np
from collections import defaultdict

def count_to_dirichlet(dictionnaire):
    keys,values=[],[]
    for key,value in dictionnaire.items():
        keys.append(key)
        values.append(value)
    results=np.random.dirichlet(values)
    return {keys[i]:results[i] for i in range(len(dictionnaire))}



class BEBLP_Agent:

    def __init__(self,environment, gamma=0.95, beta=1,step_update=10,alpha=0.5,coeff_prior=0.5):
        
        self.environment=environment
        self.gamma = gamma
        self.beta=beta
        
        self.R = defaultdict(lambda: defaultdict(lambda: 0.0)) #Récompenses
        self.tSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0))) #Transitions
        
        self.nSA = defaultdict(lambda: defaultdict(lambda: 0)) #Compteur passage (état,action)
        self.nSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0))) #Compteur (état_1,action,état_2)
        self.last_k=defaultdict(lambda: defaultdict(lambda: [(0,0)]*self.step_update))
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0)) #Q-valeur état-action
        self.bonus=defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.counter=self.nSA #Compteur (état,action)
        self.step_counter=0
        
        self.prior= defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        self.prior_0=defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.tSAS_old=defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        self.nSAS_old = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.LP=defaultdict(lambda: defaultdict(lambda: 0.0))
        self.step_update=step_update
        self.alpha=alpha
        
        self.coeff_prior=coeff_prior
        self.known_state_action=[]
        self.ajout_states()
        
    def learn(self,old_state,reward,new_state,action):
                                        
        self.nSA[old_state][action] +=1
        self.nSAS[old_state][action][new_state] += 1
        self.R[old_state][action]=reward
        self.prior[old_state][action][new_state]+=1  
        
        #Modifier les probabilités de transition selon le prior avec distribution de dirichlet
        self.tSAS[old_state][action]=count_to_dirichlet(self.prior[old_state][action])
        self.last_k[old_state][action][self.nSA[old_state][action]%self.step_update]=new_state
        #Ajout du bonus qui dépend du nombre de passages

        """if self.nSA[old_state][action]<self.step_update:
            new_CV,new_variance=self.cross_validation(self.nSAS[old_state][action],self.prior[old_state][action])
            self.LP[old_state][action]=max(new_CV+self.alpha*np.sqrt(new_variance),0.001)
            self.bonus[old_state][action]=self.beta/(1+1/np.sqrt(self.LP[old_state][action]))"""
        
        
        if self.nSA[old_state][action]>self.step_update:
            new_dict_nSAS,new_dict_prior={k:v for k,v in self.nSAS[old_state][action].items()},{k:v for k,v in self.prior[old_state][action].items()}       
            for last_seen_state in self.last_k[old_state][action]:
                new_dict_nSAS[last_seen_state]-=1
                new_dict_prior[last_seen_state]-=1
                if new_dict_nSAS[last_seen_state]==0:
                    del new_dict_nSAS[last_seen_state]
            new_CV,new_variance=self.cross_validation(self.nSAS[old_state][action],self.prior[old_state][action])
            old_CV,old_variance=self.cross_validation(new_dict_nSAS,new_dict_prior)
            #print(old_CV,new_CV,new_variance)
            self.LP[old_state][action]=max(old_CV-new_CV+self.alpha*np.sqrt(new_variance),0.001)
            #print(self.LP[old_state][action])
            self.bonus[old_state][action]=self.beta/(1+1/np.sqrt(self.LP[old_state][action]))
        
        for i in range(10):
            for state_known in self.nSAS:
                for action_known in self.nSAS[state_known]:
                    self.Q[state_known][action_known]=self.R[state_known][action_known]+self.bonus[state_known][action_known]+self.gamma*np.sum([max(self.Q[next_state].values())*self.tSAS[state_known][action_known][next_state] for next_state in self.tSAS[state_known][action_known].keys()])

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
                    self.tSAS[state_1][action][state_2]=1/number_states
                self.Q[state_1][action]=(1+self.beta)/(1-self.gamma)
                self.LP[state_1][action]=np.log(number_states)
                self.bonus[state_1][action]=self.beta/(1+1/np.sqrt(self.LP[state_1][action]))
                
    def cross_validation(self,nSAS_SA,prior_SA): 
        """cv,v=0,[]
        for next_state,next_state_count in nSAS_SA.items():
            prior_SA[next_state]-=1
            for i in range(next_state_count):
                values=count_to_dirichlet(prior_SA)
                if values[next_state]<1e-10:
                    values[next_state]=1e-10
                cv-=np.log(values[next_state])
                v.append(-np.log(values[next_state]))
            prior_SA[next_state]+=1
        v=np.array(v)
        cardinal=sum(nSAS_SA.values())
        cross_val =cv/cardinal
        v=(v-cross_val)**2
        variance_cv=np.sum(v)/cardinal
        return cross_val,max(variance_cv,1)"""
    
        cv,v=0,[]
        sum_prior=sum(prior_SA.values())
        for next_state,next_state_count in nSAS_SA.items():            
            value=(prior_SA[next_state]-1)/(sum_prior-1)
            cv-=np.log(value)*next_state_count
            v+=[-np.log(value)]*int(next_state_count)
            prior_SA[next_state]+=1
        v=np.array(v)
        cardinal=sum(nSAS_SA.values())
        cross_val =cv/cardinal
        v=(v-cross_val)**2
        variance_cv=np.sum(v)/cardinal
        return cross_val,variance_cv
    
        """ cv,v=0,[]
        for next_state,next_state_count in prior.items():
            if next_state_count>1:
                value=(next_state_count-1)/sum(prior.values())
                if value ==0: log_value=-2
                else: log_value=np.log(value)
                cv-=next_state_count*log_value
                v+=[-log_value]*int(next_state_count)
        v=np.array(v)
        cardinal=sum(nSAS_SA.values())
        cross_validation =cv/cardinal
        var=(v-cross_validation)**2
        variance_cv=np.sum(var)/cardinal
        return cross_validation,variance_cv
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
        return cross_validation,variance_cv"""
        
            
