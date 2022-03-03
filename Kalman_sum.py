import numpy as np

def create_matrix(width,height,liste):
    return [[liste for i in range(width)] for j in range(height)]

UP,DOWN,LEFT,RIGHT,STAY=0,1,2,3,4


class Kalman_agent_sum(): 
    
    def __init__(self, environment, gamma=1, variance_ob=1,variance_tr=40,curiosity_factor=2):
        self.environment = environment
        self.KF_table_mean =np.array(create_matrix(environment.width, environment.height,[0.,0.,0.,0.,0.])) 
        self.KF_table_variance = np.array(create_matrix(environment.width, environment.height,[1.,1.,1.,1.,1.])) 
        self.counter=np.array(create_matrix(environment.width, environment.height,[0.,0.,0.,0.,0.])) 
        self.gamma = gamma
        self.variance_ob=variance_ob
        self.variance_tr=variance_tr
        self.curiosity_factor=curiosity_factor

    def choose_action(self):
        get_mean=self.KF_table_mean[self.environment.current_location]
        get_variance=self.KF_table_variance[self.environment.current_location]
        probas=np.array([0,0,0,0,0])
        for move in [UP,DOWN,LEFT,RIGHT,STAY]:
            probas[move]=np.random.normal(get_mean[move],np.sqrt(get_variance[move]))+(self.curiosity_factor*get_variance[move])
        action = np.argmax(probas)
        self.counter[self.environment.current_location][action]+=1
        return action
    
    def learn(self, old_state, reward, new_state, action):
        means_new_state = self.KF_table_mean[new_state]
        max_mean_in_new_state = np.max(means_new_state)
        current_mean = self.KF_table_mean[old_state][action]
        current_variance = self.KF_table_variance[old_state][action]
        self.KF_table_mean[old_state][action] = ((current_variance+self.variance_tr)*(reward +self.gamma * max_mean_in_new_state)+(self.variance_ob*current_mean))/ (current_variance+self.variance_tr+self.variance_ob)
        self.KF_table_variance[old_state][action]=((current_variance+self.variance_tr)*self.variance_ob)/(current_variance+self.variance_ob+self.variance_tr)
        for move in [UP,DOWN,LEFT,RIGHT,STAY]:
            if move != action : 
                self.KF_table_variance[old_state][move]+=self.variance_tr
