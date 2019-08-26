from Code.evaluator import parameter_to_pipeline,X,y,get_index_of_zeros
from Code.setup import search_space,search_space_dic
from sklearn.model_selection import ShuffleSplit,cross_val_score
import random
from collections import Counter



class Store():
    def __init__(self):
        self.parameter = []
        self.score = []
        
    
    def add_one_record(self,parameter,score):
        self.parameter.append(parameter)
        self.score.append(score)
        
    
    def write_to_csv(self):
        parameter_score =  [[self.parameter[i],self.score[i]] for i in range(len(self.score)) ]
        df = pd.DataFrame(data=parameter_score,columns=['parameter','score'])
        df.to_csv('Storage/parameter_score_pair.csv',index=None)

class Game():
    def __init__(self):
        pass
    
    def start(self):
        return search_space
    
    @staticmethod
    def get_avaiable_actions(state):
        if Game.game_end(state):
            return None
        # 已经有model了，则不能再选择其他的model
        elif '1' in state.model:    
            avaiable_index = get_index_of_zeros(state.preprocessor)
            return avaiable_index
        # 没有选择model，则所有剩余的都可以选
        else:
            return get_index_of_zeros(state.id)
    
    @staticmethod
    def game_end(state):
        #if Game.get_avaiable_actions(state) == None:
        #    return True
        # 达到只有一个model的条件
        res = Counter(state.model)
        if '1' in res and res['1'] == 1:
            random_number = random.random()
            if random_number > 0.8:
                print("GAME END")
                return True
            else:
                return False
        else:
            return False
    
    
    @staticmethod
    def get_score(parameter):
        pipe_list = parameter_to_pipeline(parameter,search_space_dic)
        cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
        score = cross_val_score(pipe_list,X,y,cv=cv)
        return score.mean()  
        

        