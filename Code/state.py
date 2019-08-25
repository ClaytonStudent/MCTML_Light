from Code.setup import search_space_dic,search_space

class State:
    def __init__(self,state = None):
        if state:
            self.id = state
        else:
            self.id = search_space
        self.model_index = list(search_space_dic.keys()).index('m1')
        self.preprocessor = self.id[:self.model_index]
        self.model = self.id[self.model_index:]
        
    def update(self):
        self.model = self.id[self.model_index:]
        self.preprocessor = self.id[:self.model_index]
        
        