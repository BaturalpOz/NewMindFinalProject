from abc import abstractmethod
from typing import Tuple


class base_model(object): #Any model class using this pipeline has to inherit this class
    def __init__(self,model_name):
        self.model_name = model_name
    @abstractmethod
    def run_model(self,prompt:str,max_new_tokens:int) -> Tuple[str,int]:
        pass
    