import math
import random
from itertools import product
import copy
def get_hyperparameter_space_int(lower,upper):
    distance = upper-lower
    if distance <=10:
        space = [i for i in range(lower,upper+1)]
    elif distance >10 and distance <= 20:
        space = [i for i in range(lower,upper+1) if i%5 ==0]
    elif distance >20 and distance <= 100:
        space = [i for i in range(lower,upper+1) if i%10 ==0]        
    else:
        space = split_based_on_power(lower,upper)
    if lower<min(space):
        space.insert(0,lower)
    if upper> max(space):
        space.append(upper)  
    return space

def get_hyperparameter_space_float(lower,upper):
    if lower == 0:
        lower = lower + 0.001
    multiple = upper / lower
    if multiple < 100:
        space = split_based_on_increment(lower,upper)
    else:
        space = split_based_on_power(lower,upper)
    if lower<min(space):
        space.insert(0,lower)
    if upper> max(space):
        space.append(upper) 
    return space

def split_based_on_power(lower,upper):
    l = []
    temp = lower
    while temp<upper:
        l.append(temp)
        temp = temp * 100
    return l

def split_based_on_increment(lower,upper):
    l = []
    temp = lower
    while temp<upper:
        l.append(temp)
        temp = temp + 0.1
    return l

def initialization(hyper_parameters):
    hyper_parameters = copy.deepcopy(hyper_parameters)
    if hyper_parameters == None:
        return None
    for name,value in hyper_parameters.items():
        if type(value) == tuple:
            init_value = []
            low,high = value[0],value[1]
            if type(low) == int:
                init_value = get_hyperparameter_space_int(low,high)
            else:
                init_value = get_hyperparameter_space_float(low,high)
            hyper_parameters[name] = init_value
    return [tuple(zip(hyper_parameters, v)) for v in product(*hyper_parameters.values())] 