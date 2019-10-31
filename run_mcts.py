import warnings
warnings.filterwarnings('ignore')

from evaluator import policy_value_fn
from agent import Agent
from functions import find_best
import random


agent = Agent(policy_value_fn,c_puct=2,total_time=30)
agent.play_outs()


