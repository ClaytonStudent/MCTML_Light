import warnings
warnings.filterwarnings('ignore')
import os
from agent import Agent
from functions import find_best


total_time = 60 * 1

agent = Agent(total_time=total_time)
agent.play_outs()


