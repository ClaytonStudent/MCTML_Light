import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import os
from agent_auto import Agent
from functions import find_best




import glob
df_names = [file for file in glob.glob('./autosklearn/datasets_small/*.csv')]
total_time = 20*6
number = 42
agent = Agent(total_time=total_time,df_name=df_names[number])
print(df_names[number])
agent.play_outs()
print(df_names[number])

#print(df_name)
#for df_name in df_names[8:8]:
#    print(df_name)
#    agent = Agent(total_time=total_time,df_name=df_name)
#    agent.play_outs()
    
