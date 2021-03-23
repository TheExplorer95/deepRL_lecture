from ray_multiprocessing import A2C_Agent
from training import TrainingManagerTDa2c
from tensorflow import keras

actor = keras.models.load_model('results/20210322-215558-polAdvantage-ExplRange0.01-BatchSize128_Delay7_lr1e-05_L2reg/actor_model_rep0.h5')
breakpoint()
agent = A2C_Agent(0, 'LunarLanderContinuous-v2', 128, TDadvActor)
