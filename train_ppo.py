import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import os

def train_agent(env_name="CartPole-v1", total_timesteps=100000):
    print(f"Starting training on {env_name}...")
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    
    os.makedirs("models", exist_ok=True)
    
    eval_callback = EvalCallback(eval_env, best_model_save_path='./models/',
                                 log_path='./logs/', eval_freq=5000,
                                 deterministic=True, render=False)
                                 
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")
    
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    
    model.save(f"models/ppo_{env_name}")
    print("Training complete. Model saved.")
    
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
    
    return model

if __name__ == "__main__":
    train_agent()
