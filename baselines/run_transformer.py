import os
os.environ['TRANSFORMERS_CACHE'] = 'D:/huggingface_cache'

import torch
import torch.multiprocessing as mp
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
from red_gym_env import RedGymEnv
from pathlib import Path
import uuid
from functools import partial

def generate_pokemon_data(env_config, num_episodes):
    env = create_env(env_config)
    data = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_text = ""
        while not done:
            action = env.action_space.sample()  # You'd want a better action selection method
            step_result = env.step(action)
            obs = step_result[0]
            reward = step_result[1]
            done = step_result[2]
            info = step_result[3] if len(step_result) > 3 else None
            episode_text += f"Observation: {obs}\nAction: {action}\nReward: {reward}\n"
        data.append({"text": episode_text})
    return data

def create_env(env_config):
    return RedGymEnv(env_config)

def train_model(model, tokenizer, dataset, training_args):
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    return trainer

if __name__ == '__main__':
    mp.set_start_method('spawn')

    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'transformer_session_{sess_id}')
    sess_path.mkdir(exist_ok=True)

    env_config = {
        'headless': True, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': 2048 * 10, 
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 
        'use_screen_explore': True, 'reward_scale': 4, 'extra_buttons': False,
        'explore_weight': 3
    }

    # Set up the model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

    # Generate data using multiprocessing
    num_processes = mp.cpu_count()
    episodes_per_process = 1000 // num_processes
    with mp.Pool(processes=num_processes) as pool:
        data_chunks = pool.map(partial(generate_pokemon_data, env_config, episodes_per_process), range(num_processes))

    # Flatten the list of data chunks
    data = [item for sublist in data_chunks for item in sublist]

    # Create the dataset
    dataset = Dataset.from_list(data)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=str(sess_path),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_steps=10_000,
        save_total_limit=2,
        fp16=True,
        use_cuda=torch.cuda.is_available(),
    )

    # Train the model using multiprocessing
    with mp.Pool(processes=1) as pool:  # We use 1 process for training due to GPU memory constraints
        trainer = pool.apply(train_model, (model, tokenizer, dataset, training_args))

    # Save the model
    trainer.save_model(str(sess_path / "pokemon_transformer"))

    print(f"Training complete. Model saved to {sess_path / 'pokemon_transformer'}")
    print(f"Device used for training: {device}")
