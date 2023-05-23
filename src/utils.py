import os
import openai
import json
from json import JSONDecodeError
import yaml

## Function to read config yaml file.
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        try:
            yaml_data = yaml.safe_load(file)
            return yaml_data
        except yaml.YAMLError as e:
            print(f"Error while parsing YAML file: {e}")

def open_ai_key():
    config = read_yaml("./config.yaml")
    return config["OPEN_AI_API_KEY"]

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    openai.api_key = open_ai_key()
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]