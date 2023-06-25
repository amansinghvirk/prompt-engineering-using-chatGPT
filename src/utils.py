import os
import openai
import json
from json import JSONDecodeError
import yaml
from dotenv import dotenv_values

def open_ai_key():
    config = dotenv_values(".env")
    return config["OPENAI_API_KEY"]

def get_completion(prompt:str, model = "gpt-3.5-turbo"):
    messages = [{
        "role": "user", 
        "content": prompt
    }]
    openai.api_key = open_ai_key()
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]


