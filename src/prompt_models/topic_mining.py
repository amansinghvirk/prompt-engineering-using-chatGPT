import json
from json import JSONDecodeError

from  src.utils import get_completion

def topics(
    text_body: str
) -> list:
    """Identify the differnet topics from the text
    
    Parameters:
    ----------
        text_body: str 
            text from which topics need to be identified
    Return:
    -------
        list: Topics identified as list
    """
    
    # query to detect the language
    prompt = f"""Detect the topics from the text delimited by triple quotes :
    Instructions:
        - Topic should not be longer than 3 words
        - Output a JSON object that contains the following key: Topic
        - Output JSON should only have "Topic" as key and detected topics as value in format of list. 
        - Enclose property name in double quotes.

    Text: ```{text_body}``` 
    """
    
    # detect the language
    detected_json = get_completion(prompt)
    try:
        detected_dict = json.loads(detected_json)
        detected_topics = detected_dict["Topic"]
    except JSONDecodeError:
        detected_topics = []
    except KeyError:
        detected_topics = []

    
    return detected_topics