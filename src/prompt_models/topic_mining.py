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
    query = f"""Detect the topics from the text delimited by ``` :
    ```{text_body}```. 
    Instructions:
        - Topic should not be longer than 3 words
        - Format the results as JSON object should and only have "Topic" as key and detected topics 
        as value in format of list.
        - Enclose property name in double quotes.
    """
    
    # detect the language
    detected_json = get_completion(query)
    try:
        detected_dict = json.loads(detected_json)
        detected_topics = detected_dict["Topic"]
    except JSONDecodeError:
        detected_topics = []
    except KeyError:
        detected_topics = []

    
    return detected_topics