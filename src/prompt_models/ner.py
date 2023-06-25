import json
from json import JSONDecodeError

from  src.utils import get_completion

def named_entities(
    text_body: str,
    model="gpt-3.5-turbo"
) -> dict:
    """Identify the named entities from the text
    
    Parameters:
    ----------
        text_body: str 
            text from which entities need to be identified
        model: str
            chat GPT model either gpt-3.5-turbo or gpt-4
    Return:
    ----------
        list: Identified entities as list
    """
    # query to detect the language
    prompt = f"""
    Detect the Named Entities from the text delimeted by triple quotes :
    ### Instructions ###
        - Output a JSON object that contains the following key: NER
        - Output JSON should only have "NER" as key and detected named 
          entities as value
        - Format the detected named entities as JSON child objects with Named Entity type as key 
          and value as extracted named entities in list type. sample keys for named entities "PERSON", "ORG"
        - Enclose propery name in double quotes.

    Text: ```{text_body}```
    """
    
    # detect the language
    detected_json = get_completion(prompt, model)
    try:
        detected_dict = json.loads(detected_json)
        detected_ner = detected_dict["NER"]
    except JSONDecodeError:
        detected_ner = dict()
    except KeyError:
        detected_ner = dict()

    
    return detected_ner