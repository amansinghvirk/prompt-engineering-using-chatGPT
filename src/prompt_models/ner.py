import json
from json import JSONDecodeError

from  src.utils import get_completion

def named_entities(
    text_body: str
) -> dict:
    """Identify the named entities from the text
    
    Parameters:
    ----------
        text_body: str 
            text from which entities need to be identified
    
    Return:
    ----------
        list: Identified entities as list
    """
    # query to detect the language
    query = f"""
    Detect the Named Entities from the text delimeted by ``` :
    The text is ```{text_body}```. 
    Instructions:
        - Format the results as JSON object should and only have "NER" as key and detected named 
        entities as value in format as JSON child objects with Named Entity type as key 
        e.g., "PERSON", "ORG" etc. and value as extracted named entity.
        - Enclose property name in double quotes.
    """
    
    # detect the language
    detected_json = get_completion(query)
    try:
        detected_dict = json.loads(detected_json)
        detected_ner = detected_dict["NER"]
    except JSONDecodeError:
        detected_ner = dict()
    except KeyError:
        detected_ner = dict()

    
    return detected_ner