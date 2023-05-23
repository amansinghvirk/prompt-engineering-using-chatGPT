import json
from json import JSONDecodeError

from  src.utils import get_completion

def detect_language(text_body: str) -> dict:
    """returns the identified language
    
    Parameters:
    ----------
        text_body: str 
            text for which language needs to be detected
    Return:
    ------
        dict: dictionary with key-value
    """
    
    # query to detect the language
    lang_detect_query = f"""Detect the language of the delimted by ``` :
    ```{text_body}```. 
    Instructions:
        - Format the results as JSON object which should and only have "Language" as key 
        and detected language as value. Value should be full form of detected language.
        - Detect only single language
        - Also clean text removing special characters
        - Expecting property name enclosed in double quotes"""
    
    # detect the language
    detected_lang_json = get_completion(lang_detect_query)
    
    try:
        detected_lang_dict = json.loads(detected_lang_json)
        detected_lang = detected_lang_dict["Language"]
    except JSONDecodeError:
        detected_lang = "Not Identified"
    except KeyError:
        detected_lang = "Not Identified"
    
    result = {
        "DETECTED_LANGUAGE": detected_lang
    }
    
    return result


def translate_text(text_body: str, original_lang: str, translated_lang: str) -> dict:
    """returns the translated text from original to required language if 
       original language is different from translated language
    
    Parameters:
    ----------
        text_body: str 
            text to be translated
        original_lang: str 
            original language of the text
        translated_lang: str
            Language in which text needs to be translated
    Return:
    ------
        dict: dictionary with key-value
    """
    
    try:
        # query to translate the text
        translation_query = f"""
        Translate the following text from {original_lang} to {translated_lang} for the text delimited
        by ```: 
        ```{text_body}```
            - only keep the translated text in results
        """

        # translate the text from original language to the specified language

        if ((original_lang != translated_lang)
            & (original_lang != "Not Identified")):
            translated_text = get_completion(translation_query)
        elif original_lang == "Not Identified":
            translated_text = ""
        else:
            translated_text = text_body
    except KeyError as e:
        translated_text = ""
    except Exception as e:
        translated_text = "" 
    
    result = {
        "TRANSLATED_TEXT": translated_text
    }
    
    return result