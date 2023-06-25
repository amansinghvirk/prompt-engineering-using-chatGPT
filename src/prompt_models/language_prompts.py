import json
from json import JSONDecodeError

from  src.utils import get_completion

def detect_language(text_body: str, model="gpt-3.5-turbo") -> dict:
    """returns the identified language
    
    Parameters:
    ----------
        text_body: str 
            text for which language needs to be detected
        model: str
            chat GPT model either gpt-3.5-turbo or gpt-4
    Return:
    ------
        dict: dictionary with key-value
    """
    
    # query to detect the language
    prompt = f"""Detect the language of the delimted by triple quotes :
    ```{text_body}```. 
    ### Instructions ###
        - Output a JSON object that contains the following key: Language
        - Output JSON should only have "Language" as key and detected language as value.
        - Value should be expanded form of detected language. 
        - Enclose propery name in double quotes.    
        - Detect only single language
        - Also clean text removing special characters
        
    Text: ```{text_body}```
    """
    
    # detect the language
    detected_lang_json = get_completion(prompt, model)
    
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


def translate_text(
        text_body: str, 
        original_lang: str, 
        translated_lang: str, 
        model="gpt-3.5-turbo"
    ) -> dict:
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
        model: str
            chat GPT model either gpt-3.5-turbo or gpt-4
    Return:
    ------
        dict: dictionary with key-value
    """
    
    try:
        # query to translate the text
        prompt = f"""
        Translate the following text from {original_lang} to {translated_lang} for the text delimited
        by triple quotes: 

        ### Insturctions ###
            - only keep the translated text in results
            - Output a JSON object that contains the following key: Translation
            - Output JSON should only have "Translation" as key and translated text as value. 
            - Enclose propery name in double quotes.

        Text: ```{text_body}```
        """

        # translate the text from original language to the specified language

        if ((original_lang != translated_lang)
            & (original_lang != "Not Identified")):
            # detect the language
            translated_json = get_completion(prompt, model)
    
            try:
                translated_dict = json.loads(translated_json)
                translated_text = translated_dict["Translation"]
            except JSONDecodeError:
                translated_text = ""
            except KeyError:
                translated_text = ""
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