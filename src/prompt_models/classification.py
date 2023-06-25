import json
from json import JSONDecodeError

from  src.utils import get_completion

def detect_category(
    text_body: str, 
    category_list=["News", "Review", "Tweet", "General",
                   "Article", "Scientific Paper", "Other"],
    model="gpt-3.5-turbo"
) -> str:
    """Classify the text in defined categories
    
    Parameters:
    ----------
        text_body: str 
            text to be classified
        category_list: list 
            list of category labels to be used for classification
        model: str
            chat GPT model either gpt-3.5-turbo or gpt-4
    Return:
    -------
        str: Identified category as text
    """
    # query to detect the language
    prompt = f"""Detect the category from the text delimited by triple quotes : 
    ### Instructions ###
        - Only classify the text into following categories provided in list {category_list}

    Text: ```{text_body}```
    """
    
    # detect the language
    detected_category_json = get_completion(prompt, model)
    try:
        detected_category_dict = json.loads(detected_category_json)
        detected_category = detected_category_dict["Category"]
    except JSONDecodeError:
        detected_category = "Other"
    except KeyError:
        detected_category = "Other"

    return detected_category


def detect_sentiment(
    text_body: str, 
    sentiment_list=["Positive", "Neutral", "Negative"],
    model="gpt-3.5-turbo"
) -> str:
    """Identify the sentiment from the text
    
    Parameters:
    ----------
        text_body: str 
            text to be classified
        sentiment_list: list 
            list of sentiments to be used for classification
        model: str
            chat GPT model either gpt-3.5-turbo or gpt-4
    Return:
    -------
        str: Identified sentiment as text
    """
    
    # query to detect the language
    prompt = f"""Detect the sentiment from the text delimited by triple quotes :
    ### Instructions ###
        - Only classify the text into following sentiment into one category out of {sentiment_list}
        - Output a JSON object that contains the following key: Sentiment
        - Output JSON should only have "Sentiment" as key and detected sentiment as value. 
        - Enclose property name in double quotes.

    Text: ```{text_body}```
    """
    
    # detect the language
    detected_sentiment_json = get_completion(prompt, model)
    try:
        detected_sentiment_dict = json.loads(detected_sentiment_json)
        detected_sentiment = detected_sentiment_dict["Sentiment"]
    except JSONDecodeError:
        detected_sentiment = "Neutral"
    except KeyError:
        detected_category = "Netural"

    
    return detected_sentiment