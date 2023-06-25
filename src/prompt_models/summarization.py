import json
from json import JSONDecodeError

from  src.utils import get_completion

def summarize_text(
    text_body,
    lines=1,
    model="gpt-3.5-turbo"
) -> str:
    """Summarize the text
    
    Parameters:
    ----------
        text_body: str 
            text to be summarized
        lines: int
            required number of lines in summarized text
        model: str
            chat GPT model either gpt-3.5-turbo or gpt-4
    Returns:
        str: Summary as text
    """
    
    # query to detect the language
    prompt = f"""Summarize the text delimited by triple quotes: 
    ### Instructions ###
        - Summarized text in {lines} lines.
        - Output should only contains summarized text

    Text: ```{text_body}```    
    """
    
    # detect the language
    try:
        summary = get_completion(prompt, model)
    except KeyError as e:
        summary = ""
    except Exception as e:
        summary = ""

    return summary