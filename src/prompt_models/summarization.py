import json
from json import JSONDecodeError

from  src.utils import get_completion

def summarize_text(
    text_body,
    lines=1
) -> str:
    """Summarize the text
    
    Parameters:
    ----------
        text_body: str 
            text to be summarized
        lines: int
            required number of lines in summarized text
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
        summary = get_completion(prompt)
    except KeyError as e:
        summary = ""
    except Exception as e:
        summary = ""

    return summary