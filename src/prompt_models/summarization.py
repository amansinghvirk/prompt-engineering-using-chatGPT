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
    summarization_query = f"""Summarize the text delimited by ``` in {lines} lines: 
    ```{text_body}```. 
    Instructions:
        - Format the results as JSON object should and only have "Summary" as key 
        and summarized text as value. 
        - Enclose property name in double quotes."""
    
    # detect the language
    summary_json = get_completion(summarization_query)
    try:
        summary_json_dict = json.loads(summary_json)
        summary = summary_json_dict["Summary"]
    except JSONDecodeError:
        summary = "Fail to summarize"
    except KeyError:
        summary = "Fail to summarize"

    return summary