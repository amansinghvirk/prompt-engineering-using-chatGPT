from src.prompt_models.language_prompts import detect_language, translate_text
from src.prompt_models.summarization import summarize_text
from src.prompt_models.classification import detect_category, detect_sentiment
from src.prompt_models.topic_mining import topics
from src.prompt_models.ner import named_entities

def lang_chain(
        text_body: str, translated_lang=None, 
        text_category_list=["News", "Review", "Tweet", "General",
                        "Article", "Scientific Paper", "Other"],
        text_sentiment_list=["Positive", "Neutral", "Negative"]
    ) -> dict:
    """function apply the language chain on given text 
       and returns back a dictionary with following results
           - Language Detection
           - Language Translation
           - Summarize Text
           - Category Classification
           - Sentiment Classification
           - Topic Generation
           - Named Entity Recognition
        
        Parameters:
        ----------
            text_list: str
                List of texts
            translated_lang: str (default: None)
                Language in which text needs to be translated
            text_category_list: list
                List of Categories for the classification
            text_category_list: list
                List of sentiments for the sentiment classification
    
        Return:
        -------
            dictionary
    """
    
    # Detect the language of text
    detected_language = detect_language(text_body)

    # Translate the text if translated_lang is not none
    if translated_lang is not None:
        try:
            translated_text = translate_text(
                text_body, 
                original_lang=detected_language["DETECTED_LANGUAGE"], 
                translated_lang=translated_lang
            )
            translated_text = translated_text["TRANSLATED_TEXT"]
        except JSONDecodeError:
            print("Translation: Fail to translate")
            translated_text = text_body
        except KeyError:
            print("Translation: Fail to translate")
            detected_category = "Netural"
            translated_text = text_body
    else:
        translated_text = text_body

    # summarize the text
    text_summary = summarize_text(
        text_body=translated_text,
        lines=1
    )

    # Classify text into the category
    text_category = detect_category(
        text_body=translated_text,
        category_list=text_category_list
    )

    # Classify text sentimets
    text_sentiment = detect_sentiment(
        text_body=translated_text,
        sentiment_list=text_sentiment_list
    )

    # Detect topics from the text
    text_topics = topics(
        text_body=translated_text
    )

    # Detect the named entities from the text
    text_ner = named_entities(
        text_body=translated_text
    )


    response = {
        "ORIGINAL_TEXT": text_body,
        "TRANSLATED_TEXT": translated_text,
        "DETECTED_LANGUAGE": detected_language,
        "TRANSLATED_LANGUAGE": translated_lang,
        "SUMMARIZED_TEXT": text_summary,
        "CATEGORY": text_category,
        "SENTIMENT": text_sentiment,
        "TOPICS": text_topics,
        "NER": text_ner
    }

    return response