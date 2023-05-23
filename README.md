# Prompt Engineering for NLP tasks
Demonstration of how prompt engineering can be used for LLM models to use specific NLP tasks

This project aims to demonstrate the capabilities of the Large Language Model using Prompt Engineering and Language Chain techniques. By leveraging models like GPT3.5-turbo, we can solve multiple Natural Language Processing tasks without the need for separate models. The tasks include language identification, translation, text summarization, classification, sentiment detection, topic mining, and named entity recognition. These large language models have been trained on diverse language corpora and can efficiently handle various language-related tasks.

### Setup Requirements

Model use the paid OpenAI API for using GPT3.5-turbo model. Demonstration of code is there in nlp_prompts.ipynb notebook to execute the code in notebook follow the below setup tasks:

```python
- install the required libraries using requirements.txt file
    $ pip install -r requirements.txt
    
- create a config.yaml file in root directory with below api key to communicate with OpenAI API:
    OPEN_AI_API_KEY: <apikey>  
```

```python
import yaml
from src.prompt_models.language_prompts import detect_language, translate_text
from src.prompt_models.summarization import summarize_text
from src.prompt_models.classification import detect_category, detect_sentiment
from src.prompt_models.topic_mining import topics
from src.prompt_models.ner import named_entities
from src.langchain import lang_chain
``` 

#### Following text is used for the examples, text is in hindi language


```python
text = """टेक्सटाइल सेक्टर में घरेलू निर्माण को बढ़ाने और निर्यात को बढ़ावा देने के लिये केन्द्रीय 
    मंत्रिमंडल ने आज  10,683 करोड़ रुपये की उत्पादन आधारित प्रोत्साहन (पीएलआई) योजना को 
    मंजूरी दे दी है। केन्द्रीय मंत्री अनुराग ठाकुर ने आज इस बारे में जानकारी दी। 
    ये फैसला प्रधानमंत्री नरेन्द्र मोदी की अध्यक्षता में हुई कैबिनेट की बैठक में लिया गया है। 
    मंत्रिमंडल इससे पहले देश में विनिर्माण क्षमता और निर्यात को बढ़ावा देने के लिये 13 प्रमुख क्षेत्रों के लिये 
    पीएलआई योजना को मंजूरी दे चुका है।

    केन्द्रीय मंत्री अनुराग ठाकुर के मुताबिक कैबिनेट ने मानव निर्मित रेशे  (man-made fibre) मानव निर्मित 
    फैब्रिक और टेक्निकल टेक्सटाइल के 10 सेग्मेंट या उत्पादों के लिये प्रोत्साहन योजना को मंजूरी दी है। 
    टेक्निकल टेक्सटाइल फार्मा, मेटल, ऑटो आदि सेक्टर में काम आने वाले खास फैब्रिक होते हैं। क
    ोविड के दौरान इनकी मांग काफी बढ़ी थी, जिसके बाद सरकार ने इनके घरेलू उत्पादन को बढ़ाने पर 
    जोर दिया है। सरकार का अनुमान है कि इस योजना से सीधे तौर पर 7.5 लाख लोगों को रोजगार उपलब्ध 
    होगा वहीं अप्रत्यक्ष रूप से इससे कहीं ज्यादा लोगों को रोजगार मिल सकेगा। 
    योजना से छोटे शहरों को फायदा मिलने की उम्मीद है।"""
```

#### <b>Task 1: </b> Language Detection


```python
detected_language = detect_language(text_body=text)
detected_language
```




    {'DETECTED_LANGUAGE': 'Hindi'}



#### <b>Task 2: </b> Language Translation


```python
translated_text = translate_text(
    text_body=text, 
    original_lang=detected_language["DETECTED_LANGUAGE"], 
    translated_lang="English"
)
translated_text
```




    {'TRANSLATED_TEXT': 'The Central Cabinet has approved a production-based incentive (PBI) scheme worth Rs 10,683 crore today to increase domestic construction and boost exports in the textile sector. Central Minister Anurag Thakur provided information about this today. This decision was taken in a cabinet meeting chaired by Prime Minister Narendra Modi. Prior to this, the Cabinet had approved the PBI scheme for 13 major sectors in the country to increase manufacturing capacity and exports. According to Central Minister Anurag Thakur, the Cabinet has approved the incentive scheme for 10 segments or products of man-made fibers, man-made fabrics, and technical textiles. Technical textiles include special fabrics used in the pharma, metal, auto, and other sectors. Their demand increased significantly during COVID, after which the government emphasized increasing their domestic production. The government estimates that this scheme will directly provide employment to 7.5 lakh people and indirectly to many more. The scheme is expected to benefit small towns.'}



#### <b>Task 3: </b> Text Summarization


```python
text_summary = summarize_text(
    text_body=translated_text["TRANSLATED_TEXT"],
    lines=1
)
text_summary
```




    'The Central Cabinet has approved a production-based incentive (PBI) scheme worth Rs 10,683 crore to increase domestic construction and boost exports in the textile sector, covering 10 segments or products of man-made fibers, man-made fabrics, and technical textiles, which is expected to directly provide employment to 7.5 lakh people and benefit small towns.'



#### <b>Task 4: </b> Text CLassification


```python
text_category = detect_category(
    text_body=translated_text["TRANSLATED_TEXT"],
    category_list=["News", "Review", "Tweet", "General",
                   "Article", "Scientific Paper", "Other"]
)
text_category
```




    'News'



#### <b>Task 5: </b> Sentiment Classification


```python
text_sentiment = detect_sentiment(
    text_body=translated_text["TRANSLATED_TEXT"],
    sentiment_list=["Positive", "Neutral", "Negative"]
)
text_sentiment
```




    'Positive'



#### <b>Task 6: </b>Topic Mining


```python
text_topics = topics(
    text_body=translated_text["TRANSLATED_TEXT"]
)
text_topics
```




    ['Textile sector', 'Production-based incentive', 'Exports']



#### <b>Task 7:</b> Named Entity Recognition


```python
text_ner = named_entities(
    text_body=translated_text["TRANSLATED_TEXT"]
)
text_ner
```




    {'ORG': ['Central Cabinet',
      'PBI',
      'Cabinet',
      'Pharma',
      'Metal',
      'Auto',
      'Government'],
     'PERSON': ['Anurag Thakur', 'Narendra Modi'],
     'MONEY': ['Rs 10,683 crore'],
     'DATE': ['today']}



### Combine the results as chain

Sometimes it is difficult or not efficient to achive the results in single prompt, so chaining the results might be useful in some scenarios to get the final results. Following function utilize the above prompts to get the results


```python
model_results = lang_chain(
    text_body=text, 
    translated_lang="English", 
    text_category_list=["News", "Review", "Tweet", "General",
                        "Article", "Scientific Paper", "Other"],
    text_sentiment_list=["Positive", "Neutral", "Negative"]
)
model_results
```




    {'ORIGINAL_TEXT': 'टेक्सटाइल सेक्टर में घरेलू निर्माण को बढ़ाने और निर्यात को बढ़ावा देने के लिये केन्द्रीय \n    मंत्रिमंडल ने आज  10,683 करोड़ रुपये की उत्पादन आधारित प्रोत्साहन (पीएलआई) योजना को \n    मंजूरी दे दी है। केन्द्रीय मंत्री अनुराग ठाकुर ने आज इस बारे में जानकारी दी। \n    ये फैसला प्रधानमंत्री नरेन्द्र मोदी की अध्यक्षता में हुई कैबिनेट की बैठक में लिया गया है। \n    मंत्रिमंडल इससे पहले देश में विनिर्माण क्षमता और निर्यात को बढ़ावा देने के लिये 13 प्रमुख क्षेत्रों के लिये \n    पीएलआई योजना को मंजूरी दे चुका है।\n\n    केन्द्रीय मंत्री अनुराग ठाकुर के मुताबिक कैबिनेट ने मानव निर्मित रेशे  (man-made fibre) मानव निर्मित \n    फैब्रिक और टेक्निकल टेक्सटाइल के 10 सेग्मेंट या उत्पादों के लिये प्रोत्साहन योजना को मंजूरी दी है। \n    टेक्निकल टेक्सटाइल फार्मा, मेटल, ऑटो आदि सेक्टर में काम आने वाले खास फैब्रिक होते हैं। क\n    ोविड के दौरान इनकी मांग काफी बढ़ी थी, जिसके बाद सरकार ने इनके घरेलू उत्पादन को बढ़ाने पर \n    जोर दिया है। सरकार का अनुमान है कि इस योजना से सीधे तौर पर 7.5 लाख लोगों को रोजगार उपलब्ध \n    होगा वहीं अप्रत्यक्ष रूप से इससे कहीं ज्यादा लोगों को रोजगार मिल सकेगा। \n    योजना से छोटे शहरों को फायदा मिलने की उम्मीद है।',
     'TRANSLATED_TEXT': 'The Central Cabinet has approved a production-based incentive (PBI) scheme worth Rs 10,683 crore today to increase domestic construction and boost exports in the textile sector. Central Minister Anurag Thakur provided information about this today. This decision was taken in a cabinet meeting chaired by Prime Minister Narendra Modi. Prior to this, the Cabinet had approved the PBI scheme for 13 major sectors in the country to increase manufacturing capacity and exports. According to Central Minister Anurag Thakur, the Cabinet has approved the incentive scheme for 10 segments or products of man-made fibers, man-made fabrics, and technical textiles. Technical textiles include special fabrics used in the pharma, metal, auto, and other sectors. Their demand increased significantly during COVID, after which the government emphasized increasing their domestic production. The government estimates that this scheme will directly provide employment to 7.5 lakh people and indirectly to many more. The hope is that small towns will benefit from the scheme.',
     'DETECTED_LANGUAGE': {'DETECTED_LANGUAGE': 'Hindi'},
     'TRANSLATED_LANGUAGE': 'English',
     'SUMMARIZED_TEXT': 'The Central Cabinet has approved a production-based incentive (PBI) scheme worth Rs 10,683 crore to increase domestic construction and boost exports in the textile sector, covering 10 segments or products of man-made fibers, man-made fabrics, and technical textiles, which is expected to provide employment to 7.5 lakh people and benefit small towns.',
     'CATEGORY': 'News',
     'SENTIMENT': 'Positive',
     'TOPICS': ['Textile sector', 'Production-based incentive', 'Exports'],
     'NER': {'ORG': ['Central Cabinet',
       'PBI',
       'Cabinet',
       'Pharma',
       'Metal',
       'Auto',
       'Government'],
      'PERSON': ['Anurag Thakur', 'Narendra Modi'],
      'MONEY': ['Rs 10,683 crore'],
      'PRODUCT': ['Man-made fibers', 'Man-made fabrics', 'Technical textiles'],
      'QUANTITY': ['13', '10', '7.5 lakh'],
      'GPE': ['Country']}}




```python
text_list = [
    """केंद्रीय मंत्रीमंडल ने टेक्‍सटाइल उद्योग में रोजगार के अवसर बढ़ाने और निवेश को आकर्षित करने के ल
    िए सात मेगा इंटीग्रेटेड टेक्‍सटाइल पार्क की स्‍थापना करने के प्रस्‍ताव को अपनी मंजूरी दे दी। 
    प्रधानमंत्री नरेंद्र मोदी की अध्‍यक्षता में बुधवार को हुई केंद्रीय मंत्रिमंडल की बैठक में 7 मेगा इंटीग्रेटेड 
    टेक्‍सटाइल रीजन एंड अपैरल (पीएम-मित्र) पार्क की स्‍थापना को मंजूरी प्रदान की गई। 5 साल में 
    इन पार्कों पर कुल 4445 करोड़ रुपये का खर्च किया जाएगा।

    केंद्रीय मंत्री पीयूष गोयल ने प्रेस कॉन्‍फ्रेंस में बताया कि पीएम मित्र योजना के लिए पांच साल की अवधि 
    के लिए कुल 4445 करोड़ रुपये का प्रावधान किया गया है। यह निर्णय पीएम मोदी के 5एफ दृ्रष्टिकोण 
    से प्रेरित है, जो फार्म टू फाइबर टू फैक्‍टरी टू फैशन टू फॉरेन है।""",
    """ਪ੍ਰਧਾਨ ਮੰਤਰੀ ਨਰਿੰਦਰ ਮੋਦੀ ਨੂੰ ਫਿਜੀ ਦੇ ਸਰਵਉੱਚ ਸਨਮਾਨ ਨਾਲ ਸਨਮਾਨਿਤ ਕੀਤਾ ਗਿਆ ਹੈ। ਪੀਐਮ ਮੋਦੀ ਨੂੰ ਫ
    ਿਜੀ ਦੇ ਸਰਵਉੱਚ ਸਨਮਾਨ 'ਕੰਪੇਨੀਅਨ ਆਫ ਦਿ ਆਰਡਰ ਆਫ ਫਿਜੀ' ਨਾਲ ਫਿਜੀ ਦੀ ਪ੍ਰਧਾਨ ਮੰਤਰੀ ਸਿਤਾਵਾਨੀ 
    ਰਬੂਕਾ ਨੇ ਸਨਮਾਨਿਤ ਕੀਤਾ ਹੈ। ਹਾਲਾਂਕਿ, ਹੁਣ ਤੱਕ ਸਿਰਫ ਕੁਝ ਗੈਰ-ਫਿਜੀ ਲੋਕਾਂ ਨੂੰ ਇਹ ਸਨਮਾਨ ਮਿਲਿਆ ਹੈ। 
    ਇਸ ਦੇ ਨਾਲ ਹੀ ਪਲਾਊ ਗਣਰਾਜ ਨੇ ਪੀਐਮ ਨਰਿੰਦਰ ਮੋਦੀ ਨੂੰ ਵੀ ਸਨਮਾਨਿਤ ਕੀਤਾ। ਰਿਪਬਲਿਕ ਆਫ਼ ਪਲਾਊ ਨੂੰ 
    ਅਬਾਕਲ ਅਵਾਰਡ ਨਾਲ ਸਨਮਾਨਿਤ ਕੀਤਾ ਗਿਆ। ਇਹ ਦੋਵੇਂ ਐਵਾਰਡ ਪੀਐਮ ਮੋਦੀ ਨੂੰ ਪਾਪੂਆ ਨਿਊ ਗਿਨੀ ਵਿੱਚ ਹੀ ਦ
    ਿੱਤੇ ਗਏ ਹਨ।

    ਪਾਪੂਆ ਨਿਊ ਗਿਨੀ ਨੇ ਪ੍ਰਸ਼ਾਂਤ ਟਾਪੂ ਦੇਸ਼ਾਂ ਦੀ ਏਕਤਾ ਦਾ ਸਮਰਥਨ ਕਰਨ ਅਤੇ ਗਲੋਬਲ ਸਾਊਥ ਦੀ ਅਗਵਾਈ ਕਰਨ 
    ਲਈ ਪ੍ਰਧਾਨ ਮੰਤਰੀ ਨਰਿੰਦਰ ਮੋਦੀ ਨੂੰ ‘Companion of the Order of Logohu’ ਸਨਮਾਨ ਨਾਲ 
    ਸਨਮਾਨਿਤ ਕੀਤਾ ਹੈ। ਦੂਜੇ ਦੇਸ਼ਾਂ ਦੇ ਬਹੁਤ ਘੱਟ ਲੋਕਾਂ ਨੂੰ ਇਹ ਪੁਰਸਕਾਰ ਮਿਲਿਆ ਹੈ। 
    ਇਸ ਦੇ ਨਾਲ ਹੀ ਪ੍ਰਧਾਨ ਮੰਤਰੀ ਨਰਿੰਦਰ ਮੋਦੀ ਵੱਲੋਂ ਫੋਰਮ ਫਾਰ ਇੰਡੀਆ-ਪੈਸੀਫਿਕ 
    ਆਈਲੈਂਡਸ ਕੋਆਪਰੇਸ਼ਨ (FIPIC) ਦੇ ਨੇਤਾਵਾਂ ਲਈ ਆਯੋਜਿਤ ਦੁਪਹਿਰ ਦੇ ਖਾਣੇ ਵਿੱਚ ਬਾਜਰੇ ਦੀ ਬਣੀ ਬ
    ਿਰਯਾਨੀ ਪਰੋਸੀ ਜਾਵੇਗੀ।
    """,
    """আচ্ছা সৌরভ গঙ্গোপাধ্যায়ের সঙ্গে বিরাট কোহলির দুরত্ব কি আদৌ কমেছে? মহারাজ কি 
    'কিং কোহলি'-কে মন থেকে মাফ করে দিয়েছেন? ভারতের প্রাক্তন অধিনায়কের নতুন টুইট 
    দেখে নেটপাড়া কিন্তু অন্য গন্ধ পাচ্ছেন। কিন্তু কেন সৌরভের নতুন টুইটকে ঘিরে বিতর্ক তৈরি হল? 

    রবিবার অর্থাৎ ২২ মে ৬১ বলে ১০১ রানে অপরাজিত ছিলেন বিরাট। সেই সুবাদে নির্ধারিত 
    ২০ ওভারে ৫ উইকেটে ১৯৭ রান তুলেছিল রয়্যাল চ্যালেঞ্জার্স ব্যাঙ্গালোরকে 
    । জবাবে ব্যাট করতে নেমে দাপট দেখান শুভমন গিল । মাত্র ৫২ বলে ১০৪ রানে অপরাজিত 
    থাকেন পঞ্জাব তনয়। ফলে ৬ উইকেটে গুজরাত টাইটান্স শুধু জয়ই পায়নি, চলতি আইপিএল 
    (IPL 2023) থেকে আরসিবি-কে (RCB) ছিটকে দিয়েছে গুজরাত। 
    সেই ম্যাচ শেষ হওয়ার পরেই একটি টুইট করে সৌরভ। আর সেটা নিয়েই শুরু হয়েছে নতুন বিতর্ক।
    """
]
```

### Evaluate the prompt results on more texts

Passing the text in different results to evaluate how efficiently it can process the information


```python
for idx, text in enumerate(text_list):
    print("\n")
    print(f"Results for the text: {idx+1}")
    print("-----------------------------")
    print("Original Text:")
    print(text)
    print("-------------")
    print("Model Results:")
    model_results = lang_chain(
        text_body=text, 
        translated_lang="English", 
        text_category_list=["News", "Review", "Tweet", "General",
                            "Article", "Scientific Paper", "Other"],
        text_sentiment_list=["Positive", "Neutral", "Negative"]
    )
    model_results.pop("ORIGINAL_TEXT")
    print(yaml.dump(model_results))
    
```

    
    
    Results for the text: 1
    -----------------------------
    Original Text:
    केंद्रीय मंत्रीमंडल ने टेक्‍सटाइल उद्योग में रोजगार के अवसर बढ़ाने और निवेश को आकर्षित करने के ल
        िए सात मेगा इंटीग्रेटेड टेक्‍सटाइल पार्क की स्‍थापना करने के प्रस्‍ताव को अपनी मंजूरी दे दी। 
        प्रधानमंत्री नरेंद्र मोदी की अध्‍यक्षता में बुधवार को हुई केंद्रीय मंत्रिमंडल की बैठक में 7 मेगा इंटीग्रेटेड 
        टेक्‍सटाइल रीजन एंड अपैरल (पीएम-मित्र) पार्क की स्‍थापना को मंजूरी प्रदान की गई। 5 साल में 
        इन पार्कों पर कुल 4445 करोड़ रुपये का खर्च किया जाएगा।
    
        केंद्रीय मंत्री पीयूष गोयल ने प्रेस कॉन्‍फ्रेंस में बताया कि पीएम मित्र योजना के लिए पांच साल की अवधि 
        के लिए कुल 4445 करोड़ रुपये का प्रावधान किया गया है। यह निर्णय पीएम मोदी के 5एफ दृ्रष्टिकोण 
        से प्रेरित है, जो फार्म टू फाइबर टू फैक्‍टरी टू फैशन टू फॉरेन है।
    -------------
    Model Results:
    CATEGORY: News
    DETECTED_LANGUAGE:
      DETECTED_LANGUAGE: Hindi
    NER:
      MONEY:
      - Rs 4,445 crore
      - 5 years
      ORG:
      - Central Cabinet
      - PM-MITRA
      - Central Minister
      PERSON:
      - Narendra Modi
      - Piyush Goyal
    SENTIMENT: Positive
    SUMMARIZED_TEXT: The Central Cabinet has approved the establishment of seven mega
      integrated textile parks with a total budget of Rs 4,445 crore for five years to
      increase employment opportunities and attract investment in the textile industry,
      inspired by PM Modi's 5F vision.
    TOPICS:
    - Textile Parks
    - Employment Opportunities
    - Investment
    TRANSLATED_LANGUAGE: English
    TRANSLATED_TEXT: The Central Cabinet has approved the proposal to establish seven
      mega integrated textile parks to increase employment opportunities and attract investment
      in the textile industry. The meeting of the Central Cabinet was held on Wednesday
      under the chairmanship of Prime Minister Narendra Modi, in which the establishment
      of 7 mega integrated textile region and apparel (PM-MITRA) parks was approved. A
      total of Rs 4,445 crore will be spent on these parks in 5 years. Central Minister
      Piyush Goyal said in a press conference that a provision of Rs 4,445 crore has been
      made for a period of five years for the PM-MITRA scheme. This decision is inspired
      by PM Modi's 5F vision, which is Farm to Fiber to Factory to Fashion to Foreign.
    
    
    
    Results for the text: 2
    -----------------------------
    Original Text:
    ਪ੍ਰਧਾਨ ਮੰਤਰੀ ਨਰਿੰਦਰ ਮੋਦੀ ਨੂੰ ਫਿਜੀ ਦੇ ਸਰਵਉੱਚ ਸਨਮਾਨ ਨਾਲ ਸਨਮਾਨਿਤ ਕੀਤਾ ਗਿਆ ਹੈ। ਪੀਐਮ ਮੋਦੀ ਨੂੰ ਫ
        ਿਜੀ ਦੇ ਸਰਵਉੱਚ ਸਨਮਾਨ 'ਕੰਪੇਨੀਅਨ ਆਫ ਦਿ ਆਰਡਰ ਆਫ ਫਿਜੀ' ਨਾਲ ਫਿਜੀ ਦੀ ਪ੍ਰਧਾਨ ਮੰਤਰੀ ਸਿਤਾਵਾਨੀ 
        ਰਬੂਕਾ ਨੇ ਸਨਮਾਨਿਤ ਕੀਤਾ ਹੈ। ਹਾਲਾਂਕਿ, ਹੁਣ ਤੱਕ ਸਿਰਫ ਕੁਝ ਗੈਰ-ਫਿਜੀ ਲੋਕਾਂ ਨੂੰ ਇਹ ਸਨਮਾਨ ਮਿਲਿਆ ਹੈ। 
        ਇਸ ਦੇ ਨਾਲ ਹੀ ਪਲਾਊ ਗਣਰਾਜ ਨੇ ਪੀਐਮ ਨਰਿੰਦਰ ਮੋਦੀ ਨੂੰ ਵੀ ਸਨਮਾਨਿਤ ਕੀਤਾ। ਰਿਪਬਲਿਕ ਆਫ਼ ਪਲਾਊ ਨੂੰ 
        ਅਬਾਕਲ ਅਵਾਰਡ ਨਾਲ ਸਨਮਾਨਿਤ ਕੀਤਾ ਗਿਆ। ਇਹ ਦੋਵੇਂ ਐਵਾਰਡ ਪੀਐਮ ਮੋਦੀ ਨੂੰ ਪਾਪੂਆ ਨਿਊ ਗਿਨੀ ਵਿੱਚ ਹੀ ਦ
        ਿੱਤੇ ਗਏ ਹਨ।
    
        ਪਾਪੂਆ ਨਿਊ ਗਿਨੀ ਨੇ ਪ੍ਰਸ਼ਾਂਤ ਟਾਪੂ ਦੇਸ਼ਾਂ ਦੀ ਏਕਤਾ ਦਾ ਸਮਰਥਨ ਕਰਨ ਅਤੇ ਗਲੋਬਲ ਸਾਊਥ ਦੀ ਅਗਵਾਈ ਕਰਨ 
        ਲਈ ਪ੍ਰਧਾਨ ਮੰਤਰੀ ਨਰਿੰਦਰ ਮੋਦੀ ਨੂੰ ‘Companion of the Order of Logohu’ ਸਨਮਾਨ ਨਾਲ 
        ਸਨਮਾਨਿਤ ਕੀਤਾ ਹੈ। ਦੂਜੇ ਦੇਸ਼ਾਂ ਦੇ ਬਹੁਤ ਘੱਟ ਲੋਕਾਂ ਨੂੰ ਇਹ ਪੁਰਸਕਾਰ ਮਿਲਿਆ ਹੈ। 
        ਇਸ ਦੇ ਨਾਲ ਹੀ ਪ੍ਰਧਾਨ ਮੰਤਰੀ ਨਰਿੰਦਰ ਮੋਦੀ ਵੱਲੋਂ ਫੋਰਮ ਫਾਰ ਇੰਡੀਆ-ਪੈਸੀਫਿਕ 
        ਆਈਲੈਂਡਸ ਕੋਆਪਰੇਸ਼ਨ (FIPIC) ਦੇ ਨੇਤਾਵਾਂ ਲਈ ਆਯੋਜਿਤ ਦੁਪਹਿਰ ਦੇ ਖਾਣੇ ਵਿੱਚ ਬਾਜਰੇ ਦੀ ਬਣੀ ਬ
        ਿਰਯਾਨੀ ਪਰੋਸੀ ਜਾਵੇਗੀ।
        
    -------------
    Model Results:
    CATEGORY: News
    DETECTED_LANGUAGE:
      DETECTED_LANGUAGE: Punjabi
    NER:
      LOCATION:
      - Fiji
      - Palau
      - Papua New Guinea
      ORG:
      - Companion of the Order of Fiji
      - Order of Abakal Award
      - Republic of Palau
      - Companion of the Order of Logohu
      - Global South
      - Forum for India-Pacific Islands Cooperation (FIPIC)
      PERSON:
      - Narendra Modi
      - Sita Veeni Rabuka
    SENTIMENT: Positive
    SUMMARIZED_TEXT: Prime Minister Narendra Modi has been honored with the highest honor
      of Fiji, 'Companion of the Order of Fiji', and also received the 'Companion of the
      Order of Logohu' from Papua New Guinea for supporting peaceful countries and leading
      Global South. He also organized a business lunch for the leaders of FIPIC.
    TOPICS:
    - Honors
    - Fiji
    - Palau
    - Papua New Guinea
    - Unity
    - Global South
    - Business Lunch
    TRANSLATED_LANGUAGE: English
    TRANSLATED_TEXT: 'Prime Minister Narendra Modi has been honored with the highest honor
      of Fiji. PM Modi has been honored with the ''Companion of the Order of Fiji'' for
      his contribution to Fiji, along with the Prime Minister of Fiji, Sita Veeni Rabuka.
      However, so far only a few non-Fijian people have received this honor. Palau President
      also honored PM Narendra Modi. The Republic of Palau has now been honored with the
      Order of Abakal Award. Both these awards were given to PM Modi only in Papua New
      Guinea.
    
    
      Papua New Guinea has honored Prime Minister Narendra Modi with the ''Companion of
      the Order of Logohu'' for supporting the unity of peaceful countries and leading
      Global South. Few people from other countries have received this award. Along with
      this, Prime Minister Narendra Modi has organized a business lunch for the leaders
      of the Forum for India-Pacific Islands Cooperation (FIPIC) in which the business
      community will participate.'
    
    
    
    Results for the text: 3
    -----------------------------
    Original Text:
    আচ্ছা সৌরভ গঙ্গোপাধ্যায়ের সঙ্গে বিরাট কোহলির দুরত্ব কি আদৌ কমেছে? মহারাজ কি 
        'কিং কোহলি'-কে মন থেকে মাফ করে দিয়েছেন? ভারতের প্রাক্তন অধিনায়কের নতুন টুইট 
        দেখে নেটপাড়া কিন্তু অন্য গন্ধ পাচ্ছেন। কিন্তু কেন সৌরভের নতুন টুইটকে ঘিরে বিতর্ক তৈরি হল? 
    
        রবিবার অর্থাৎ ২২ মে ৬১ বলে ১০১ রানে অপরাজিত ছিলেন বিরাট। সেই সুবাদে নির্ধারিত 
        ২০ ওভারে ৫ উইকেটে ১৯৭ রান তুলেছিল রয়্যাল চ্যালেঞ্জার্স ব্যাঙ্গালোরকে 
        । জবাবে ব্যাট করতে নেমে দাপট দেখান শুভমন গিল । মাত্র ৫২ বলে ১০৪ রানে অপরাজিত 
        থাকেন পঞ্জাব তনয়। ফলে ৬ উইকেটে গুজরাত টাইটান্স শুধু জয়ই পায়নি, চলতি আইপিএল 
        (IPL 2023) থেকে আরসিবি-কে (RCB) ছিটকে দিয়েছে গুজরাত। 
        সেই ম্যাচ শেষ হওয়ার পরেই একটি টুইট করে সৌরভ। আর সেটা নিয়েই শুরু হয়েছে নতুন বিতর্ক।
        
    -------------
    Model Results:
    CATEGORY: News
    DETECTED_LANGUAGE:
      DETECTED_LANGUAGE: Bengali
    NER:
      DATE:
      - Sunday
      - May 22
      - '2023'
      ORG:
      - Royal Challengers Bangalore
      - Kolkata Knight Riders
      - Punjab
      - Gujarat Titans
      - IPL
      PERSON:
      - Sachin
      - Sourav Ganguly
      - Maharaj
      - King Kohli
      - Virat
      - Shubman Gill
    SENTIMENT: Neutral
    SUMMARIZED_TEXT: Controversy surrounds Sourav Ganguly's tweet after Virat Kohli's
      Royal Challengers Bangalore lost to Kolkata Knight Riders and were eliminated from
      IPL 2023.
    TOPICS:
    - Cricket
    - Controversy
    - IPL
    TRANSLATED_LANGUAGE: English
    TRANSLATED_TEXT: '"What, has the distance between Sachin and Sourav Ganguly decreased?
      Has the Maharaj forgiven ''King Kohli'' from his heart? After seeing the new tweet
      of the former leader of India, the internet is buzzing with a different scent. But
      why has there been controversy surrounding Sourav''s new tweet?
    
    
      On Sunday, May 22, 61, Virat remained unbeaten at 101 runs. In the allotted 20 overs,
      Royal Challengers Bangalore scored 197 runs against Kolkata Knight Riders. In response,
      Shubman Gill scored an unbeaten 104 runs in just 52 balls, leading Punjab to victory
      with 6 wickets. As a result, Gujarat Titans not only lost the match but also eliminated
      RCB from IPL 2023.
    
    
      After the end of that match, Sourav tweeted, and that''s where the new controversy
      began."'
    

