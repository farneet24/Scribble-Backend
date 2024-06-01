import os
import json
import spacy
from spacy.lang.en import English
from spacy.pipeline import Sentencizer
import textstat
import numpy as np
from collections import Counter
from langchain_groq import ChatGroq
from django.http import JsonResponse
from langchain_openai import ChatOpenAI
from django.views.decorators.csrf import csrf_exempt
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


# Initialize a blank English model
nlp = English()

# Add the built-in sentencizer component to the pipeline
nlp.add_pipe('sentencizer')

# You could use the default spaCy's stop words or define your own
stop_words = spacy.lang.en.stop_words.STOP_WORDS

def analyze_text(text):
    # Process the text using the NLP object
    doc = nlp(text)
    
    # Tokenization and filtering stopwords
    all_words = [token.text for token in doc if token.is_alpha]
    filtered_words = [word for word in all_words if word.lower() not in stop_words]
    
    # Getting sentences
    sentences = list(doc.sents)
    
    # Handling paragraphs
    paragraphs = [p for p in text.split('\n') if p]

    # Calculations
    num_chars_with_spaces = len(text)
    num_chars_without_spaces = len(text.replace(" ", ""))
    num_words = len(all_words)
    num_sentences = len(sentences)
    num_paragraphs = len(paragraphs)
    avg_word_length = np.mean([len(word) for word in all_words]) if all_words else 0
    avg_sentence_length = num_words / num_sentences if num_sentences else 0
    word_freq = Counter(filtered_words)
    most_common_words = word_freq.most_common(5)
    least_common_words = word_freq.most_common()[:-6:-1]
    unique_words = len(set(filtered_words))
    lexical_density = (unique_words / num_words) * 100 if num_words else 0

    # Readability scores using textstat
    readability_scores = {
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'smog_index': textstat.smog_index(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'coleman_liau_index': textstat.coleman_liau_index(text),
        'automated_readability_index': textstat.automated_readability_index(text),
        'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
    }

    # Estimate reading time (average reading speed is approx. 200 wpm)
    reading_time_minutes = num_words / 200

    # Compile results
    analysis_results = {
        'Total Characters (with spaces)': num_chars_with_spaces,
        'Total Characters (without spaces)': num_chars_without_spaces,
        'Total Words': num_words,
        'Total Sentences': num_sentences,
        'Total Paragraphs': num_paragraphs,
        'Average Word Length': round(avg_word_length, 2),
        'Average Sentence Length': round(avg_sentence_length, 2),
        'Most Common Words': most_common_words,
        'Least Common Words': least_common_words,
        'Lexical Density (%)': round(lexical_density, 2),
        'Readability Scores': readability_scores,
        'Reading Time': reading_time_minutes,
        'Difficult Words': textstat.difficult_words(text)
    }

    return analysis_results


@csrf_exempt
def analysis(request):

    data = json.loads(request.body.decode('utf-8'))
    paragraph = data['Body'];
    analysis = analyze_text(paragraph)

    return JsonResponse({'message' : 'The message was received. Thank you', 'analysis' : analysis})


gemini_llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv('GEMINI_KEY'))
gpt_llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=os.getenv('OPENAI_KEY'))
llama_llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", groq_api_key=os.getenv('GROQ_API_KEY'))

# Task can be general, question, summary, paraphrase, sentiment, emotion, named, topic, translate
def gemini(task, data):
    try:
        if task == 'general':
            question = data.get('Question')
            if not question:
                return "Error: Question is required for General task."
            
            response = gemini_llm.invoke('Answer the question in detail: ' + question)
            return response.content

        elif task == 'question':
            question = data.get('Question')
            text = data.get('Body')
            if not question or not text:
                return "Error: Both Question and Editor text are required for Question and Answering."
            response = gemini_llm.invoke(f'Answer the question from the text: {question}?\n{text}')
            return response.content
        
        elif task == 'summary':
            text = data.get('Body')
            if not text:
                return "Error: Editor text is required for Summarization."
            response = gemini_llm.invoke('Summarize the text: ' + text)
            return response.content
        
        elif task == 'named':
            text = data.get('Body')
            if not text:
                return "Error: Editor text is required for Named Entity Recognition."
            response = gemini_llm.invoke('Extract named entities from the text: ' + text)
            return response.content

        elif task == 'topic':
            text = data.get('Body');
            if not text:
                return "Error: Editor text is required for Topic Modelling."
            response = gemini_llm.invoke('Extract topics from the text: ' + text)
            return response.content

        elif task == 'sentiment':
            text = data.get('Body')
            if not text:
                return "Error: Editor text is required for Sentiment Analysis."
            response = gemini_llm.invoke('Analyze the sentiment of the text: ' + text)
            return response.content

        elif task == 'emotion':
            text = data.get('Body')
            if not text:
                return "Error: Editor text is required for Emotion Recognition."
            response = gemini_llm.invoke('Analyze the emotion of the text: ' + text)
            return response.content

        elif task == 'translate':
            text = data.get('Body')
            language = data.get('Language')
            if not text:
                return "Error: Editor text is required for Machine Translation."
            if not language:
                return "Error: Please select a language for Machine Translation."
            response = gemini_llm.invoke(f'Translate the text to {language}: {text}')
            return response.content
        
        elif task == 'paraphrase':
            text = data.get('Body')
            if not text:
                return "Error: Editor text is required for Paraphrasing."
            response = gemini_llm.invoke('Paraphrase the text: ' + text)
            return response.content
        
        else:
            return "Error: Unsupported task type."
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Task can be general, question, summary, paraphrase, sentiment, emotion, named, topic, translate
def gpt(task, data):

    try:
        if task == 'general':
            question = data.get('Question', '')
            if not question:
                return "Error: Question is required for General task."
            
            messages = [
                ("system", "Answer the question in detail"),
                ("human", question)
            ]

            response = gpt_llm.invoke(messages)
            return response.content

        elif task == 'question':
            question = data.get('Question')
            text = data.get('Body')

            if not question or not text:
                return "Error: Both Question and Editor text are required for Question and Answering."
            
            messages = [
                ("system", f"Answer the question from the text: {question}?"),
                ("human", text)
            ]

            response = gpt_llm.invoke(messages)
            return response.content
        
        elif task == 'summary':
            text = data.get('Body')
            if not text:
                return "Error: Editor text is required for Summarization."
            
            messages = [
                ("system", "Summarize the text"),
                ("human", text)
            ]

            response = gpt_llm.invoke(messages)
            return response.content

        elif task == 'paraphrase':
            text = data.get('Body')
            if not text:
                return "Error: Editor text is required for Paraphrasing."
            
            messages = [
                ("system", "Paraphrase the text"),
                ("human", text)
            ]

            response = gpt_llm.invoke(messages)
            return response.content

        elif task == 'sentiment':
            text = data.get('Body')
            if not text:
                return "Error: Editor text is required for Sentiment Analysis."
            
            messages = [
                ("system", "Analyze the sentiment of the text"),
                ("human", text)
            ]
            response = gpt_llm.invoke(messages)
            return response.content

        elif task == 'emotion':
            text = data.get('Body')
            if not text:
                return "Error: Editor text is required for Emotion Recognition."
            
            messages = [
                ("system", "Analyze the emotion of the text"),
                ("human", text)
            ]

            response = gpt_llm.invoke(messages)
            return response.content

        elif task == 'named':
            text = data.get('Body')
            if not text:
                return "Error: Editor text is required for Named Entity Recognition."
            
            messages = [
                ("system", "Extract named entities from the text"),
                ("human", text)
            ]

            response = gpt_llm.invoke(messages)
            return response.content
        
        elif task == 'translate':
            text = data.get('Body')
            language = data.get('Language')
            if not text:
                return "Error: Editor text is required for Machine Translation."
            if not language:
                return "Error: Please select a language for Machine Translation."
            
            messages = [
                ("system", f"Translate the text to {language}"),
                ("human", text)
            ]

            response = gpt_llm.invoke(messages)
            return response.content
        
        elif task == 'topic':
            text = data.get('Body')
            if not text:
                return "Error: Editor text is required for Topic Modelling."
            
            messages = [
                ("system", "Extract topics from the text"),
                ("human", text)
            ]

            response = gpt_llm.invoke(messages)
            return response.content

        else:
            return "Error: Unsupported task type."
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Task can be general, question, summary, paraphrase, sentiment, emotion, named, topic, translate
def llama(task, data):

    try: 
        if task == 'general':
            text = data.get('Question')
            if not text:
                return "Error: Question is required for General task."
            
            system = "Answer the question in detail. If code is asked, always tell language in the beginning of the code."
            prompt = ChatPromptTemplate.from_messages([("system", system), ("human", text)])
            chain = prompt | llama_llm
            response = chain.invoke({})
            return response.content

        elif task == 'question':
            question = data.get('Question')
            text = data.get('Body')
            if not question or not text:
                return "Error: Both Question and Editor text are required for Question and Answering."
            
            system = f"Answer the question from the text: {question}?"
            prompt = ChatPromptTemplate.from_messages([("system", system), ("human", text)])
            chain = prompt | llama_llm
            response = chain.invoke({})
            return response.content
        
        elif task == 'summary':
            text = data.get('Body')
            if not text:
                return "Error: Editor text is required for Summarization."
            
            system = "Summarize the text."
            prompt = ChatPromptTemplate.from_messages([("system", system), ("human", text)])
            chain = prompt | llama_llm
            response = chain.invoke({})
            return response.content
        
        elif task == 'paraphrase':
            text = data.get('Body')
            if not text:
                return "Error: Editor text is required for Paraphrasing."
            
            system = "Paraphrase the text."
            prompt = ChatPromptTemplate.from_messages([("system", system), ("human", text)])
            chain = prompt | llama_llm
            response = chain.invoke({})
            return response.content

        elif task == 'sentiment':
            text = data.get('Body')
            if not text:
                return "Error: Editor text is required for Sentiment Analysis."
            
            system = "Analyze the sentiment of the text."
            prompt = ChatPromptTemplate.from_messages([("system", system), ("human", text)])
            chain = prompt | llama_llm
            response = chain.invoke({})
            return response.content

        elif task == 'emotion':
            text = data.get('Body')
            if not text:
                return "Error: Editor text is required for Emotion Recognition."
            
            system = "Analyze the emotion of the text."
            prompt = ChatPromptTemplate.from_messages([("system", system), ("human", text)])
            chain = prompt | llama_llm
            response = chain.invoke({})
            return response.content

        elif task == 'named':
            text = data.get('Body')
            if not text:
                return "Error: Editor text is required for Named Entity Recognition."
            
            system = "Extract named entities from the text."
            prompt = ChatPromptTemplate.from_messages([("system", system), ("human", text)])
            chain = prompt | llama_llm
            response = chain.invoke({})
            return response.content

        elif task == 'topic':
            text = data.get('Body')
            if not text:
                return "Error: Editor text is required for Topic Modelling."
            
            system = "Extract topics from the text."
            prompt = ChatPromptTemplate.from_messages([("system", system), ("human", text)])
            chain = prompt | llama_llm
            response = chain.invoke({})
            return response.content

        elif task == 'translate':
            text = data.get('Body')
            language = data.get('Language')
            if not text:
                return "Error: Editor text is required for Machine Translation."
            
            if not language:
                return "Error: Please select a language for Machine Translation."
            
            system = f"Translate the text to {language}."
            prompt = ChatPromptTemplate.from_messages([("system", system), ("human", text)])
            chain = prompt | llama_llm
            response = chain.invoke({})
            return response.content

        else:
            return "Error: Unsupported task type."

    except Exception as e:
        return f"An error occurred: {str(e)}"

@csrf_exempt
def langchain(request):

    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method, POST required'}, status=405)

    try:
        data = json.loads(request.body.decode('utf-8'))
        llm = data.get('LLM')
        task = data.get('Task')

        if not llm or not task:
            return JsonResponse({'error': 'Missing LLM or Task parameter'}, status=400)
        
        if llm == 'gemini':
            response = gemini(task, data)
        elif llm == 'gpt':
            response = gpt(task, data)
        elif llm == 'llama':
            response = llama(task, data)
        else:
            return JsonResponse({'error': 'Unsupported LLM'}, status=400)
        

        return JsonResponse({'message': 'The message was received. Thank you', 'answer': response})
    
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    
    except Exception as e:
        return JsonResponse({'error': f'An unexpected error occurred: {str(e)}'}, status=500)
