import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import sent_tokenize
from rouge import Rouge
import pandas as pd

def extractive_summary(text, model, tokenizer):
    sentences = sent_tokenize(text)
    inputs = tokenizer.batch_encode_plus(sentences, return_tensors='tf', max_length=512, truncation=True, padding='longest')
    outputs = model(inputs['input_ids'])
    sentence_embeddings = tf.reduce_mean(outputs[0], axis=1).numpy()
    similarity_matrix = cosine_similarity(sentence_embeddings)
    np.fill_diagonal(similarity_matrix, 0)
    sentence_scores = np.sum(similarity_matrix, axis=0)
    ranked_sentence = sorted(((sentence_scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # Calculate the number of sentences to return
    num_sentences = int(len(sentences) * 0.6)
    
    return ' '.join([s for (_, s) in ranked_sentence[:num_sentences]])

def summarize_reports(all_reports, model, tokenizer, rouge):
    data = []
    
    for i, report in enumerate(all_reports):
        summary = extractive_summary(report, model, tokenizer)
        try:
            scores = rouge.get_scores(summary, report)
            data.append([f'summary_{i}.txt', scores[0]['rouge-1']['f'], scores[0]['rouge-2']['f'], scores[0]['rouge-l']['f']])        
        except ValueError as e:
            print(f"Error calculating ROUGE score for file {i}: {e}")
            
    return data
