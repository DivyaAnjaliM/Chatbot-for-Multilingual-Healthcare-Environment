import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import genai
import os
class GenerativeModel:
    def __init__(self):
        # Load the pre-trained Pegasus model and tokenizer
        self.tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')
        self.model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-large')
        api_key = os.getenv("AIzaSyAJONLVfr744J2u3OEyNW5mMrESRuQdCUg")  # Set this in your environment variables
        genai.configure(api_key=api_key)
    def summarize(self, text):
        """
        Summarizes the given text using the Pegasus model.
        
        Parameters:
            text (str): The text to summarize.
        
        Returns:
            str: The summarized text.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        
        # Generate the summary
        summary_ids = self.model.generate(
            inputs['input_ids'],
            max_length=150,  # Maximum length of summary
            min_length=30,  # Minimum length of summary
            length_penalty=2.0,  # Penalty for longer summaries
            num_beams=4,  # Number of beams for beam search
            early_stopping=True  # Stop early if the summary is good enough
        )
        
        # Decode the summary and return it
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def generate_answer(self, question, context):
        """
        Generate an answer to a question based on the context using the Pegasus model.
        
        Parameters:
            question (str): The question to answer.
            context (str): The context or document that provides information for answering.
        
        Returns:
            str: The generated answer.
        """
        # Format the prompt
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        
        # Tokenize the input prompt
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, padding=True)
        
        # Generate the answer
        answer_ids = self.model.generate(
            inputs['input_ids'],
            max_length=150,  # Maximum length for the answer
            min_length=30,  # Minimum length for the answer
            length_penalty=2.0,  # Penalty for longer answers
            num_beams=4,  # Number of beams for beam search
            early_stopping=True  # Stop early if the answer is good enough
        )
        
        # Decode the answer and return it
        answer = self.tokenizer.decode(answer_ids[0], skip_special_tokens=True)
        return answer


if __name__ == "__main__":
    # Sample usage of the GenerativeModel class
    generative_model = GenerativeModel()
    
    # Example of text summarization
    sample_report = """
    The patient was admitted with acute chest pain. Initial tests showed elevated troponin levels and an ECG showed
    signs of myocardial infarction. The patient was given aspirin and nitroglycerin and is being monitored in ICU.
    Further tests are scheduled to confirm the diagnosis and assess the patient's condition.
    """
    
    summary = generative_model.summarize(sample_report)
    print("Summary:", summary)
    
    # Example of question answering
    question = "What is the diagnosis for the patient?"
    context = sample_report
    answer = generative_model.generate_answer(question, context)
    print("Answer:", answer)
