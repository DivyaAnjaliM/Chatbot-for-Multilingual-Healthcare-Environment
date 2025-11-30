import pandas as pd
import os
import re
import PyPDF2
from sklearn.model_selection import train_test_split

def load_data(file_path):
    # Handle text files
    if file_path.endswith('.txt'):
        with open(file_path, 'r') as file:
            return file.read()
    
    # Handle PDF files
    elif file_path.endswith('.pdf'):
        pdf_text = ''
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in range(len(reader.pages)):
                pdf_text += reader.pages[page].extract_text()
        return pdf_text

def clean_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

def preprocess_reports(raw_dir, processed_dir):
    reports = []
    for filename in os.listdir(raw_dir):
        if filename.endswith('.txt') or filename.endswith('.pdf'):
            report = load_data(os.path.join(raw_dir, filename))
            cleaned_report = clean_text(report)
            reports.append(cleaned_report)
    
    # Simulated labels for demonstration (e.g., binary classification: 0 or 1)
    labels = [0 if i % 2 == 0 else 1 for i in range(len(reports))]  # Replace with actual labeling logic

    # Create DataFrame
    df = pd.DataFrame({'text': reports, 'label': labels})

    # Split into train and test datasets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save datasets
    df.to_csv(os.path.join(processed_dir, 'cleaned_reports.csv'), index=False)
    train_df.to_csv(os.path.join(processed_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(processed_dir, 'test.csv'), index=False)

if __name__ == "__main__": 
    raw_data_directory = r'C:\medical-report--summarizer\data\raw'  # Update with actual path to raw data directory
    processed_data_directory = r'C:\medical-report--summarizer\data\processed'  # Update with actual path to processed data directory
    preprocess_reports(raw_data_directory, processed_data_directory)
