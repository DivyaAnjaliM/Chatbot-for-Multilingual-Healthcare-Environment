import re

def clean_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_sample_report(file_path):
    with open(file_path, 'r') as file:
        return file.read()

if __name__ == "__main__":
    sample_text = load_sample_report('data/sample_report.txt')
    cleaned_text = clean_text(sample_text)
    print(cleaned_text)
