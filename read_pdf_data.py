import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pypdf import PdfReader

def read_pdf_text_pypdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
    except FileNotFoundError:
        print(f"Error: File not found")
    except Exception as e:
        print(f"An error occurred: {e}")
    return text

import nltk
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    word_tokenize("example")
except LookupError:
    nltk.download('punkt')

def clean_pdf_text(text):
    """Cleans the extracted text from a PDF."""

    # 1. Remove leading/trailing whitespace
    text = text.strip()

    # 2. Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # 3. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 4. Convert to lowercase
    text = text.lower()

    # # 5. Remove stop words (English)
    # stop_words = set(stopwords.words('english'))
    # word_tokens = word_tokenize(text)
    # text = " ".join([w for w in word_tokens if not w in stop_words])

    return text

pdf_file_path = 'GCP_Document_E6_R2_Addendum.pdf'  # Replace with your PDF path
extracted_text = read_pdf_text_pypdf(pdf_file_path)

if extracted_text:
    cleaned_data = clean_pdf_text(extracted_text)
    print("Cleaned Text:\n")
    print(cleaned_data)
else:
    print("No text to clean.")