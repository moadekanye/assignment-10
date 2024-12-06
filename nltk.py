# question 1
import spacy
from collections import Counter
import re

# Load the spaCy model for English
nlp = spacy.load("en_core_web_sm")

def load_text(file_path):
    """Function to load text from a file."""
    with open(file_path, "r") as file:
        return file.read()

def preprocess_text(text):
    """Function to clean and preprocess the text."""
    # Remove non-alphabetic characters 
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def get_common_tokens(text, top_n=20):
    """Get the most common tokens in the text after tokenization, stemming, and lemmatization."""
    doc = nlp(text)
    
    # Tokenization and preprocessing
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    
    # Stemming
    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    
    # Calculate frequency of lemmatized tokens
    token_freq = Counter(lemmatized_tokens)
    
    # Get the top N common tokens
    common_tokens = token_freq.most_common(top_n)
    return common_tokens

def get_named_entities(text):
    """Extract named entities from the text using spaCy's NER."""
    doc = nlp(text)
    
    # Extract named entities
    named_entities = [ent.text for ent in doc.ents]
    
    # Return named entities and their counts
    return named_entities, len(named_entities)

def determine_subject_from_tokens(common_tokens, named_entities):
    """Determine the likely subject of the text based on common tokens and named entities."""
    subject = {}
  
    subject['common_tokens'] = [token[0] for token in common_tokens]
    subject['named_entities'] = named_entities
    
    # Heuristic: Based on common terms and named entities
    subject_keywords = ['data', 'technology', 'science', 'business', 'economy', 'government', 'health', 'politics']
    inferred_subject = [word for word in subject['common_tokens'] if word in subject_keywords]
    
    return subject, inferred_subject

# Load the three texts
text_1 = load_text("Text_1.txt")
text_2 = load_text("Text_2.txt")
text_3 = load_text("Text_3.txt")

# Preprocess the texts 
text_1 = preprocess_text(text_1)
text_2 = preprocess_text(text_2)
text_3 = preprocess_text(text_3)

# Get the most common tokens in each text
common_tokens_1 = get_common_tokens(text_1)
common_tokens_2 = get_common_tokens(text_2)
common_tokens_3 = get_common_tokens(text_3)

# Get named entities and counts for each text
named_entities_1, num_entities_1 = get_named_entities(text_1)
named_entities_2, num_entities_2 = get_named_entities(text_2)
named_entities_3, num_entities_3 = get_named_entities(text_3)

# Determine the subjects based on common tokens and named entities
subject_1, inferred_subject_1 = determine_subject_from_tokens(common_tokens_1, named_entities_1)
subject_2, inferred_subject_2 = determine_subject_from_tokens(common_tokens_2, named_entities_2)
subject_3, inferred_subject_3 = determine_subject_from_tokens(common_tokens_3, named_entities_3)

# Output results for all three texts
print(f"Text 1 - Common Tokens: {common_tokens_1}")
print(f"Text 1 - Named Entities: {named_entities_1} (Total: {num_entities_1})")
print(f"Text 1 - Inferred Subject: {inferred_subject_1}")

print(f"\nText 2 - Common Tokens: {common_tokens_2}")
print(f"Text 2 - Named Entities: {named_entities_2} (Total: {num_entities_2})")
print(f"Text 2 - Inferred Subject: {inferred_subject_2}")

print(f"\nText 3 - Common Tokens: {common_tokens_3}")
print(f"Text 3 - Named Entities: {named_entities_3} (Total: {num_entities_3})")
print(f"Text 3 - Inferred Subject: {inferred_subject_3}")

# question 2
import nltk
from nltk.util import ngrams
from collections import Counter
import re

nltk.download('punkt')

def load_text(file_path):
    with open(file_path, "r") as file:
        return file.read()

def preprocess_text(text):
    return re.sub(r'[^a-zA-Z\s]', '', text).lower()

def generate_ngrams(text, n=3):
    tokens = nltk.word_tokenize(text)
    return Counter(ngrams(tokens, n))

def print_most_common_ngrams(ngram_freq, top_n=20):
    return ngram_freq.most_common(top_n)

def compare_ngrams(ngram_freq_4, ngram_freq_others, top_n=20):
    common_with_others = {}
    for i, ngram_freq_other in enumerate(ngram_freq_others):
        common_ngrams = ngram_freq_4 & ngram_freq_other
        common_with_others[f"Text {i+1}"] = common_ngrams.most_common(top_n)
    return common_with_others

text_1 = load_text("Text_1.txt")
text_2 = load_text("Text_2.txt")
text_3 = load_text("Text_3.txt")
text_4 = load_text("Text_4.txt")

text_1 = preprocess_text(text_1)
text_2 = preprocess_text(text_2)
text_3 = preprocess_text(text_3)
text_4 = preprocess_text(text_4)

ngram_freq_1 = generate_ngrams(text_1, n=3)
ngram_freq_2 = generate_ngrams(text_2, n=3)
ngram_freq_3 = generate_ngrams(text_3, n=3)
ngram_freq_4 = generate_ngrams(text_4, n=3)

common_ngrams_1 = print_most_common_ngrams(ngram_freq_1)
common_ngrams_2 = print_most_common_ngrams(ngram_freq_2)
common_ngrams_3 = print_most_common_ngrams(ngram_freq_3)
common_ngrams_4 = print_most_common_ngrams(ngram_freq_4)

print("Text 1 - Most Common 3-grams:", common_ngrams_1)
print("Text 2 - Most Common 3-grams:", common_ngrams_2)
print("Text 3 - Most Common 3-grams:", common_ngrams_3)
print("Text 4 - Most Common 3-grams:", common_ngrams_4)

ngram_comparison = compare_ngrams(ngram_freq_4, [ngram_freq_1, ngram_freq_2, ngram_freq_3])

for text, common_ngrams in ngram_comparison.items():
    print(f"Common 3-grams with {text}:", common_ngrams)
