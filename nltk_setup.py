import nltk

try:
    nltk.data.find('punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('averaged_perceptron_tagger_eng')  # New download
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')  # New download

print("NLTK data downloaded.")