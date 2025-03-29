from flask import Flask, render_template, request, jsonify
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# --- NLTK Downloads (Ensure these are downloaded) ---
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')

# --- Data Loading and Preprocessing (Module 1 equivalent) ---
# This is done *once* when the app starts
csv_file_path = 'large_questions.csv'  # Ensure this is correct!

def prepare_data(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path)
        print("CSV file successfully loaded.")
    except FileNotFoundError:
        print(f"Error: File not found at {csv_file_path}")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
        print("Please check the file path, file format, and ensure the file is not corrupted.")
        return None, None, None, None

    try:
        questions = df['Question'].tolist()
        answers = df['Answer'].tolist()
    except KeyError:
        print("Error: The CSV file must have columns named 'Question' and 'Answer'.")
        return None, None, None, None

    def tokenize_questions(questions):
        return [word_tokenize(question) for question in questions]

    def remove_stopwords(tokenized_questions):
        stop_words = set(stopwords.words('english'))
        return [[token.lower() for token in tokens if token.lower() not in stop_words] for tokens in tokenized_questions]

    def lemmatize_questions(filtered_questions):
        lemmatizer = WordNetLemmatizer()
        lemmatized_questions = []
        for tokens in filtered_questions:
            lemmatized_token_list = []
            tagged_tokens = nltk.pos_tag(tokens)
            for token, tag in tagged_tokens:
                pos = 'a' if tag.startswith('J') else \
                      'v' if tag.startswith('V') else \
                      'n' if tag.startswith('N') else \
                      'r' if tag.startswith('R') else 'n'
                lemmatized_token = lemmatizer.lemmatize(token, pos=pos)
                lemmatized_token_list.append(lemmatized_token)
            lemmatized_questions.append(lemmatized_token_list)
        return lemmatized_questions

    tokenized_questions = tokenize_questions(questions)
    filtered_questions = remove_stopwords(tokenized_questions)
    lemmatized_questions = lemmatize_questions(filtered_questions)

    vectorizer = TfidfVectorizer()
    try:
        vectorized_questions = vectorizer.fit_transform([" ".join(q) for q in lemmatized_questions])
    except ValueError:
        print("Error: The dataset is empty or contains very short questions. Vectorization failed.")
        return None, None, None, None

    print("Module 1: Data Preprocessing complete.")
    return vectorized_questions, questions, answers, vectorizer

vectorized_questions, questions, answers, vectorizer = prepare_data(csv_file_path)

if vectorized_questions is None:
    print("Error: Data preparation failed. Exiting.")
    exit()

# --- Flask Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_response", methods=['POST'])
def get_response():
    user_message = request.form['user_message']

    def preprocess_question(question):
        tokens = word_tokenize(question)
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [
            token.lower() for token in tokens if token.lower() not in stop_words
        ]
        lemmatizer = WordNetLemmatizer()
        tagged_tokens = nltk.pos_tag(filtered_tokens)
        return " ".join([
            lemmatizer.lemmatize(token, pos='a' if tag.startswith('J') else
                                 'v' if tag.startswith('V') else
                                 'n' if tag.startswith('N') else
                                 'r' if tag.startswith('R') else 'n')
            for token, tag in tagged_tokens
        ])

    processed_user_question = preprocess_question(user_message)

    try:
        vectorized_user_question = vectorizer.transform([processed_user_question])
        similarities = cosine_similarity(vectorized_user_question, vectorized_questions)
        most_similar_index = np.argmax(similarities)
        similarity_score = similarities[0][most_similar_index]

        if similarity_score > 0.6:  # Adjust this threshold as needed
            chatbot_response = answers[most_similar_index]
        else:
            chatbot_response = "I'm sorry, I couldn't find a close enough answer to that question."

        return jsonify({'response': chatbot_response})

    except ValueError:
        return jsonify({'response': "I couldn't process that question. Please try rephrasing."})

if __name__ == "__main__":
    app.run(debug=True)