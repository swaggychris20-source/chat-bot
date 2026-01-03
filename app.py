from flask import Flask, render_template, request, jsonify
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class WebChatBot:
    def __init__(self, intents_file='intents.json'):
        self.intents_file = intents_file
        self.lemmatizer = WordNetLemmatizer()
        self.intents = self.load_intents()
    
    def load_intents(self):
        with open(self.intents_file, 'r', encoding='utf-8') as file:
            return json.load(file)['intents']
    
    def clean_text(self, text):
        words = nltk.word_tokenize(text.lower())
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return words
    
    def get_response(self, user_input):
        user_words = set(self.clean_text(user_input))
        best_match = None
        highest_score = 0
        
        for intent in self.intents:
            for pattern in intent['patterns']:
                pattern_words = set(self.clean_text(pattern))
                common_words = user_words.intersection(pattern_words)
                score = len(common_words) / len(pattern_words) if pattern_words else 0
                
                if score > highest_score:
                    highest_score = score
                    best_match = intent
        
        if highest_score > 0.3 and best_match:
            return random.choice(best_match['responses'])
        
        return "I'm not sure I understand. Could you rephrase that?"

chatbot = WebChatBot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'response': 'Please enter a message.'})
    
    if user_message.lower() in ['quit', 'exit', 'bye', 'goodbye']:
        return jsonify({'response': 'Goodbye! Have a great day! 👋'})
    
    response = chatbot.get_response(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    print("🌐 Starting Web ChatBot...")
    print("📡 Open http://localhost:5000 in your browser")
    app.run(debug=True)