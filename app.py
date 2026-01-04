from flask import Flask, render_template, request, jsonify
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import os

app = Flask(__name__)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

class WebChatBot:
    def __init__(self, intents_file='intents.json'):
        self.intents_file = intents_file
        self.lemmatizer = WordNetLemmatizer()
        self.intents = self.load_intents()
    
    def load_intents(self):
        try:
            with open(self.intents_file, 'r', encoding='utf-8') as file:
                return json.load(file).get('intents', [])
        except FileNotFoundError:
            print(f"Warning: {self.intents_file} not found. Using default intents.")
            return self.get_default_intents()
        except json.JSONDecodeError:
            print(f"Warning: {self.intents_file} is not valid JSON. Using default intents.")
            return self.get_default_intents()
    
    def get_default_intents(self):
        return [
            {
                "tag": "greeting",
                "patterns": ["hello", "hi", "hey", "good morning", "good evening"],
                "responses": ["Hello! 👋 How can I help you today?", "Hi there! What can I do for you?"]
            },
            {
                "tag": "about",
                "patterns": ["who are you", "what are you", "tell me about yourself"],
                "responses": ["I'm SWAGGBOT, an AI chatbot designed to help you with various tasks!"]
            },
            {
                "tag": "help",
                "patterns": ["help", "can you help", "what can you do"],
                "responses": ["I can answer questions, help with creative writing, assist with coding, and much more! Try asking me anything."]
            }
        ]
    
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
                score = len(common_words) / max(len(pattern_words), 1)
                
                if score > highest_score:
                    highest_score = score
                    best_match = intent
        
        if highest_score > 0.3 and best_match:
            return random.choice(best_match['responses'])
        
        fallback_responses = [
            "I'm not sure I understand. Could you rephrase that?",
            "That's an interesting question! Could you provide more details?",
            "I'm still learning! Could you ask me something else?",
            "I don't have enough information about that. Try asking me something different!"
        ]
        return random.choice(fallback_responses)

chatbot = WebChatBot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'response': 'No data received'}), 400
        
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'response': 'Please enter a message.'}), 400
        
        exit_words = ['quit', 'exit', 'bye', 'goodbye', 'stop', 'end']
        if user_message.lower() in exit_words:
            return jsonify({'response': 'Goodbye! Have a great day! 👋'})
        
        response = chatbot.get_response(user_message)
        return jsonify({'response': response})
    
    except Exception as e:
        print(f"Error processing chat request: {e}")
        return jsonify({'response': 'Sorry, I encountered an error. Please try again.'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    print("🌐 Starting SWAGGBOT...")
    print("📡 Open http://localhost:5000 in your browser")
    app.run(
        debug=True,
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000))
    )