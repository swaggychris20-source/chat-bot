import json
import random
import numpy as np
from typing import List, Dict, Any
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-eng', quiet=True)

class SimpleChatBot:
    def __init__(self, intents_file: str = 'intents.json'):
        """Initialize the chatbot with intents from JSON file."""
        self.intents_file = intents_file
        self.lemmatizer = WordNetLemmatizer()
        self.intents = None
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_chars = ['?', '!', '.', ',']
        
        self.model = None
        self.label_encoder = LabelEncoder()
        
        self.load_intents()
        self.prepare_data()
        
    def load_intents(self):
        """Load intents from JSON file."""
        try:
            with open(self.intents_file, 'r', encoding='utf-8') as file:
                self.intents = json.load(file)['intents']
            print(f"✓ Loaded {len(self.intents)} intents from {self.intents_file}")
        except FileNotFoundError:
            print(f"✗ Error: File {self.intents_file} not found!")
            exit(1)
        except json.JSONDecodeError:
            print(f"✗ Error: Invalid JSON in {self.intents_file}!")
            exit(1)
    
    def clean_text(self, text: str) -> List[str]:
        """Tokenize and lemmatize text."""
        words = nltk.word_tokenize(text.lower())
        words = [self.lemmatizer.lemmatize(word) for word in words 
                if word not in self.ignore_chars]
        return words
    
    def prepare_data(self):
        """Prepare training data from intents."""
        for intent in self.intents:
            tag = intent['tag']
            if tag not in self.classes:
                self.classes.append(tag)
            
            for pattern in intent['patterns']:
                words = self.clean_text(pattern)
                self.words.extend(words)
                self.documents.append((words, tag))
        
        # Remove duplicates and sort
        self.words = sorted(set(self.words))
        self.classes = sorted(set(self.classes))
        
        print(f"✓ Prepared data: {len(self.words)} unique words, {len(self.classes)} classes")
    
    def create_training_data(self):
        """Create training data in bag-of-words format."""
        training = []
        output_empty = [0] * len(self.classes)
        
        for document in self.documents:
            bag = []
            pattern_words = document[0]
            
            # Create bag of words
            for word in self.words:
                bag.append(1) if word in pattern_words else bag.append(0)
            
            # Create output row
            output_row = list(output_empty)
            output_row[self.classes.index(document[1])] = 1
            
            training.append([bag, output_row])
        
        # Shuffle the training data
        random.shuffle(training)
        
        # Convert to numpy arrays
        train_x = np.array([item[0] for item in training])
        train_y = np.array([item[1] for item in training])
        
        return train_x, train_y
    
    def build_model(self, input_shape: int, output_shape: int):
        """Build and compile the neural network model."""
        model = keras.Sequential([
            layers.Dense(128, input_shape=(input_shape,), activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(output_shape, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, epochs: int = 200):
        """Train the chatbot model."""
        print("⚡ Preparing training data...")
        train_x, train_y = self.create_training_data()
        
        print("🧠 Building model...")
        self.model = self.build_model(len(train_x[0]), len(train_y[0]))
        
        print("🚀 Training model...")
        self.model.fit(
            train_x, train_y,
            epochs=epochs,
            batch_size=5,
            verbose=1
        )
        print("✓ Training completed!")
    
    def bag_of_words(self, sentence: str) -> np.ndarray:
        """Convert sentence to bag-of-words representation."""
        sentence_words = self.clean_text(sentence)
        bag = [0] * len(self.words)
        
        for word in sentence_words:
            if word in self.words:
                bag[self.words.index(word)] = 1
        
        return np.array(bag)
    
    def predict_intent(self, sentence: str, threshold: float = 0.25):
        """Predict the intent of a sentence."""
        bow = self.bag_of_words(sentence)
        if self.model is None:
            return None, 0
        
        results = self.model.predict(np.array([bow]), verbose=0)[0]
        
        # Filter predictions below threshold
        results = [[i, r] for i, r in enumerate(results) if r > threshold]
        
        if not results:
            return None, 0
        
        # Sort by probability
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return the highest probability intent
        intent_index = results[0][0]
        probability = results[0][1]
        
        return self.classes[intent_index], probability
    
    def get_response(self, intent_tag: str) -> str:
        """Get a random response for the given intent tag."""
        for intent in self.intents:
            if intent['tag'] == intent_tag:
                return random.choice(intent['responses'])
        return "I'm not sure how to respond to that."
    
    def chat(self):
        """Start an interactive chat session."""
        print("\n" + "="*50)
        print("🤖 ChatBot Activated!")
        print("Type 'quit', 'exit', or 'bye' to end the chat.")
        print("="*50 + "\n")
        
        # Train the model if not already trained
        if self.model is None:
            print("⏳ Model not trained. Training now...")
            self.train(epochs=100)
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("ChatBot: Goodbye! Have a great day! 👋")
                    break
                
                if not user_input:
                    continue
                
                # Predict intent
                intent, probability = self.predict_intent(user_input)
                
                if intent:
                    response = self.get_response(intent)
                    print(f"ChatBot: {response}")
                else:
                    print("ChatBot: I'm sorry, I don't understand. Could you rephrase that?")
                    
            except KeyboardInterrupt:
                print("\n\nChatBot: Goodbye! 👋")
                break
            except Exception as e:
                print(f"ChatBot: Sorry, I encountered an error. Please try again.")

# Alternative simpler version without machine learning
class RuleBasedChatBot:
    """A simpler rule-based chatbot for quick setup."""
    
    def __init__(self, intents_file: str = 'intents.json'):
        """Initialize the rule-based chatbot."""
        self.intents_file = intents_file
        self.intents = None
        self.lemmatizer = WordNetLemmatizer()
        
        self.load_intents()
    
    def load_intents(self):
        """Load intents from JSON file."""
        try:
            with open(self.intents_file, 'r', encoding='utf-8') as file:
                self.intents = json.load(file)['intents']
            print(f"✓ Loaded {len(self.intents)} intents")
        except FileNotFoundError:
            print(f"✗ Error: File {self.intents_file} not found!")
            exit(1)
    
    def clean_text(self, text: str) -> List[str]:
        """Clean and tokenize text."""
        words = nltk.word_tokenize(text.lower())
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return words
    
    def find_best_match(self, user_input: str) -> str:
        """Find the best matching intent using simple pattern matching."""
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
                    best_match = intent['tag']
        
        # Threshold for matching
        if highest_score > 0.3:
            return best_match
        return None
    
    def chat(self):
        """Start an interactive chat session."""
        print("\n" + "="*50)
        print("🤖 Rule-Based ChatBot Activated!")
        print("Type 'quit', 'exit', or 'bye' to end the chat.")
        print("="*50 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("ChatBot: Goodbye! 👋")
                    break
                
                if not user_input:
                    continue
                
                # Find matching intent
                intent = self.find_best_match(user_input)
                
                if intent:
                    for int_data in self.intents:
                        if int_data['tag'] == intent:
                            response = random.choice(int_data['responses'])
                            print(f"ChatBot: {response}")
                            break
                else:
                    print("ChatBot: I'm not sure I understand. Can you rephrase?")
                    
            except KeyboardInterrupt:
                print("\n\nChatBot: Goodbye! 👋")
                break

def main():
    """Main function to run the chatbot."""
    print("Select chatbot type:")
    print("1. Machine Learning ChatBot (needs training)")
    print("2. Rule-Based ChatBot (instant)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        print("\n" + "="*50)
        print("🤖 Machine Learning ChatBot")
        print("="*50)
        bot = SimpleChatBot('intents.json')
        bot.chat()
    elif choice == "2":
        print("\n" + "="*50)
        print("🤖 Rule-Based ChatBot")
        print("="*50)
        bot = RuleBasedChatBot('intents.json')
        bot.chat()
    else:
        print("Invalid choice. Using Rule-Based ChatBot by default.")
        bot = RuleBasedChatBot('intents.json')
        bot.chat()

if __name__ == "__main__":
    main()