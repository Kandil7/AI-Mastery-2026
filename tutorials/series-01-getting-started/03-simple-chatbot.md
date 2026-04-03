# 💬 Tutorial 3: Build a Simple Chatbot

**Create your first AI chatbot from scratch in 60 minutes**

---

## 🎯 What You'll Build

By the end of this tutorial, you will have:

- ✅ Built a rule-based chatbot from scratch
- ✅ Implemented natural language processing basics
- ✅ Created a conversational AI with memory
- ✅ Added sentiment analysis
- ✅ Deployed an interactive chat interface
- ✅ Tested with real conversations

**Time Required:** 60 minutes  
**Difficulty:** ⭐⭐☆☆☆ (Beginner)  
**Prerequisites:** Tutorial 1 & 2

---

## 📋 What You'll Learn

- How chatbots work
- Rule-based vs AI chatbots
- Pattern matching for intent detection
- Response generation
- Conversation state management
- Basic NLP concepts

---

## 🛠️ Step 1: Setup (5 minutes)

### Create Project File

```python
# Create new file: simple_chatbot.py
# This will be a complete chatbot application
```

### Import Libraries

```python
import re
import random
from datetime import datetime
from typing import Optional

# For sentiment analysis (optional, install if needed)
# pip install textblob
try:
    from textblob import TextBlob
    HAS_SENTIMENT = True
except ImportError:
    HAS_SENTIMENT = False
    print("💡 Tip: Install textblob for sentiment analysis: pip install textblob")

print("✅ Chatbot libraries loaded!")
```

---

## 🧠 Step 2: Understand Chatbot Architecture (5 minutes)

### How Chatbots Work

```
User Input → Intent Detection → Response Selection → Generate Response → Output
```

### Types of Chatbots

1. **Rule-Based** (What we're building)
   - Pattern matching
   - Predefined responses
   - Fast and predictable

2. **AI-Powered** (Advanced)
   - Machine learning models
   - Understands context
   - Learns from conversations

### Our Approach

We'll build a **hybrid chatbot**:
- Rule-based pattern matching
- Simple sentiment analysis
- Conversation memory
- Fallback responses

---

## 🎨 Step 3: Build the Chatbot Core (15 minutes)

### Define Response Patterns

```python
class SimpleChatbot:
    """A rule-based chatbot with pattern matching."""
    
    def __init__(self):
        """Initialize chatbot with patterns and responses."""
        self.name = "AI Assistant"
        self.user_name = None
        self.conversation_history = []
        
        # Define patterns and responses
        self.patterns = {
            'greeting': {
                'patterns': [
                    r'(?i)\b(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b',
                ],
                'responses': [
                    "Hello! How can I help you today?",
                    "Hi there! What can I do for you?",
                    "Hey! Nice to see you. How can I assist?",
                    "Greetings! What brings you here today?",
                ]
            },
            'name': {
                'patterns': [
                    r'(?i)\b(my name is|i am|i\'m|call me)\b\s+(\w+)',
                ],
                'responses': [
                    "Nice to meet you, {name}! How can I help you?",
                    "Hello {name}! What can I do for you today?",
                    "Great to meet you, {name}! How can I assist?",
                ]
            },
            'help': {
                'patterns': [
                    r'(?i)\b(help|support|assist|what can you do|capabilities)\b',
                ],
                'responses': [
                    "I can help you with:\n- Answering questions\n- Having conversations\n- Providing information\n- Just chatting! What would you like to do?",
                    "I'm here to help! I can:\n- Chat with you\n- Answer simple questions\n- Tell you about myself\nWhat interests you?",
                ]
            },
            'time': {
                'patterns': [
                    r'(?i)\b(what time|current time|time is it|tell me the time)\b',
                ],
                'responses': [
                    "The current time is {time}.",
                ]
            },
            'date': {
                'patterns': [
                    r'(?i)\b(what date|today\'s date|what day|today is)\b',
                ],
                'responses': [
                    "Today is {date}.",
                ]
            },
            'joke': {
                'patterns': [
                    r'(?i)\b(tell me a joke|joke|make me laugh|something funny)\b',
                ],
                'responses': [
                    "Why do programmers prefer dark mode? Because light attracts bugs! 😄",
                    "Why do Java developers wear glasses? Because they can't C#! 😂",
                    "What's a programmer's favorite hangout place? Foo Bar! 🍺",
                    "Why was the JavaScript developer sad? Because he didn't Node how to Express himself! 😅",
                ]
            },
            'thanks': {
                'patterns': [
                    r'(?i)\b(thank you|thanks|appreciate it|thank you very much)\b',
                ],
                'responses': [
                    "You're welcome! Is there anything else I can help with?",
                    "Happy to help! Let me know if you need anything else.",
                    "No problem at all! What else can I do for you?",
                ]
            },
            'goodbye': {
                'patterns': [
                    r'(?i)\b(bye|goodbye|see you|take care|have a good day|exit|quit)\b',
                ],
                'responses': [
                    "Goodbye! Have a great day! 👋",
                    "See you later! Take care! 👋",
                    "Bye! Feel free to come back anytime! 👋",
                ]
            },
            'about': {
                'patterns': [
                    r'(?i)\b(who are you|what are you|tell me about yourself|your name)\b',
                ],
                'responses': [
                    f"I'm {self.name}, a simple rule-based chatbot! I'm here to help answer your questions and have conversations.",
                    f"My name is {self.name}. I'm a beginner chatbot built with Python. I'm learning to have conversations!",
                ]
            },
            'python': {
                'patterns': [
                    r'(?i)\b(python|learn python|python programming|python tutorial)\b',
                ],
                'responses': [
                    "Python is a great language for beginners! It's readable, versatile, and widely used in AI/ML. Check out python.org to get started!",
                    "Python is perfect for AI development! It has libraries like NumPy, Pandas, and scikit-learn. What would you like to learn about Python?",
                ]
            },
            'ai': {
                'patterns': [
                    r'(?i)\b(artificial intelligence|machine learning|deep learning|AI|ML)\b',
                ],
                'responses': [
                    "AI is fascinating! It ranges from simple rule-based systems (like me!) to complex neural networks. What aspect interests you?",
                    "Machine Learning is a subset of AI where computers learn from data. It's used in everything from recommendations to self-driving cars!",
                ]
            },
        }
        
        # Fallback responses
        self.fallback_responses = [
            "I'm not sure I understand. Could you rephrase that?",
            "Interesting! Can you tell me more about that?",
            "I'm still learning. Could you try asking differently?",
            "Hmm, I don't have a specific response for that. Try asking about Python, AI, or just say 'help'!",
            "I'm a simple chatbot, so I might not understand everything. Try 'help' to see what I can do!",
        ]
    
    def get_response(self, user_input: str) -> str:
        """
        Get response based on user input.
        
        Args:
            user_input: User's message
        
        Returns:
            Chatbot's response
        """
        # Add to conversation history
        self.conversation_history.append({
            'user': user_input,
            'timestamp': datetime.now()
        })
        
        # Check patterns
        for intent, data in self.patterns.items():
            for pattern in data['patterns']:
                match = re.search(pattern, user_input)
                if match:
                    # Get random response
                    response = random.choice(data['responses'])
                    
                    # Handle special cases
                    if intent == 'name' and match.group(2):
                        self.user_name = match.group(2)
                        response = response.format(name=self.user_name)
                    elif intent == 'time':
                        response = response.format(time=datetime.now().strftime("%I:%M %p"))
                    elif intent == 'date':
                        response = response.format(date=datetime.now().strftime("%A, %B %d, %Y"))
                    
                    return response
        
        # No pattern matched - use fallback
        return random.choice(self.fallback_responses)
```

---

## 💬 Step 4: Test the Chatbot (10 minutes)

### Interactive Testing

```python
# Create chatbot instance
bot = SimpleChatbot()

print("=" * 60)
print(f"💬 {bot.name} - Interactive Chat")
print("=" * 60)
print("Type 'quit' or 'exit' to end the conversation\n")

# Chat loop
while True:
    user_input = input("You: ").strip()
    
    if not user_input:
        continue
    
    # Get response
    response = bot.get_response(user_input)
    
    print(f"\n{bot.name}: {response}\n")
    
    # Check for exit
    if re.search(r'(?i)\b(bye|goodbye|exit|quit)\b', user_input):
        break

print("\nThanks for chatting! 👋")
```

### Test Different Intents

Try these inputs:
- "Hello"
- "My name is [Your Name]"
- "What can you do?"
- "Tell me a joke"
- "What time is it?"
- "Tell me about Python"
- "What is AI?"
- "Thank you"
- "Goodbye"

---

## 🎭 Step 5: Add Sentiment Analysis (10 minutes)

### What is Sentiment Analysis?

Sentiment analysis determines if text is **positive**, **negative**, or **neutral**.

### Implement Sentiment Detection

```python
class AdvancedChatbot(SimpleChatbot):
    """Enhanced chatbot with sentiment analysis."""
    
    def __init__(self):
        super().__init__()
        self.name = "AI Assistant Pro"
        self.user_mood = []
    
    def analyze_sentiment(self, text: str) -> str:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Sentiment: positive, negative, or neutral
        """
        if HAS_SENTIMENT:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return "positive"
            elif polarity < -0.1:
                return "negative"
            else:
                return "neutral"
        else:
            # Simple keyword-based sentiment
            positive_words = ['good', 'great', 'awesome', 'amazing', 'love', 'happy', 'excellent']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'sad', 'angry', 'worst']
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                return "positive"
            elif neg_count > pos_count:
                return "negative"
            else:
                return "neutral"
    
    def get_response(self, user_input: str) -> str:
        """Get response with sentiment awareness."""
        # Analyze sentiment
        sentiment = self.analyze_sentiment(user_input)
        self.user_mood.append(sentiment)
        
        # Get base response
        response = super().get_response(user_input)
        
        # Add sentiment-aware responses
        if sentiment == "negative" and random.random() > 0.5:
            empathetic_responses = [
                "I'm sorry you're feeling that way. Is there anything I can do to help?",
                "That sounds tough. I'm here if you need to talk.",
                "I understand. Let me know how I can make things better!",
            ]
            response = random.choice(empathetic_responses)
        elif sentiment == "positive" and random.random() > 0.7:
            positive_responses = [
                "That's great to hear! 😊",
                "Awesome! Keep up the positive energy! 🎉",
                "Glad to hear you're doing well! 😄",
            ]
            response += " " + random.choice(positive_responses)
        
        return response
    
    def get_conversation_summary(self) -> dict:
        """Get summary of conversation."""
        if not self.user_mood:
            return {"status": "No conversations yet"}
        
        from collections import Counter
        mood_counts = Counter(self.user_mood)
        
        return {
            "total_messages": len(self.conversation_history),
            "mood_distribution": dict(mood_counts),
            "overall_mood": mood_counts.most_common(1)[0][0] if mood_counts else "unknown",
        }
```

### Test Sentiment Analysis

```python
# Create advanced chatbot
adv_bot = AdvancedChatbot()

# Test sentiments
test_inputs = [
    "I'm having a great day!",
    "This is terrible and frustrating",
    "The weather is okay today",
    "I love learning about AI!",
    "I'm feeling sad and lonely",
]

print("🎭 Sentiment Analysis Test")
print("=" * 60)

for text in test_inputs:
    sentiment = adv_bot.analyze_sentiment(text)
    print(f"\nText: {text}")
    print(f"Sentiment: {sentiment}")

# Get summary
summary = adv_bot.get_conversation_summary()
print(f"\n📊 Conversation Summary:")
print(summary)
```

---

## 🧠 Step 6: Add Conversation Memory (10 minutes)

### Why Memory Matters

Without memory, chatbots forget everything between messages. Memory enables:
- Context-aware responses
- Personalization
- Better user experience

### Implement Memory

```python
class MemoryChatbot(AdvancedChatbot):
    """Chatbot with conversation memory and context."""
    
    def __init__(self):
        super().__init__()
        self.name = "AI Assistant with Memory"
        self.user_preferences = {}
        self.topics_discussed = []
    
    def get_response(self, user_input: str) -> str:
        """Get response with memory and context."""
        # Extract topics
        topics = self._extract_topics(user_input)
        self.topics_discussed.extend(topics)
        
        # Check for context-aware responses
        context_response = self._check_context(user_input)
        if context_response:
            return context_response
        
        # Get base response
        response = super().get_response(user_input)
        
        return response
    
    def _extract_topics(self, text: str) -> list:
        """Extract key topics from text."""
        keywords = ['python', 'ai', 'machine learning', 'programming', 'data', 
                   'web development', 'career', 'learning', 'projects']
        
        text_lower = text.lower()
        return [kw for kw in keywords if kw in text_lower]
    
    def _check_context(self, user_input: str) -> Optional[str]:
        """Check for context-aware responses."""
        text_lower = user_input.lower()
        
        # Remember user name
        if self.user_name and 'my name' in text_lower:
            return f"Your name is {self.user_name}! How can I help you today?"
        
        # Reference previous topics
        if self.topics_discussed and 'what were we talking' in text_lower:
            unique_topics = list(set(self.topics_discussed[-5:]))
            return f"We've been discussing: {', '.join(unique_topics)}"
        
        # Personalized greeting
        if self.user_name and any(word in text_lower for word in ['hi', 'hello', 'hey']):
            return f"Welcome back, {self.user_name}! What would you like to talk about?"
        
        return None
    
    def get_user_profile(self) -> dict:
        """Get user profile based on conversation."""
        return {
            "name": self.user_name,
            "topics_discussed": list(set(self.topics_discussed)),
            "message_count": len(self.conversation_history),
            "overall_mood": self.user_mood[-1] if self.user_mood else "unknown",
        }
```

### Test Memory Features

```python
# Create memory chatbot
mem_bot = MemoryChatbot()

print("🧠 Memory Chatbot Test")
print("=" * 60)

# Simulate conversation
conversation = [
    "Hello",
    "My name is Alex",
    "I'm interested in learning Python",
    "What were we talking about?",
    "Hi again",
]

for msg in conversation:
    response = mem_bot.get_response(msg)
    print(f"\nYou: {msg}")
    print(f"Bot: {response}")

# Get user profile
profile = mem_bot.get_user_profile()
print(f"\n👤 User Profile:")
print(profile)
```

---

## 🌐 Step 7: Create Web Interface (10 minutes)

### Build Simple Web Chat

```python
# Create file: web_chat.py
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)
bot = MemoryChatbot()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Chatbot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .chat-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        #messages {
            height: 400px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
        }
        .user {
            background: #007bff;
            color: white;
            text-align: right;
        }
        .bot {
            background: #e9ecef;
            color: black;
        }
        input {
            width: 80%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        button {
            padding: 10px 20px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background: #0056b3;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <h2>💬 AI Chatbot</h2>
        <div id="messages"></div>
        <input type="text" id="userInput" placeholder="Type your message..." 
               onkeypress="if(event.key==='Enter') sendMessage()">
        <button onclick="sendMessage()">Send</button>
    </div>
    
    <script>
        function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            if (!message) return;
            
            // Add user message
            addMessage(message, 'user');
            input.value = '';
            
            // Get bot response
            fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: message})
            })
            .then(response => response.json())
            .then(data => addMessage(data.response, 'bot'));
        }
        
        function addMessage(text, sender) {
            const messages = document.getElementById('messages');
            const div = document.createElement('div');
            div.className = `message ${sender}`;
            div.textContent = text;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    response = bot.get_response(message)
    return jsonify({'response': response})

if __name__ == '__main__':
    print("🌐 Starting web chatbot...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
```

### Run Web Interface

```bash
# Install Flask if needed
pip install flask

# Run the web chat
python web_chat.py

# Open browser to http://localhost:5000
```

---

## ✅ Tutorial Checklist

- [ ] Built rule-based chatbot
- [ ] Implemented pattern matching
- [ ] Added sentiment analysis
- [ ] Created conversation memory
- [ ] Built web interface
- [ ] Tested with real conversations

---

## 🎓 Key Takeaways

1. **Pattern Matching:** Use regex to detect user intents
2. **Response Selection:** Random responses for variety
3. **Sentiment Analysis:** Understand user mood
4. **Memory:** Track context for better conversations
5. **Web Interface:** Make chatbot accessible

---

## 🚀 Next Steps

1. **Continue to Tutorial 4:** [Introduction to RAG](04-intro-to-rag.md)
2. **Enhance Your Chatbot:**
   - Add more patterns and responses
   - Integrate with LLM API
   - Deploy to cloud
   - Add voice support

---

## 💡 Challenge (Optional)

**Make your chatbot smarter!**

1. Add 10+ new intent patterns
2. Implement conversation state machine
3. Add user authentication
4. Create admin dashboard
5. Deploy to Heroku/Railway

**Share your enhanced chatbot in Discord!** 🤖

---

**Tutorial Created:** April 2, 2026  
**Last Updated:** April 2, 2026  
**Estimated Time:** 60 minutes  
**Difficulty:** Beginner

---

[← Back to Tutorials](../README.md) | [Previous: First ML Model](02-first-ml-model.md) | [Next: Intro to RAG](04-intro-to-rag.md)
