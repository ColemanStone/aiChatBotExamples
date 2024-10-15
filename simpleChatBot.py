import nltk
from nltk.chat.util import Chat, reflections
# Define a list of patterns and responses
patterns = [
 (r"hi|hello|hey", ["Hello!", "Hi there!", "Hey!"]),
 (r"how are you?", ["I'm doing well, how about you?"]),
 (r"what is your name?", ["I'm a chatbot created just for you!"]),
 (r"quit", ["Goodbye! Have a great day!"]),
]
# Create the chatbot
chatbot = Chat(patterns, reflections)
# Start the chatbot
print("Chatbot: Hello! Type 'quit' to exit.")
chatbot.converse()
