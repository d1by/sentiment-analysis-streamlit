import re

def preprocessor(text):
    return re.sub(r'[^a-z ]', '', text.lower())
