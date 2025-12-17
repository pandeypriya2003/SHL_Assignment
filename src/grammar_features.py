import language_tool_python
import nltk

nltk.download('punkt')
tool = language_tool_python.LanguageTool('en-US')

def extract_grammar_features(text):
    if not text.strip():
        return [0, 0, 0, 0]

    errors = len(tool.check(text))
    words = text.split()
    sentences = nltk.sent_tokenize(text)

    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    error_density = errors / len(words) if words else 0

    return [errors, len(words), avg_sentence_length, error_density]
