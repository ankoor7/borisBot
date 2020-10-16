import re

# Maximum number of samples to preprocess
MAX_SAMPLES = 50000

def preprocess_sentence(sentence):
  sentence = sentence.lower().strip()
  # because hon. stands for honorable,
  sentence = re.sub(r"(hon\.)", r" honorable ", sentence)
  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)
  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  sentence = re.sub(r"[^a-zA-Z?.!',]+", " ", sentence)
  sentence = sentence.strip()
  # adding a start and an end token to the sentence
  return sentence


def load_conversations(interactions):
    answers = [preprocess_sentence(interaction['text']) for interaction in interactions]
    questions = [preprocess_sentence(interaction['responding_to_text']) for interaction in interactions]
    return questions, answers

