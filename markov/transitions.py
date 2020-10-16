import math

from parseDebate.Debates import Debates
from numpy import random

class WordStore:
    def __init__(self):
        self.word_store = {}

    def get_word(self, word):
        if word not in self.word_store:
            self.word_store[word] = Word(word)

        return self.word_store[word]

class Transition:
    def __init__(self):
        self.followers_count = {}
        self.probabilities = []
        self.probability_limit = 1
        self.has_changed = False
        self.previous_words = {}
        self.data_set_words = []
        self.data_set_probabilities = []

    def predict_next_word(self, probability_limit):
        if self.has_changed or probability_limit != self.probability_limit:
            self.update_probabilities(probability_limit)

        next_word = random.choice(self.data_set_words, size=1, p=self.data_set_probabilities)[0]
        return next_word

    def follows_word(self, previous_word):
        self.previous_words[previous_word] = previous_word

    def update_probabilities(self, limit):
        total_followers = self.number_of_followers()
        min_number_of_occurrences = math.ceil(total_followers * limit)
        data_set_words = []
        total_included_followers = 0

        for word in list(self.followers_count.keys()):
            if self.followers_count[word] < min_number_of_occurrences:
                continue
            data_set_words.append(word)
            total_included_followers = total_included_followers + self.followers_count[word]

        self.data_set_probabilities = list([self.followers_count[word] / total_included_followers for word in data_set_words])
        self.data_set_words = data_set_words
        self.has_changed = False
        self.probability_limit = limit

    def is_followed_by(self, next_word):
        self.has_changed = True

        if next_word in self.followers_count.keys():
            self.followers_count[next_word] = self.followers_count[next_word] + 1
        else:
            self.followers_count[next_word] = 1

        next_word.follows_word(self)

    def number_of_followers(self):
        return sum(self.followers_count.values())


class Word(Transition):
    def __init__(self, base_id):
        Transition.__init__(self)
        self.base = base_id


class MarkhovModel:
    def __init__(self):
        self.first_words = Transition()
        self.transitions = {}
        self.known_speeches = set()
        self.sentence_ends = ['.', '!', '?']
        self.word_count = {}
        self.total_words_learnt = 0
        self.word_store = WordStore()

    def learn_speeches(self, speeches):
        for speech in speeches:
            if speech['id'] not in self.known_speeches:
                self.known_speeches.add(speech['id'])
                self.learn_speech(speech['text'])

    def update_total_words_learnt(self, word_list):
        self.total_words_learnt = self.total_words_learnt + len(word_list)

    def learn_speech(self, speech):
        word_list = speech.split()
        if len(word_list) > 5:
            self.update_total_words_learnt(word_list)
            self.first_words.is_followed_by(self.word_store.get_word(word_list[0]))
            for i in range(0, len(word_list) - 1):
                current_word = word_list[i]
                next_word = word_list[i+1]

                if current_word[0] is '.' and len(current_word) is not 1:
                    current_word = current_word[1:]
                    self.link_words('.', current_word)

                self.link_words(current_word, next_word)

    def link_words(self, current_word, next_word):
        if current_word not in self.transitions:
            self.transitions[current_word] = self.word_store.get_word(current_word)
        if next_word not in self.transitions:
            self.transitions[next_word] = self.word_store.get_word(next_word)

        self.transitions[current_word].is_followed_by(self.transitions[next_word])

    def predict_speech(self, limit, probability=0.03):
        words = [self.first_words.predict_next_word(probability)]
        try:
            for i in range(0, limit):
                words.append(self.transitions[words[-1].base].predict_next_word(probability))

            while words[-1].base not in self.sentence_ends:
                words.append(self.transitions[words[-1].base].predict_next_word(probability))
        except:
            print(words)

        return ' '.join([w.base for w in words])


if __name__ == '__main__':
    debate_holder = Debates()

    debate_holder.between('2014-01-01', '2020-02-02')
    x = debate_holder.debates_from(25353)

    m = MarkhovModel()
    m.learn_speeches(x)
    for i in range(0, 30):
        print(m.predict_speech(20, 0.01))
        print('')

    print(m.predict_speech(20, 0.03))

