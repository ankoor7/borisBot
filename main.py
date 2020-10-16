from parseDebate import Debate


def parse_debate():
    debate = Debate.Debate('2020-06-24')
    debate_old = Debate.Debate('1919-02-20')
    s = debate.responses_by(10999)
    r = 4

if __name__ == '__main__':
    parse_debate()
