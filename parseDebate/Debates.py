import glob
import re
from datetime import timedelta, date
from pathlib import Path

from transformer.LoadDebates import load_conversations
from parseDebate.Debate import Debate


class Debates:
    def __init__(self):
        p = Path('../scrapedxml/debates')
        debate_files = sorted(list(p.glob('*')))
        self.debate_file_names = [file.name for file in debate_files]

        self.debates_files = {file.name: file for file in debate_files}
        self.debates = {}
        self.phrases = set()
        return

    def latest_file_name(self):
        debate_files = sorted(glob.glob(f'scrapedxml/debates/{self.day}'))
        if debate_files.__len__() < 1:
            raise Exception(f'no debate file {self.day}')

        return debate_files[-1]

    def debate_files_exist(self, day):
        matcher = re.compile(f'debates{day}')
        return any(matcher.match(file) for file in self.debate_file_names)

    def debate_file_for_day(self, day):
        matcher = re.compile(f'debates{day}')
        file_names_for_day = list(
            filter(
                lambda file:
                matcher.match(file),
                self.debate_file_names
            )
        )
        return self.debates_files[file_names_for_day[-1]]

    def between(self, day_from, day_to):
        sdate = date.fromisoformat(day_from)  # start date
        edate = date.fromisoformat(day_to)  # end date

        delta = edate - sdate       # as timedelta

        for i in range(delta.days + 1):
            day = str(sdate + timedelta(days=i))
            if self.debate_files_exist(day) and day not in self.debates:
                self.debates[day] = Debate(self.debate_file_for_day(day))

    def debates_from(self, numeric_id):
        data = [debate.responses_by(numeric_id) for debate in self.debates.values()]
        return [val for sublist in data for val in sublist]

if __name__ == '__main__':
    debate_holder = Debates()

    debate_holder.between('2020-06-01', '2020-06-30')
    x = debate_holder.debates_from(25353)

    y = 1
