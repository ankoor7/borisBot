import xml.dom.minidom as minidom
import glob

from transformer.LoadDebates import preprocess_sentence


class Debate:

    def __init__(self, fileObj):
        self.fileObj = fileObj
        self.day = fileObj.name
        self._tree = minidom.parse(str(self.fileObj))
        self.speeches = []
        self.parsed_speech_ids = {}
        return

    @staticmethod
    def person_id(numeric_id):
        return f'uk.org.publicwhip/person/{numeric_id}'

    def getText(self, nodelist):
        # Iterate all Nodes aggregate TEXT_NODE
        rc = []
        for node in nodelist:
            if node.nodeType == node.TEXT_NODE:
                rc.append(preprocess_sentence(node.data))
            if node.nodeName == 'phrase':
                rc.append(f' ++phrase_{node._attrs["class"].value}++ ')
            else:
                # Recursive
                rc.append(self.getText(node.childNodes))
        return ''.join(rc)

    def parse_speeches(self, numeric_id):
        previous_speaker = False
        previous_text = False

        for persons_speech in self._tree.getElementsByTagName('speech'):
            if 'person_id' not in persons_speech.attributes._attrs:
                continue

            if persons_speech.attributes._attrs['person_id'].value != Debate.person_id(numeric_id):
                continue

            if persons_speech.attributes._attrs['id'].value in self.parsed_speech_ids:
                continue

            current_speaker = persons_speech._attrs['person_id'].value

            data = {
                'id': persons_speech.attributes._attrs['id'].value,
                'person_id': current_speaker,
                'text': [],
            }

            if previous_speaker and current_speaker is not previous_speaker:
                data['responding_to_person_id'] = previous_speaker
                data['responding_to_text'] = previous_text

            data['text'] = self.getText(persons_speech.getElementsByTagName('p'))

            self.speeches.append(data)
            self.parsed_speech_ids[persons_speech.attributes._attrs['id'].value] = True

            previous_speaker = current_speaker
            previous_text = data['text']

    def speeches_by(self, numeric_id):
        id_match = Debate.person_id(numeric_id)
        self.parse_speeches(numeric_id)
        return list(
            filter(
                lambda speech:
                speech['person_id'] == id_match,
                self.speeches
            )
        )

    def responses_by(self, numeric_id):
        id_match = Debate.person_id(numeric_id)
        self.parse_speeches(numeric_id)
        return list(
            filter(
                lambda speech:
                'responding_to_person_id' in speech and speech['person_id'] == id_match,
                self.speeches
            )
        )

    def speeches_including(self, numeric_id):
        id_match = Debate.person_id(numeric_id)
        self.parse_speeches(numeric_id)
        return list(
            filter(
                lambda speech:
                (speech['person_id'] == id_match) or (speech['responding_to_person_id'] == id_match),
                self.speeches
            )
        )
