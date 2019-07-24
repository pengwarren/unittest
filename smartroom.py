import json as Json
import requests as Requests
import speech_recognition as Speech
import yaml as Yaml

from pixel_ring import pixel_ring as ReSpeaker
from textblob import TextBlob as Text
from textblob.classifiers import NaiveBayesClassifier as NaiveBayes

class Smartroom(object):
    def __init__(self):
        self._nouns = list()
        self._verbs = list()
        self._polarities = list()
        self._response = dict()

        self._text = None
        self._tags = list()
        self._words = list()
        self._ngrams = list()

        self._state = None
        self._classifier = None

        self._credentials = dict()
        self._telegram = dict()

        self.WAKE = ReSpeaker.listen
        self.ASRM = ReSpeaker.think
        self.BACK = ReSpeaker.trace
        self.IDLE = ReSpeaker.speak
        self.KNXM = ReSpeaker.think
        self.NLPM = ReSpeaker.spin
        self.POST = Requests.post
        self.PUTS = Requests.put

        self.state = self.NLPM
        self.configurations = self.get_configuration_file()
        self.microphone = Speech.Microphone(device_index=self.get_microphone_index())
        self.recognizer = Speech.Recognizer()
        self.classifier = NaiveBayes(
            train_set=self.build_training_data(),
            feature_extractor=self.extract_features
        )
        self.state = self.IDLE

    def __del__(self):
        print(f"{self.__class__.__name__} has been terminated")

    def __response__(self, verbs=None, polarities=None):
        verbs = verbs if verbs is not None else self.verbs
        polarities = polarities if polarities is not None else self.polarities
        response = dict()

        try:
            for i in range(len(self.nouns)):
                response[self.nouns[i]] = (verbs[i], polarities[i])
        except IndexError:
            verb, *_ = verbs
            polarity, *_ = polarities
            response = {
                self.nouns[i]: (verb, polarity)
                for i in range(len(self.nouns))
            }
        self._response = response
        return self._response

    def __str__(self):
        return str(self._response)

    @property
    def classifier(self):
        return self._classifier

    @classifier.setter
    def classifier(self, value):
        self._classifier = value

    @property
    def credentials(self):
        self._credentials = {
            "username": self.configurations["KNX_BAOS_SERVER"]["USERNAME"],
            "password": self.configurations["KNX_BAOS_SERVER"]["PASSWORD"]
        }
        return self._credentials

    @property
    def ngrams(self, n=2):
        self._ngrams = Text(self.text).ngrams(n=n)
        return self._ngrams

    @property
    def nouns(self):
        return self._nouns

    @nouns.setter
    def nouns(self, value):
        self._nouns = value

    @property
    def polarities(self):
        return self._polarities

    @polarities.setter
    def polarities(self, value):
        self._polarities = value

    @property
    def raw_response(self):
        return self.__response__()

    @property
    def response(self):
        return self._response

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value
        self._state()

    @property
    def telegram(self):
        return self._telegram

    @telegram.setter
    def telegram(self, value):
        self._telegram = {
            "command": self.configurations["COMMANDS"]["BAOS"],
            "value": value
        }

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

        universal_parameters, *_ = self.configurations["PARAMETERS"]
        self.nouns = [
            " ".join((first_word, second_word))
            for first_word, second_word in self.ngrams
            if first_word in universal_parameters
        ]

        self.nouns += [
            word
            for word, tag in self.tags
            if tag in self.configurations["NOUN_TAGS"]
            and not [word for noun in self.nouns if word in noun]
            and word in [
                parameter
                for parameters in self.configurations["PARAMETERS"]
                for parameter in parameters
                ]
        ]

        self.verbs.clear()
        self.polarities.clear()
        activation, deactivation = self.configurations["COMMANDS"]["WORDS"].values()

        for word, tag in self.tags:
            if tag in self.configurations["VERB_TAGS"]:
                for activation_verb, deactivation_verb in zip(activation, deactivation):
                    if word == activation_verb:
                        self.verbs += [word]
                        self.polarities += [1]
                    elif word == deactivation_verb:
                        if self.verbs and word in ["n't", "not"]:
                            self.verbs.pop()
                            self.polarities.pop()
                        self.verbs += [word]
                        self.polarities += [0]

        if (self.nouns and self.verbs and self.polarities) != True:
            self._response = "?"

    @property
    def tags(self):
        self._tags = Text(self.text).tags
        return self._tags

    @property
    def verbs(self):
        return self._verbs

    @verbs.setter
    def verbs(self, value):
        self._verbs = value

    @property
    def words(self):
        self._words = Text(self.text).words
        return self._words

    def build_training_data(self):
        return [
            (text, label)
            for label, texts in self.configurations["COMMANDS"]["PHRASES"].items()
            for text in texts
        ]

    def convert_speech_to_text(self):
        try:
            with self.microphone as input:
                self.recognizer.adjust_for_ambient_noise(input)
                speech = self.recognizer.listen(input)
            self.state = self.ASRM
            self.text = self.recognizer.recognize_google(speech)
            self.state = self.IDLE
            return self.text
        except Speech.RequestError:
            print(f"{self.__class__.__name__} failed to communicate with API")
            return None
        except Speech.UnknownValueError:
            print(f"{self.__class__.__name__} failed to understand the spoken words")
            return None
        except KeyboardInterrupt:
            print(f"{self.__class__.__name__} failed to complete on time")
            return None

    def extract_features(self, document, tokens):
        return {
            f"contains({word})": word in tokens
            for word, tag in Text(document).tags
            if tag in self.configurations["VERB_TAGS"]
        }

    def get_microphone_index(self):
        microphone_index, = [
            index
            for index, name in enumerate(Speech.Microphone.list_microphone_names())
            if self.configurations["MICROPHONE_MODEL_NAME"].lower() in name.lower()
        ]
        return microphone_index

    def perform_classification(self, is_naive=False):
        self.state = self.NLPM
        verbs = self.verbs
        polarities = self.polarities

        if set(["n't", "not"]).intersection(self.verbs):
            verbs = list(self.verbs)
            polarities = list(self.polarities)
            try:
                i = self.verbs.index("n't")
            except ValueError:
                i = self.verbs.index("not")
            verbs.pop(i)
            polarities.pop(i)
            verbs[i] = f"!{verbs[i]}"

            if is_naive:
                polarities[i] = 0 if self.classifier.classify(self.text) > 0 else 1
            else:
                polarities[i] = 0 if polarities[i] > 0 else 1
        elif is_naive:
            self.polarities = [self.classifier.classify(self.text)]

        self.state = self.IDLE
        return self.__response__(verbs=verbs, polarities=polarities)

    def perform_naive_bayes_classification(self):
        return self.perform_classification(is_naive=True)

    def perform_request(self, method, link, payload=None, key=None):
        try:
            response = method(
                f"{self.configurations['KNX_BAOS_SERVER']['URL']}{link}",
                cookies=key,
                data=Json.dumps(payload)
            )
        except Exception:
            raise NotImplementedError

        self.verify_status_code(response)
        return response

    def wait_for_wake_word(self, wake_word):
        while self.convert_speech_to_text() != wake_word.lower():
            self.state = self.IDLE

        self.state = self.WAKE
        return self.convert_speech_to_text()

    @classmethod
    def get_configuration_file(cls):
        try:
            with open(f"{str(cls.__name__).lower()}.yaml") as file:
                return Yaml.full_load(file)
        except OSError:
            print(f"{cls.__name__} failed to locate its configuration file")
            return None

    @classmethod
    def verify_status_code(cls, response):
        if response.status_code not in (Requests.codes.ok, Requests.codes.no_content):
            response.raise_for_status()

    @classmethod
    def throw_parameter_exception(cls):
        raise Smartroom.ParameterError(f"{cls.__name__} received bad input")

    class ParameterError(Exception):
        pass
