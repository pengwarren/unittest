import yaml as Yaml
from textblob import TextBlob as Text
from textblob.classifiers import NaiveBayesClassifier as NaiveBayes

class Smartroom(object):
    def __init__(self):
        self.configurations = self.get_configuration_file()
        self._classifier = NaiveBayes(train_set=self.build_training_data(), feature_extractor=self.extract_features)
        self._ngrams = list()
        self._nouns = list()
        self._verbs = list()
        self._polarities = list()
        self._response = dict()
        self._tags = list()
        self._text = None

    @classmethod
    def get_configuration_file(cls):
        try:
            with open(f"{str(cls.__name__).lower()}.yaml") as file:
                return Yaml.full_load(file)
        except OSError:
            print(f"{cls.__name__} failed to locate its configuration file")
            return None

    def build_training_data(self):
        return [(text, label) for label, texts in self.configurations["COMMANDS"]["PHRASES"].items() for text in texts]

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
    def verify_status_code(cls, response):
        if response.status_code not in (Requests.codes.ok, Requests.codes.no_content):
            response.raise_for_status()

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
            print(f"{self.__class__.__name__} failed to hear the spoken words")
            return None
        except KeyboardInterrupt:
            print(f"{self.__class__.__name__} failed to complete on time")
            return None

    def extract_features(self, document, tokens):
        return {f"contains({word})": word in tokens for word, tag in Text(document).tags if tag in self.configurations["VERB_TAGS"]}

    def __response__(self, verbs=None, polarities=None):
        verbs = verbs if verbs is not None and verbs else self.verbs
        polarities = polarities if polarities is not None and polarities else self.polarities
        response = dict()

        try:
            for i in range(len(self.nouns)):
                response[self.nouns[i]] = (verbs[i], polarities[i])
        except IndexError:
            verb, *_ = verbs if verbs else "?"
            polarity, *_ = polarities
            response = {self.nouns[i]: (verb, polarity) for i in range(len(self.nouns))}
            self._response = response
            return self._response

    def __str__(self):
        return str(self._response)

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
    def verbs(self):
        return self._verbs

    @verbs.setter
    def verbs(self, value):
        self._verbs = value

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
    def polarities(self):
        return self._polarities

    @polarities.setter
    def polarities(self, value):
        self._polarities = value

    @property
    def response(self):
        return self._response

    @property
    def classifier(self):
        return self._classifier

    @classifier.setter
    def classifier(self, value):
        self._classifier = value

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

        universal_parameters, *_ = self.configurations["PARAMETERS"]
        self.nouns = [" ".join((first_word, second_word)) for first_word, second_word in self.ngrams if first_word in universal_parameters]
        self.nouns += [word for word, tag in self.tags if tag in self.configurations["NOUN_TAGS"] and not [word for noun in self.nouns if word in noun] and word in [parameter for parameters in self.configurations["PARAMETERS"] for parameter in parameters]]

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

        if not (self.nouns and self.verbs and self.polarities):
            self._response = "?"
        else:
            self._response = dict()

    @property
    def tags(self):
        self._tags = Text(self.text).tags
        return self._tags

luna = Smartroom()
