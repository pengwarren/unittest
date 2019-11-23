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
    def polarities(self):
        return self._polarities

    @polarities.setter
    def polarities(self, value):
        self._polarities = value

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
