import math
# Naive Bayes 3-class Classifier 


class NaiveBayesClassifier:

    def __init__(self, classes : list[str]) -> None:
        # initialization: 
        # inputs: classes(list) --> list of label names
        # class_word_counts --> frequency dictionary for each class
        # class_counts --> number of instances of each class
        # vocab --> all unique words  
        self.classes = classes
        self.class_word_counts = dict(map(lambda key: (key, dict()), self.classes))
        self.class_counts = dict(map(lambda key: (key, 0), self.classes))
        self.vocab = set[str]()
        self.feature_count = dict(map(lambda key: (key, 0), self.classes)) # number of features for each class
        self.prior_p_to_n = 0.0; self.prior_p_to_u = 0.0; self.prior_n_to_u = 0.0
        self.alpha = 1


    def train(self, data : list[tuple[list, str]]) -> None:
        # training process:
        # inputs: data(list) --> each item of list is a tuple 
        # the first index of the tuple is a list of words and the second index is the label(positive, negative, or neutral)

        for features, label in data:
            self.feature_count[label] += 1
            self.class_counts[label] += len(features)
            self.vocab.union(set(features))
            for token in features:
                self.class_word_counts[label].setdefault(token, 0)
                self.class_word_counts[label][token] += 1


    def calculate_prior(self) -> None:
        # calculate log prior
        # you can add some attributes to this method
  
        self.prior_p_to_n = math.log(self.feature_count[self.classes[0]] / self.feature_count[self.classes[1]])
        self.prior_p_to_u = math.log(self.feature_count[self.classes[0]] / self.feature_count[self.classes[2]])
        self.prior_n_to_u = math.log(self.feature_count[self.classes[1]] / self.feature_count[self.classes[2]])


    def calculate_likelihood(self, word : str, label : str) -> float:
        # calculate likelihhood: P(word | label)
        # return the corresponding value

        self.class_word_counts[label].setdefault(word, 0)
        return (self.alpha + self.class_word_counts[label][word]) / (self.class_counts[label] + sum(self.feature_count.values()))

    def classify(self, features : list[str]) -> str:
        # predict the class
        # inputs: features(list) --> words of a tweet 
        self.calculate_prior()
        best_class = None 
        score = self.prior_p_to_n + sum(map(lambda token : math.log(self.calculate_likelihood(token, self.classes[0]) / self.calculate_likelihood(token, self.classes[1])), features))
        if score >= 0:
            score = self.prior_p_to_u + sum(map(lambda token : math.log(self.calculate_likelihood(token, self.classes[0]) / self.calculate_likelihood(token, self.classes[2])), features))
            best_class = self.classes[0] if score > 0 else self.classes[2]
        else:
            score = self.prior_n_to_u + sum(map(lambda token : math.log(self.calculate_likelihood(token, self.classes[1]) / self.calculate_likelihood(token, self.classes[2])), features))
            best_class = self.classes[1] if score > 0 else self.classes[2]

        # when should learn new things?
        self.feature_count[best_class] += 1  
        self.class_counts[best_class] += len(features)
        for token in features:
            self.class_word_counts[best_class][token] += 1
            self.vocab.add(token)

        return best_class
    

# Good luck :)
