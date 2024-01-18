# Naive Bayes 3-class Classifier 
# Authors: Baktash Ansari - Sina Zamani 

# complete each of the class methods  

class NaiveBayesClassifier:

    def __init__(self, classes : list[str]) -> None:
        # initialization: 
        # inputs: classes(list) --> list of label names
        # class_word_counts --> frequency dictionary for each class
        # class_counts --> number of instances of each class
        # vocab --> all unique words  
        self.classes = classes
        self.class_word_counts = None
        self.class_counts = len(classes)
        self.vocab = None

    def train(self, data : list[tuple[list, str]]) -> None:
        # training process:
        # inputs: data(list) --> each item of list is a tuple 
        # the first index of the tuple is a list of words and the second index is the label(positive, negative, or neutral)

        for features, label in data:
            pass
            # Your Code

    def calculate_prior(self) -> float:
        # calculate log prior
        # you can add some attributes to this method
  
        # Your Code
        return None 

    def calculate_likelihood(self, word : str, label : str) -> float:
        # calculate likelihhood: P(word | label)
        # return the corresponding value

        # Your Code
        return None

    def classify(self, features : list[str]) -> str:
        # predict the class
        # inputs: features(list) --> words of a tweet 
        best_class = None 

        # Your Code
        return best_class
    

# Good luck :)
