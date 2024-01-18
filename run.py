from template import NaiveBayesClassifier
import time

def preprocess(tweet_string : str):
    # clean the data and tokenize it

    features = []
    # Your Code
    return features

def load_data(data_path : str):
    # load the csv file and return the data
    data = []
    with open(data_path, "r") as file:
        data = map(lambda string: string.split()[2:], file.readlines()[1:])
    return list(map(lambda _tuple : (preprocess(_tuple[0]), _tuple[1]), data))


# train your model and report the duration time
start_time = time.time()

train_data_path = input("address for train data:\n") 
classes = ['positive', 'negative', 'neutral']
nb_classifier = NaiveBayesClassifier(classes)
nb_classifier.train(load_data(train_data_path))

end_time = time.time()

test_string = "I love playing football"

print(f"total training time {end_time - start_time}")

print(nb_classifier.classify(preprocess(test_string)))
