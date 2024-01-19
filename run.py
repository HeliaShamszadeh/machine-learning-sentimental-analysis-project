from template import NaiveBayesClassifier
import time
import nltk
import string
import csv

# config for tokenizer
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words("english")).union(set(string.punctuation))
tokenizer = nltk.TweetTokenizer()

def preprocess(tweet_string : str):
    # clean the data and tokenize it

    words = tokenizer.tokenize(tweet_string)
    features = [word for word in words if word.lower() not in stop_words]

    return features

def load_data(data_path : str):
    # load the csv file and return the data
    data = []
    with open(data_path, "r", newline="") as csv_file:
        data = map(lambda _line: _line[2:], list(csv.reader(csv_file))[1:])
    return list(map(lambda _tuple : (preprocess(_tuple[0]), _tuple[2]), data))

def test_model(eval_data_path : str):
    data = []
    with open(eval_data_path, "r", newline="") as csv_file:
        data = map(lambda _line: _line[2:], list(csv.reader(csv_file))[1:])
    data = map(lambda _tuple : (preprocess(_tuple[0]), _tuple[2]), data)
    currect = 0
    incurrect = 0
    for features, label in data:
        if nb_classifier.classify(features) == label:
            currect += 1
        else:
            incurrect += 1
    return currect / (currect + incurrect)


# train your model and report the duration time
start_time = time.time()

# train_data_path = input("address for train data:\n") 
train_data_path = "train_data.csv"
classes = ['positive', 'negative', 'neutral']
nb_classifier = NaiveBayesClassifier(classes)
nb_classifier.train(load_data(train_data_path))

end_time = time.time()

test_string = "I love playing football"

print(f"total training time {end_time - start_time}")

# print(nb_classifier.classify(preprocess(test_string)))

print(f"accuracy is {test_model('train_data.csv')*100}%")
print(f"accuracy is {test_model('eval_data.csv')*100}%")