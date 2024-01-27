from template import NaiveBayesClassifier
import time
import nltk
import string
import csv

# Config for tokenizer
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words("english")).union(set(string.punctuation))
tokenizer = nltk.TweetTokenizer()

# Tokenize each tweet and extracts useful words
def preprocess(tweet_string : str):
    # clean the data and tokenize it
    words = tokenizer.tokenize(tweet_string)
    features = [word for word in words if word.lower() not in stop_words]
    return features

# Reads and tokenizes extracted tweet of each line of the csv file
def load_data(data_path : str):
    # load the csv file and return the data
    data = []
    with open(data_path, "r", newline="") as csv_file:
        data = map(lambda _line: _line[2:], list(csv.reader(csv_file))[1:])
    return list(map(lambda _tuple : (preprocess(_tuple[0]), _tuple[2]), data))


# Extracts the features of each tweet, estimates a label for that and compares it with the actual label
# returns the correct matches / all compares
def test_model(eval_data_path : str):
    data = []
    with open(eval_data_path, "r", newline="") as csv_file:
        data = map(lambda _line: _line[2:], list(csv.reader(csv_file))[1:])
    data = map(lambda _tuple : (preprocess(_tuple[0]), _tuple[2]), data)
    correct = 0
    incorrect = 0
    for features, label in data:
        if nb_classifier.classify(features) == label:
            correct += 1
        else:
            incorrect += 1
    return correct / (correct + incorrect)


# Classifies the test data and writes the expected labels in a txt file.
def classify_test(test_data_path):
    with open(test_data_path, "r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # skip header
        with open("result.txt", "w") as result_file:
            for index, row in enumerate(reader):
                text_id, text = row[0], row[2]  # extract text id and text
                preprocessed_text = preprocess(text)
                expected_label = nb_classifier.classify(preprocessed_text)
                result_file.write(f"{text_id}: {expected_label}\n")


# train your model and report the duration time
start_time = time.time()

# train_data_path = input("address for train data:\n") 
train_data_path = "train_data.csv"
test_data_path = "test_data_nolabel.csv"
classes = ['positive', 'negative', 'neutral']
nb_classifier = NaiveBayesClassifier(classes)
nb_classifier.train(load_data(train_data_path))

end_time = time.time()

print(f"total training time : {end_time - start_time}")

print(f"accuracy of training data is {test_model('train_data.csv')*100}%")
print(f"accuracy of eval data is {test_model('eval_data.csv')*100}%")

classify_test(test_data_path)

