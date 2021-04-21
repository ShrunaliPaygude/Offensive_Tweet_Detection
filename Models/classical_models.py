import pandas as pd
import warnings

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import LatentDirichletAllocation as LDA

warnings.filterwarnings("ignore")
from sklearn.svm import SVC

def read_data(filename):
    file_content = pd.read_excel(filename)
    return file_content

def word_extraction(sentence):
    words = sentence.split()
    cleaned_text = [w for w in words]
    return cleaned_text


def tokenize(sentences):
    words = []
    for sentence in sentences:
        w = word_extraction(sentence)
        words.extend(w)
        words = sorted(list(set(words)))
    return words


def TFIDF_SVM(train_data, test_data):
    print("TFIDF + SVM")
    model = make_pipeline(TfidfVectorizer(ngram_range=(1,1)), SVC())
    X_train = train_data['Tweet']
    y_train = train_data['Class']

    X_test = test_data['Tweet']
    y_test = test_data['Class']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels, train_data['Class'].unique())
    print("Confusion matrix", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")

def TFIDF_Multi_Naive_Bayes(train_data, test_data):
    print("TFIDF + MultiNomial Naive Bayes")
    model = make_pipeline(TfidfVectorizer(ngram_range=(1,1)), MultinomialNB())
    X_train = train_data['Tweet']
    y_train = train_data['Class']

    X_test = test_data['Tweet']
    y_test = test_data['Class']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels, train_data['Class'].unique())
    print("Confusion matrix", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")


def TFIDF_Decision(train_data, test_data):
    print("TFIDF + Decision tree")
    model = make_pipeline(TfidfVectorizer(ngram_range=(1,1)), DecisionTreeClassifier())
    X_train = train_data['Tweet']
    y_train = train_data['Class']

    X_test = test_data['Tweet']
    y_test = test_data['Class']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels, train_data['Class'].unique())
    print("Confusion matrix", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")


def TFIDF_Random_forest(train_data, test_data):
    print("TFIDF + Random forest")
    model = make_pipeline(TfidfVectorizer(ngram_range=(1,1)), RandomForestClassifier())
    X_train = train_data['Tweet']
    y_train = train_data['Class']

    X_test = test_data['Tweet']
    y_test = test_data['Class']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels, train_data['Class'].unique())
    print("Confusion matrix", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")


def BOW_Random_forest(train_data, test_data):
    print("BOW + Random forest")
    model = make_pipeline(CountVectorizer(ngram_range=(1, 1)), RandomForestClassifier())
    X_train = train_data['Tweet']
    y_train = train_data['Class']

    X_test = test_data['Tweet']
    y_test = test_data['Class']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels, train_data['Class'].unique())
    print("Confusion matrix", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")

def BOW_Decision_Tree(train_data, test_data):
    print("BOW + Decsion Tree")
    model = make_pipeline(CountVectorizer(ngram_range=(1, 1)), DecisionTreeClassifier())
    X_train = train_data['Tweet']
    y_train = train_data['Class']

    X_test = test_data['Tweet']
    y_test = test_data['Class']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels, train_data['Class'].unique())
    print("Confusion matrix", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")
def BOW_Multi_Naive_Bayes(train_data, test_data):
    print("BOW + Naive Bayes")
    model = make_pipeline(CountVectorizer(ngram_range=(1, 1)), MultinomialNB())
    X_train = train_data['Tweet']
    y_train = train_data['Class']

    X_test = test_data['Tweet']
    y_test = test_data['Class']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels, train_data['Class'].unique())
    print("Confusion matrix", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")

def BOW_SVM(train_data, test_data):
    print("BOW + SVM")
    model = make_pipeline(CountVectorizer(ngram_range=(1,1)), SVC())
    # model = make_pipeline(TfidfVectorizer(), SVC())
    X_train = train_data['Tweet']
    y_train = train_data['Class']

    X_test = test_data['Tweet']
    y_test = test_data['Class']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels, train_data['Class'].unique())
    print("Confusion matrix", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")


def LDA_Multi_Naive_Bayes(train_data, test_data):
    print("LDA + MultiNomial Naive Bayes")
    model = make_pipeline(CountVectorizer(), LDA(), MultinomialNB())
    X_train = train_data['Tweet']
    y_train = train_data['Class']

    X_test = test_data['Tweet']
    y_test = test_data['Class']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels, train_data['Class'].unique())
    print("Confusion matrix", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")


def LDA_Decision(train_data, test_data):
    print("LDA + Decision tree")
    model = make_pipeline(CountVectorizer(), LDA(), DecisionTreeClassifier())
    X_train = train_data['Tweet']
    y_train = train_data['Class']

    X_test = test_data['Tweet']
    y_test = test_data['Class']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels, train_data['Class'].unique())
    print("Confusion matrix", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")


def LDA_SVM(train_data, test_data):
    print("LDA + SVM")
    model = make_pipeline(CountVectorizer(), LDA(), SVC())
    X_train = train_data['Tweet']
    y_train = train_data['Class']

    X_test = test_data['Tweet']
    y_test = test_data['Class']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels, train_data['Class'].unique())
    print("Confusion matrix", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")

def LDA_Random_forest(train_data, test_data):
    print("LDA + Random forest")
    model = make_pipeline(CountVectorizer(), LDA(), RandomForestClassifier())
    X_train = train_data['Tweet']
    y_train = train_data['Class']

    X_test = test_data['Tweet']
    y_test = test_data['Class']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels, train_data['Class'].unique())
    print("Confusion matrix", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")


def word_clouds(tweets):
    comment_words = ""
    map_of_words = {}
    for tweet in tweets['Tweet']:
        # comment_words += tweet + " "
        for word in tweet.split():
            if word in map_of_words:
                map_of_words[word] += 1
            else:
                map_of_words[word] = 1
    map_of_words = [(v, k) for k, v in map_of_words.items()]
    map_of_words.sort(reverse=True)  # natively sort tuples by first element
    i = 0
    for v, k in map_of_words:
        if i == 5:
            break
        i += 1
        print ("%s: %d" % (k, v))

    print()


def main():
    train_filename = 'train_data_offensive_language_marathi_updated.xlsx'
    train_data = read_data(train_filename)
    train_data = train_data[['Tweet', 'Class']]

    test_filename = 'test_data_offensive_language_marathi_updated.xlsx'
    test_data = read_data(test_filename)
    test_data = test_data[['Tweet', 'Class']]

    # print(len(train_data[train_data['Class'] == 'offensive']))
    # print(len(train_data[train_data['Class'] == 'not offensive']))
    #
    # print(len(test_data[test_data['Class'] == 'offensive']))
    # print(len(test_data[test_data['Class'] == 'not offensive']))
    #
    # print(len(train_data))
    # print(len(test_data))

    # TFIDF_Decision(train_data, test_data)
    TFIDF_Multi_Naive_Bayes(train_data, test_data)
    # TFIDF_Random_forest(train_data, test_data)
    TFIDF_SVM(train_data, test_data)

    BOW_Decision_Tree(train_data, test_data)
    # BOW_Multi_Naive_Bayes(train_data, test_data)
    BOW_Random_forest(train_data, test_data)
    # BOW_SVM(train_data, test_data)
    #
    # LDA_Decision(train_data, test_data)
    # LDA_Multi_Naive_Bayes(train_data, test_data)  # 66.88
    # LDA_Random_forest(train_data, test_data)
    # LDA_SVM(train_data, test_data)  # 65.60


if __name__ == '__main__':
    main()
