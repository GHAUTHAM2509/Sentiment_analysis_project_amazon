from string import punctuation, digits
import numpy as np
import random

def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses given parameters to classify a set of
    data points.

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.

    Returns:
        a numpy array of 1s and -1s where the kth element of the array is the
        predicted classification of the kth row of the feature matrix using the
        given theta and theta_0. If a prediction is GREATER THAN zero, it
        should be considered a positive classification.
    """
    # Your code here
    return (feature_matrix @ theta + theta_0 > 1e-7) * 2.0 - 1 
    return (feature_matrix @ theta + theta_0 )
    raise NotImplementedError


def extract_words(text):
    """
    Helper function for `bag_of_words(...)`.
    Args:
        a string `text`.
    Returns:
        a list of lowercased words in the string, where punctuation and digits
        count as their own words.
    """
    # Your code here
    for c in punctuation + digits:
        text = text.replace(c, ' ' + c + ' ')
    return text.lower().split()

def read_words_from_file(filename):
    """
    Reads words from a file where each line contains a single word
    and stores them in a list.

    Args:
        filename (str): The path to the text file.
    
    Returns:
        list: A list containing all the words from the file.
    """
    word_list = []
    with open(filename, 'r') as file:
        for line in file:
            # Strip any leading/trailing whitespace and append to list
            word_list.append(line.strip())
    return word_list

stopword = read_words_from_file("stopwords.txt")

def bag_of_words(texts, remove_stopword=False):
    """
    NOTE: feel free to change this code as guided by Section 3 (e.g. remove
    stopwords, add bigrams etc.)

    Args:
        `texts` - a list of natural language strings.
    Returns:
        a dictionary that maps each word appearing in `texts` to a unique
        integer `index`.
    """
    
    dictionary = {} # maps word to unique index
    
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary and word not in stopword:
                dictionary[word] = len(dictionary)
    return dictionary



def extract_bow_feature_vectors(reviews, indices_by_word, binarize=True):
    """
    Args:
        `reviews` - a list of natural language strings
        `indices_by_word` - a dictionary of uniquely-indexed words.
    Returns:
        a matrix representing each review via bag-of-words features.  This
        matrix thus has shape (n, m), where n counts reviews and m counts words
        in the dictionary.
    """
    # Your code here
    feature_matrix = np.zeros([len(reviews), len(indices_by_word)], dtype=np.float64)
    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word not in indices_by_word: continue
            feature_matrix[i, indices_by_word[word]] += 1
    if binarize:
        # Your code here
        feature_matrix = (feature_matrix > 0).astype(np.float64)
        
    return feature_matrix

def classify_star_rating(Feature_matrix, theta, theta_0):
    prediction = classify(Feature_matrix, theta, theta_0)
    return (prediction)
    