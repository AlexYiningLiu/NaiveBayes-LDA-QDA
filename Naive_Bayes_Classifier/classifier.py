import os.path
import numpy as np
import matplotlib.pyplot as plt
import util

def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """
    spam_files = file_lists_by_category[0]
    ham_files = file_lists_by_category[1]

    ham_word_dict = util.get_word_freq(ham_files)
    spam_word_dict = util.get_word_freq(spam_files)

    spam_words_total = sum(spam_word_dict.values())
    ham_words_total = sum(ham_word_dict.values())
    spam_word_keys = list(spam_word_dict.keys())
    ham_word_keys = list(ham_word_dict.keys())
    # use set() to make a set of distinct words between both spam and ham 
    D = len(set(spam_word_keys + ham_word_keys))
    print(D)
    p_dict = spam_word_dict.copy()
    q_dict = ham_word_dict.copy()
    
    for word in spam_word_dict:
        p_dict[word] = (spam_word_dict[word] + 1) / (spam_words_total + D)

    for word in ham_word_dict:
        q_dict[word] = (ham_word_dict[word] + 1) / (ham_words_total + D)

    probabilities_by_category = (p_dict, q_dict)
    print(len(p_dict), len(q_dict))
    print(spam_words_total, ham_words_total)
    return probabilities_by_category, spam_words_total, ham_words_total, D

def classify_new_email(filename,probabilities_by_category,prior_by_category, spam_words_total, ham_words_total, D, r):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """

    p_dict = probabilities_by_category[0]
    q_dict = probabilities_by_category[1]
    spam_prior = prior_by_category[0]
    ham_prior = prior_by_category[1]
    log_spam_conditional = 0
    log_ham_conditional = 0
    file_word_frequencies = util.get_word_freq([filename])

    for word in file_word_frequencies:
        if word in p_dict:
            p_d = p_dict[word]
        elif word in q_dict:
            p_d = 1 / (spam_words_total + D)

        if word in q_dict:
            q_d = q_dict[word]
        elif word in p_dict:
            q_d = 1 / (ham_words_total + D)
        
        if (word not in q_dict) and (word not in p_dict):
            continue
        
        log_spam_conditional += (file_word_frequencies[word] * np.log(p_d))
        log_ham_conditional += (file_word_frequencies[word] * np.log(q_d))
    
    log_spam_posterior = log_spam_conditional + np.log(spam_prior) 
    log_ham_posterior = log_ham_conditional + np.log(ham_prior) 
    
    if log_spam_posterior > np.log(r) + log_ham_posterior:
        result = 'spam'
    else:
        result = 'ham'
    classify_result = (result, [log_spam_posterior, log_ham_posterior])
    return classify_result

if __name__ == '__main__':
    
    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
        
    # Learn the distributions    
    probabilities_by_category, spam_words_total, ham_words_total, D = learn_distributions(file_lists)
    
    # prior class distribution
    priors_by_category = [0.5, 0.5]
    
    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # Classify emails from testing set and measure the performance
    # for filename in (util.get_files_in_folder(test_folder)):
    #     # Classify
    #     label,log_posterior = classify_new_email(filename,
    #                                              probabilities_by_category,
    #                                              priors_by_category, spam_words_total, ham_words_total, D)
        
    #     # Measure performance (the filename indicates the true label)
    #     base = os.path.basename(filename)
    #     true_index = ('ham' in base) 
    #     guessed_index = (label == 'ham')
    #     performance_measures[int(true_index), int(guessed_index)] += 1

    # template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # # Correct counts are on the diagonal
    # correct = np.diag(performance_measures)
    # # totals are obtained by summing across guessed labels
    # totals = np.sum(performance_measures, 1)
    # print(template % (correct[0],totals[0],correct[1],totals[1]))
    
    type1_error = []
    type2_error = []

    for r in [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e5, 1e7, 1e9, 1e11, 1e15, 1e22]:
        performance_measures = np.zeros([2,2])
        for filename in (util.get_files_in_folder(test_folder)):
            # Classify
            label,log_posterior = classify_new_email(filename,
                                                    probabilities_by_category,
                                                    priors_by_category, spam_words_total, ham_words_total, D, r)
            
            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)
            true_index = ('ham' in base) 
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1

        template="For r = %.0E You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
        # Correct counts are on the diagonal
        correct = np.diag(performance_measures)
        # totals are obtained by summing across guessed labels
        totals = np.sum(performance_measures, 1)
        print(template % (r,correct[0],totals[0],correct[1],totals[1]))
        type1_error.append(totals[0] - correct[0])
        type2_error.append(totals[1] - correct[1])
    
    plt.plot(type1_error, type2_error, '--b*')
    plt.xlabel('Type 1 Errors')
    plt.ylabel('Type 2 Errors')
    plt.title('Trade-off Between Type 1 and Type 2 Error')
    plt.savefig("nbc.pdf")
    plt.show()

 