import numpy as np
import matplotlib.pyplot as plt
import util

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    num_males = np.count_nonzero(y == 1)
    num_females = np.count_nonzero(y == 2)
    N = x.shape[0]

    indicator_male = y.copy()
    indicator_male[indicator_male == 2] = 0
    indicator_male = np.expand_dims(indicator_male, axis=1)
    indicator_male_double = np.concatenate([indicator_male, indicator_male], axis=1)

    indicator_female = y.copy()
    indicator_female[indicator_female == 1] = 0
    indicator_female[indicator_female == 2] = 1 
    indicator_female = np.expand_dims(indicator_female, axis=1)
    indicator_female_double = np.concatenate([indicator_female, indicator_female], axis=1)

    mu_male = np.sum(np.multiply(indicator_male_double, x), axis=0) / num_males
    mu_female = np.sum(np.multiply(indicator_female_double, x), axis=0) / num_females
    print("Male means")
    print(mu_male)
    print('Female means')
    print(mu_female)

    cov = np.zeros((x.shape[1], x.shape[1]))
    for n in range(0, N):
        if y[n] == 1:
            diff = np.expand_dims(x[n], axis=1)-np.expand_dims(mu_male, axis=1)
            cov = cov + np.dot(diff, diff.T) 
        else:
            diff = np.expand_dims(x[n], axis=1)-np.expand_dims(mu_female, axis=1)
            cov = cov + np.dot(diff, diff.T)
    cov = cov / (N)
    print('cov')
    print(cov)

    cov_male = np.zeros(cov.shape)
    cov_female = np.zeros(cov.shape)
    for n in range(0, N):
        if y[n] == 1:
            diff = np.expand_dims(x[n], axis=1)-np.expand_dims(mu_male, axis=1)
            cov_male = cov_male + np.dot(diff, diff.T)
        else:
            diff = np.expand_dims(x[n], axis=1)-np.expand_dims(mu_female, axis=1)
            cov_female = cov_female + np.dot(diff, diff.T)
    cov_male = cov_male / (num_males)
    cov_female = cov_female / (num_females)
    print('cov_male')
    print(cov_male)
    print('cov_female')
    print(cov_female)

    # plot N data points
    male_height = []
    male_weight = []
    female_height = []
    female_weight = []
    for n in range(0, N):
        if y[n] == 1:
            male_height.append(x[n][0])
            male_weight.append(x[n][1])
        else:
            female_height.append(x[n][0])
            female_weight.append(x[n][1])

    # there are N height and N weight values 
    x_limits = np.linspace(50, 80, N)   
    y_limits = np.linspace(80, 280, N)   
    x_mesh, y_mesh = np.meshgrid(x_limits, y_limits)

    male_lda_criteria = []
    male_qda_criteria = []
    female_lda_criteria = []
    female_qda_criteria = []
    x_coordinates = x_mesh[0].reshape(100, 1)
    for n in range(0, N):
        y_coordinates = y_mesh[n].reshape(100, 1)
        x_set = np.concatenate((x_coordinates, y_coordinates), axis=1)
        male_lda_criteria.append(util.density_Gaussian(mu_male,cov,x_set))
        female_lda_criteria.append(util.density_Gaussian(mu_female,cov,x_set))
        male_qda_criteria.append(util.density_Gaussian(mu_male,cov_male,x_set))
        female_qda_criteria.append(util.density_Gaussian(mu_female,cov_female,x_set))

    plt.scatter(male_height, male_weight, color = 'b', label='Male')
    plt.scatter(female_height, female_weight, color = 'r', label='Female')
    plt.legend(loc=2)
    male_CS = plt.contour(x_mesh, y_mesh, male_lda_criteria, colors='b')
    female_CS = plt.contour(x_mesh, y_mesh, female_lda_criteria, colors='r')
    lda_decision_boundary = np.asarray(male_lda_criteria) - np.asarray(female_lda_criteria)
    plt.contour(x_mesh, y_mesh, lda_decision_boundary, colors='k', levels=[0])
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('LDA Contours and Decision Boundary')
    plt.savefig('lda.pdf')
    plt.show()

    plt.scatter(male_height, male_weight, color = 'b', label='Male')
    plt.scatter(female_height, female_weight, color = 'r', label='Female')
    plt.legend(loc=2)
    plt.contour(x_mesh, y_mesh, male_qda_criteria, colors='b')
    plt.contour(x_mesh, y_mesh, female_qda_criteria, colors='r')
    qda_decision_boundary = np.asarray(male_qda_criteria) - np.asarray(female_qda_criteria)
    plt.contour(x_mesh, y_mesh, qda_decision_boundary, colors='k', levels=[0])
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('QDA Contours and Decision Boundary')
    plt.savefig('qda.pdf')
    plt.show()
    return (mu_male,mu_female,cov,cov_male,cov_female)
    

def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    # apply the decision formulas for LDA
    N = y.shape[0]
    cov_inverse = np.linalg.inv(cov)
    male_lda_criteria = np.dot((np.dot(cov_inverse, mu_male)).T, x.T) - 1/2*np.dot(mu_male.T, np.dot(cov_inverse, mu_male))
    female_lda_criteria = np.dot((np.dot(cov_inverse, mu_female)).T, x.T) - 1/2*np.dot(mu_female.T, np.dot(cov_inverse, mu_female))
    
    # correct if male > female and label is male then correct, or male < female and label is female 
    correct_lda=0
    for n in range(0, N):
        if male_lda_criteria[n]>=female_lda_criteria[n] and y[n]==1:
            correct_lda = correct_lda + 1
        elif male_lda_criteria[n]<=female_lda_criteria[n] and y[n]==2:
            correct_lda = correct_lda + 1
    mis_lda = 1-correct_lda/N

    male_qda_criteria = util.density_Gaussian(mu_male, cov_male, x)
    female_qda_criteria = util.density_Gaussian(mu_female, cov_female, x)
    
    correct_qda = 0   
    for n in range(0, N):
        if (male_qda_criteria[n]>=female_qda_criteria[n] and y[n]==1):
            correct_qda = correct_qda + 1
        elif (male_qda_criteria[n]<=female_qda_criteria[n] and y[n]==2):
            correct_qda = correct_qda + 1
    mis_qda = 1-correct_qda/N

    print('lda, qda')
    print(mis_lda, mis_qda)
    
    return (mis_lda, mis_qda)


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)
    

    
    
    

    
