from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from helper import get_data
from helper import get_training_data, movies_to_data



def least_squares_classifier(X_data, Y_data):

    linear_classifier = LinearRegression()
    linear_classifier.fit(X_data, Y_data)

    return linear_classifier
    
def neural_network_classifier(X_data, Y_data):
    
    neural_network = MLPClassifier(hidden_layer_sizes=(4, 4), activation='logistic', 
        solver='sgd', alpha=1e-5, learning_rate='adaptive', max_iter=1000)
    neural_network.fit(X_data, Y_data)

    return neural_network

