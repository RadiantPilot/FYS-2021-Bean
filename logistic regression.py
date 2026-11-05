import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


  
df = pd.read_excel("Dry_Bean_Dataset.xlsx")

#print(df.head())
#print(df.info())
#print(df["Class"].value_counts())

X = df.drop("Class", axis=1).values   #features and target
y = df["Class"].values

classes = np.unique(y)    #makes the classes to numbers from 0 - 6 
y_encoded = np.searchsorted(classes, y)

# One-hot encoding, Y are the actual classes for the beans, we can then use the model to predict a class for each bean and then use Y to tell it how
K = len(classes)     #how much error there is to update the weights. 
Y = np.eye(K)[y_encoded]

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size = 0.2,
    random_state=1,
    stratify=y
)

#standardises the data 
mean = np.mean(X_train,axis=0)
std = np.std(X_train,axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

def softmax(Z):      #gives us probailities between 0 and 1, where the sum of all probabilities for all classes is 1 
    exp_Z = np.exp(Z-np.max(Z, axis = 1, keepdims = True))
    return exp_Z / np.sum(exp_Z, axis = 1, keepdims= True)

def compute_cost(Y, P):    #cross-entropy loss, measures how long the models predicted probabilities are from the actual classes.
    epsilon = 1e-15            #the goal is to reduce this value
    P = np.clip(P, epsilon, 1 - epsilon)

    cost = -np.sum(Y * np.log(P)) / len(Y)

    return cost

def logistic_regression(X,Y,learning_rate = 0.1 , epochs = 3000):
    N, n_features = X.shape
    K = Y.shape[1]

    #initialize weights and bias
    W = np.zeros((n_features,K))
    b = np.zeros(K)

    costs = []

    for epoch in range(epochs):
        Z = X @ W + b      #computes the linear combination
        P = softmax(Z)      #makes the linear combination into probabilities

        cost = compute_cost(Y,P)    #computes the cost
        costs.append(cost)

        dW = (X.T @ (P-Y)) / N       #gradient descent, helps us find the weight W and bias b which gives the lowest cost. 
        db = np.sum(P-Y,axis = 0) / N

        if epoch == 0:
            print("Initial cost:", cost)
            print("Gradient W:", np.linalg.norm(dW))
            print("Gradient b:", np.linalg.norm(db))

        W -= learning_rate * dW      #updates weight and bias
        b -= learning_rate * db

        if epoch % 100 == 0:
            print(f'Epoch: {epoch}, Cost: {cost:.4f}')

    print("Final cost:", costs[-1])
    print("Weight magnitude:", np.linalg.norm(W))

    return W, b, costs

W, b, costs = logistic_regression(    #trains the model
    X_train,
    Y_train,
    learning_rate=0.1,
    epochs = 6000
)

plt.plot(costs)      #plots the cost during training, when the cross entropy loss converges our model is finished learning
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Cost during training")
plt.grid()
plt.show()

def predict(X, W, b):    #with the weight and bias found we can predict the class of a bean
    Z = X @ W + b
    P = softmax(Z)

    predictions = np.argmax(P, axis=1)   #argmax finds the index with the highest probability

    return predictions

y_pred = predict(X_test, W, b)    #gets predictions for the test set

y_test_labels = np.argmax(Y_test,axis = 1)   #finds the highest index in each row, and turns the one hot encoding to the original numerical
                                                #class indexes by finding the 1 in each row. 
cm = confusion_matrix(y_test_labels, y_pred)   #makes a confusion matrix with sklearn that compares the true class of a bean and the predicted one

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

disp.plot(xticks_rotation=45)   
plt.show()

accuracy = np.mean(y_pred == y_test_labels) #compares the numpy elements with each other and finds the average 

print(f'Accuracy: ', accuracy)

# Training predictions
y_train_pred = predict(X_train, W, b)

train_accuracy = np.mean(
    y_train_pred == np.argmax(Y_train, axis=1)
)

# Test predictions
y_test_pred = predict(X_test, W, b)

test_accuracy = np.mean(
    y_test_pred == np.argmax(Y_test, axis=1)
)

print("Training accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)
