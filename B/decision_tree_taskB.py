import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def decision_tree_B():
    print("\nTask B Decision Tree:")

    # load data
    data = np.load("Datasets/bloodmnist.npz")
    train_images = data['train_images']
    train_labels = data['train_labels']
    val_images = data['val_images']
    val_labels = data['val_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

    # preprocess
    X_train = train_images.reshape(train_images.shape[0], -1) / 255.0
    X_val = val_images.reshape(val_images.shape[0], -1) / 255.0
    X_test = test_images.reshape(test_images.shape[0], -1) / 255.0

    y_train = train_labels.ravel()
    y_val = val_labels.ravel()
    y_test = test_labels.ravel()

    # standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # train decision tree
    dt = DecisionTreeClassifier(max_depth=None, random_state=1)
    dt.fit(X_train, y_train)

    # evaluate on validation and test sets
    val_pred = dt.predict(X_val)
    test_pred = dt.predict(X_test)

    val_accuracy = accuracy_score(y_val, val_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    print(f"Task B Decision Tree Validation Accuracy: {val_accuracy:.4f}")
    print(f"Task B Decision Tree Test Accuracy: {test_accuracy:.4f}")

    print(classification_report(y_test, test_pred))
    print(confusion_matrix(y_test, test_pred))