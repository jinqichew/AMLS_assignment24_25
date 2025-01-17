import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def log_reg_A():
    print("\nTask A Logistic Regression:")
    # load data
    data = np.load("Datasets/breastmnist.npz")
    train_images = data['train_images']
    train_labels = data['train_labels']
    val_images = data['val_images']
    val_labels = data['val_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

    # preprosess
    X_train = train_images.reshape(train_images.shape[0], -1) / 255.0
    X_val = val_images.reshape(val_images.shape[0], -1) / 255.0
    X_test = test_images.reshape(test_images.shape[0], -1) / 255.0

    y_train = train_labels.ravel()
    y_val = val_labels.ravel()
    y_test = test_labels.ravel()

    # train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # evaluate validation and test sets
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"Task A Logistic Regression Validation Accuracy: {val_acc:.4f}")
    print(f"Task A Logistic Regression Test Accuracy: {test_acc:.4f}")

    print(classification_report(y_test, test_pred))
    print(confusion_matrix(y_test, test_pred))