import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def svm_A():
    print("\nTask A SVM:")

    # load data
    data = np.load("Datasets/breastmnist.npz")
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

    # train svm
    svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(X_train, y_train)

    # evaluate on validation and test sets
    val_pred = svm.predict(X_val)
    test_pred = svm.predict(X_test)

    # Calculate accuracies
    val_accuracy = accuracy_score(y_val, val_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    print(f"Task A SVM Validation Accuracy: {val_accuracy:.4f}")
    print(f"Task A SVM Test Accuracy: {test_accuracy:.4f}")

    print(classification_report(y_test, test_pred))
    print(confusion_matrix(y_test, test_pred))
