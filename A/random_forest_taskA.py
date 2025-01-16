import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def random_forest_A():
    print("\nTask A Random Forest:")

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

    # train random forest, random state set to 1 to reproduce result
    model = RandomForestClassifier(n_estimators=100, random_state=13)
    model.fit(X_train, y_train)

    # evaluate on validation and test sets
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"Task A Random Forest Validation Accuracy: {val_acc:.4f}")
    print(f"Task A Random Forest Test Accuracy: {test_acc:.4f}")

    print(classification_report(y_test, test_pred))
    print(confusion_matrix(y_test, test_pred))

