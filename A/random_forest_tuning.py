import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

def rf_tuning():
    print("\nRandom Forest hyperparameter tuning:")

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


    # Validation Curve
    n_estimators_range = np.arange(50, 501, 50)

    # store cross-validated scores
    mean_scores = []

    for n in n_estimators_range:
        rf = RandomForestClassifier(n_estimators=n, random_state=24, n_jobs=-1)
        scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
        mean_scores.append(scores.mean())

    # plot the results
    import matplotlib.pyplot as plt

    plt.plot(n_estimators_range, mean_scores, marker='o')
    plt.xlabel('Number of Trees (n_estimators)')
    plt.ylabel('Cross-Validated Accuracy')
    plt.title('Validation Curve for n_estimators')
    plt.grid()
    plt.show()


    # Gridsearch
    rf = RandomForestClassifier(random_state=24)
    # define the parameter grid
    param_grid = {'n_estimators': np.arange(50, 301, 50)}

    # grid search with cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # output the best parameters and score
    print(f"Best n_estimators: {grid_search.best_params_['n_estimators']}")
    print(f"Best cross-validated accuracy: {grid_search.best_score_}")
