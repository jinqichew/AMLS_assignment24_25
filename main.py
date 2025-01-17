from A import random_forest_taskA
from B import cnn_taskB

from A import random_forest_tuning
from A import logistic_regression_taskA
from A import knn_taskA
from A import svm_taskA
from A import decision_tree_taskA
from A import cnn_taskA

from B import logistic_regression_taskB
from B import knn_taskB
from B import svm_taskB
from B import decision_tree_taskB
from B import random_forest_taskB

def main():
    # the chosen model for task A and task B
    random_forest_taskA.random_forest_A()
    cnn_taskB.cnn_B()

    # uncomment the code below if needed to test

    # # hyperparameter tuning code for random forest in task A
    # random_forest_tuning.rf_tuning()

    # # other tested models in task A
    # logistic_regression_taskA.log_reg_A()
    # knn_taskA.knn_A()
    # svm_taskA.svm_A()
    # decision_tree_taskA.decision_tree_A()
    # cnn_taskA.cnn_A()

    # # other tested model in task B
    # logistic_regression_taskB.log_reg_B()
    # knn_taskB.knn_B()
    # svm_taskB.svm_B()
    # decision_tree_taskB.decision_tree_B()
    # random_forest_taskB.random_forest_B()


if __name__ == "__main__":
    main()
