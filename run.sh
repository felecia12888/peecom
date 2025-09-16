# This script runs experiments on multiple datasets with different models.

# ## Peecom

# python main.py --dataset equipmentad --model peecom --eval-all && python main.py --dataset mlclassem --model peecom --eval-all && python main.py --dataset motorvd --model peecom --eval-all && python main.py --dataset multivariatetsd --model peecom --eval-all && python main.py --dataset sensord --model peecom --eval-all && python main.py --dataset smartmd --model peecom --eval-all
# echo "Peecom experiments completed."

## Random Forest

python main.py --dataset equipmentad --model random_forest --eval-all && python main.py --dataset mlclassem --model random_forest --eval-all && python main.py --dataset motorvd --model random_forest --eval-all && python main.py --dataset multivariatetsd --model random_forest --eval-all && python main.py --dataset sensord --model random_forest --eval-all && python main.py --dataset smartmd --model random_forest --eval-all
echo "Random Forest experiments completed."

## SVM

python main.py --dataset equipmentad --model svm --eval-all && python main.py --dataset mlclassem --model svm --eval-all && python main.py --dataset motorvd --model svm --eval-all && python main.py --dataset multivariatetsd --model svm --eval-all && python main.py --dataset sensord --model svm --eval-all && python main.py --dataset smartmd --model svm --eval-all
echo "SVM experiments completed."

## Logistic Regression

python main.py --dataset equipmentad --model logistic_regression --eval-all && python main.py --dataset mlclassem --model logistic_regression --eval-all && python main.py --dataset motorvd --model logistic_regression --eval-all && python main.py --dataset multivariatetsd --model logistic_regression --eval-all && python main.py --dataset sensord --model logistic_regression --eval-all && python main.py --dataset smartmd --model logistic_regression --eval-all
echo "Logistic Regression experiments completed."

# Dataset-based experiments (alternative organization)

## Equipment Anomaly Dataset

python main.py --dataset equipmentad --model random_forest --eval-all && python main.py --dataset equipmentad --model svm --eval-all && python main.py --dataset equipmentad --model logistic_regression --eval-all
echo "Equipment Anomaly Dataset experiments completed."

## ML Classification Energy Monthly Dataset

python main.py --dataset mlclassem --model random_forest --eval-all && python main.py --dataset mlclassem --model svm --eval-all && python main.py --dataset mlclassem --model logistic_regression --eval-all
echo "ML Classification Energy Monthly Dataset experiments completed."

## Motor Vibrations Dataset

python main.py --dataset motorvd --model random_forest --eval-all && python main.py --dataset motorvd --model svm --eval-all && python main.py --dataset motorvd --model logistic_regression --eval-all
echo "Motor Vibrations Dataset experiments completed."

## Multivariate Time Series Dataset

python main.py --dataset multivariatetsd --model random_forest --eval-all && python main.py --dataset multivariatetsd --model svm --eval-all && python main.py --dataset multivariatetsd --model logistic_regression --eval-all
echo "Multivariate Time Series Dataset experiments completed."

## Sensor Dataset

python main.py --dataset sensord --model random_forest --eval-all && python main.py --dataset sensord --model svm --eval-all && python main.py --dataset sensord --model logistic_regression --eval-all
echo "Sensor Dataset experiments completed."

## Smart Maintenance Dataset

python main.py --dataset smartmd --model random_forest --eval-all && python main.py --dataset smartmd --model svm --eval-all && python main.py --dataset smartmd --model logistic_regression --eval-all
echo "Smart Maintenance Dataset experiments completed."
