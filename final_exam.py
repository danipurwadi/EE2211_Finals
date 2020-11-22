'''
Copyright (C) 2020, Dani Purwadi
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''
import math
import numpy as np
from colorama import Fore
from colorama import Style
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt


#@TODO Plotting of graphs lmaoz
#@TODO Linear Dependency

def main():
    print("--------------------------------------------------------")
    inp = input("What do you want me to do?\n")
    try:
        # Matrix Inverse
        if inp == '90' or inp == "inverse":
            print(f"{Fore.GREEN}{Style.BRIGHT}Matrix Inverse{Style.RESET_ALL}")
            inv_input = matrix_converter()
            inv_matrix = inverse(inv_input)
            print("The inverse matrix is:")
            print(inv_matrix)
            main()
        
        if inp =='91' or inp == "transpose":
            text = "Transpose of matrix"
            print(f"{Fore.GREEN}{Style.BRIGHT}{text}{Style.RESET_ALL}")
            inp = matrix_converter()
            transposed = inp.T
            print(f"Transposed matrix:\n{transposed}")
            main()
            
        # Find w given X and y
        if inp =='1':
            text = "Linear Regression"
            print(f"{Fore.GREEN}{Style.BRIGHT}{text}{Style.RESET_ALL}")    
            linear_regression("non-ridge", "regression")
            
        if inp =='11':
            text = "Linear Regression with Ridge Regression"
            print(f"{Fore.GREEN}{Style.BRIGHT}{text}{Style.RESET_ALL}")    
            linear_regression("ridge", "regression")
        
        if inp =='2':
            text = "Linear Classification"
            print(f"{Fore.GREEN}{Style.BRIGHT}{text}{Style.RESET_ALL}")    
            linear_regression("non-ridge", "classification")
        
        if inp =='21':
            text = "Linear Classification with Ridge Regression"
            print(f"{Fore.GREEN}{Style.BRIGHT}{text}{Style.RESET_ALL}")    
            linear_regression("ridge", "classification")
        
        if inp =='3':
            text = "Polynomial Regression"
            print(f"{Fore.GREEN}{Style.BRIGHT}{text}{Style.RESET_ALL}")    
            polynomial_regression("non-ridge", "regression")
            
        if inp =='31':
            text = "Polynomial Regression with Ridge Regression"
            print(f"{Fore.GREEN}{Style.BRIGHT}{text}{Style.RESET_ALL}")    
            polynomial_regression("ridge", "regression")
        
        if inp =='4':
            text = "Polynomial Classification"
            print(f"{Fore.GREEN}{Style.BRIGHT}{text}{Style.RESET_ALL}")    
            polynomial_regression("non-ridge", "classification")
        
        if inp =='41':
            text = "Polynomial Classification with Ridge Regression"
            print(f"{Fore.GREEN}{Style.BRIGHT}{text}{Style.RESET_ALL}")    
            polynomial_regression("ridge", "classification")
        
        if inp =='5':
            text = "Performing Gradient Descent"
            print(f"{Fore.GREEN}{Style.BRIGHT}{text}{Style.RESET_ALL}")    
            gradient_descent()
        
        if inp =='6':
            text = "MSE for Classification"
            print(f"{Fore.GREEN}{Style.BRIGHT}{text}{Style.RESET_ALL}")    
            misclassification_tree()
        
        if inp =='61':
            text = "MSE for Regression"
            print(f"{Fore.GREEN}{Style.BRIGHT}{text}{Style.RESET_ALL}")    
            regression_tree_mse()
        
        if inp =='7':
            text = "Regression Trees"
            print(f"{Fore.GREEN}{Style.BRIGHT}{text}{Style.RESET_ALL}")    
            decision_tree_regressor()
        
        if inp =='8':
            text = "Classification Trees"
            print(f"{Fore.GREEN}{Style.BRIGHT}{text}{Style.RESET_ALL}")    
            decision_tree_classifier()
        
        if inp =='9':
            text = "Evaluation Metric"
            print(f"{Fore.GREEN}{Style.BRIGHT}{text}{Style.RESET_ALL}")    
            evaluation_metric()
            
    except EOFError:
        main()
    except KeyboardInterrupt:
        main()

def evaluation_metric():
    print("Please input matrix X")
    matrix = matrix_converter()
    if len(matrix) != 2:
        pass
    else:
        TP = matrix[0][0]
        FN = matrix[0][1]
        
        FP = matrix[1][0]
        TN = matrix[1][1]
        
        tpr = TP/(TP + FN)
        fnr = FN/(TP + FN)
        
        tnr = TN/(FP + TN)
        fpr = FP/(FP + TN)
        
        accuracy = (TP + TN) / (TP + TN + FN + FP)
        precision = TP / (TP + FP)
    
    text = "\nSummary Page"
    print(f"{Fore.GREEN}{Style.BRIGHT}{text}{Style.RESET_ALL}")
    print(f"True Positive Rate: {tpr}\n")
    print(f"False Negative Rate: {fnr}\n")
    print(f"True Negative Rate: {tnr}\n")
    print(f"False Positive Rate: {fpr}\n")
    
    print(f"Accuracy: {accuracy}\n")
    print(f"Precision: {precision}\n")
        
    main()
    
def decision_tree_classifier():
    try:
        print("Please input matrix X")
        X = matrix_converter()
        print("Please input matrix Y")
        y = matrix_converter()
        criteria = input("Enter criteria of tree:\n")
        max_dept = int(input("Enter maximum depth:\n"))
        
        test_status = input("Add Test cases? (y/n)\n")
        
        dtree = DecisionTreeClassifier(criterion=criteria, max_depth=max_dept)
        dtree = dtree.fit(X, y) # reshape necessary because tree expects 2D array 
        
        y_trainpred = dtree.predict(X)
        y_train_mse = mean_squared_error(y, y_trainpred)
        
        if test_status == "y":
            print("Please input your test case:")
            X_test = matrix_converter()
            # Auto add bias as this is a polynomial regression
            y_testpred = dtree.predict(X_test)
            
            mse_status = input("MSE? (y/n)\n")
            if mse_status == "y":
                print("Please input y test value:")
                y_test = matrix_converter()
                y_test_mse = mean_squared_error(y_test, y_testpred)
        
        text = "\nSummary Page"
        print(f"{Fore.GREEN}{Style.BRIGHT}{text}{Style.RESET_ALL}")
        print(f"Train MSE:\n{y_train_mse}\n")

        if test_status == "y":
            print(f"Test MSE:\n{y_test_mse}\n")
        tree.plot_tree(dtree)
        
        main()
    except EOFError:
        main()
    except KeyboardInterrupt:
        main()
        
def decision_tree_regressor():
    try:
        print("Please input matrix X")
        X = matrix_converter()
        print("Please input matrix Y")
        y = matrix_converter()
        criteria = input("Enter criteria of tree:\n")
        max_dept = int(input("Enter maximum depth:\n"))
        
        test_status = input("Add Test cases? (y/n)\n")
        
        dtree = DecisionTreeRegressor(criterion=criteria, max_depth=max_dept)
        dtree.fit(X, y) # reshape necessary because tree expects 2D array 
        y_trainpred = dtree.predict(X)
        y_train_mse = mean_squared_error(y, y_trainpred)
        
        if test_status == "y":
            print("Please input your test case:")
            X_test = matrix_converter()
            # Auto add bias as this is a polynomial regression
            y_testpred = dtree.predict(X_test)
            
            mse_status = input("MSE? (y/n)\n")
            if mse_status == "y":
                print("Please input y test value:")
                y_test = matrix_converter()
                y_test_mse = mean_squared_error(y_test, y_testpred)
        
        text = "\nSummary Page"
        print(f"{Fore.GREEN}{Style.BRIGHT}{text}{Style.RESET_ALL}")
        print(f"Train MSE:\n{y_train_mse}\n")

        if test_status == "y":
            print(f"Test MSE:\n{y_test_mse}\n")
            
        plt.scatter(X, y, c='steelblue', s=20)
        plt.plot(X, dtree, color='black', lw=2, label='scikit-learn')
        main()
    except EOFError:
        main()
    except KeyboardInterrupt:
        main()

def regression_tree_mse():
    print("Please input Distribution Matrix")
    mse_above = 0
    mse_below = 0
    mse_root = 0
    y_above = []
    y_below = []
    y_list = []
    matrix = matrix_converter()
    
    threshold = float(input("Enter desired threshold: \n"))
    for i in range (len(matrix)):
        y_list.append(matrix[i][1])
        if matrix[i][0] > threshold:
            y_above.append(matrix[i][1])
        else:
            y_below.append(matrix[i][1])
            
    ybar_above = sum(y_above) / len(y_above)
    for j in range (len(y_above)):
        mse_above += (y_above[j] - ybar_above)**2
    mse_above = mse_above / len(y_above)
    
    ybar_below = sum(y_below) / len(y_below)
    for k in range (len(y_below)):
        mse_below += (y_below[k] - ybar_below)**2
    mse_below = mse_below / len(y_below)
    
    mse_overall = len(y_above) / len(matrix) * mse_above + len(y_below) / len(matrix) * mse_below
    
    ybar_root = sum(y_list) / len(y_list)
    
    for a in range (len(y_list)):
        mse_root += (y_list[a] - ybar_root)**2
    mse_root = mse_root / len(y_list)
    
    text = "\nSummary Page"
    print(f"{Fore.GREEN}{Style.BRIGHT}{text}{Style.RESET_ALL}")
    
    print("\ny bar above: " + str(ybar_above))
    print("\nmse above: " + str(mse_above))
    
    print("\ny bar below: " + str(ybar_below))
    print("\nmse below: " + str(mse_below))
    
    print("\nmse overall: " + str(mse_overall))
    
    print("\n-------")
    print("\nroot y bar: " + str(ybar_root))   
    print("\nmse root: " + str(mse_root))
    print("\nMSE improvement: " + str(mse_overall - mse_root))

def misclassification_tree():
    root_gini = 0
    root_entropy = 0
    root_miss = 0

    over_gini = 0
    over_entropy = 0
    over_miss = 0
    gini_lst = []
    entropy_lst = []
    miss_lst = []
    total = []
    total_leaves = []
    
    print("Please input Leaf Distribution")
    leaves = matrix_converter()
    
    root_status = input("Add root? (y/n)\n")
    
    if root_status == "y":
        root_gini = 1
        root_entropy = 0
        root_miss = 1
        root = []
        print("Please input Root Distribution")
        roots = matrix_converter()
        sum_roots = sum(roots[0])
        for a in range (len(roots[0])):
            p = roots[0][a] / sum_roots
            root.append(p)
            root_gini -= p**2
            if p == 0:
                root_entropy -= 0
            else:
                root_entropy -= p * math.log2(p)
        root_miss -= max(root)
    
    for i in range (len(leaves)):
        gini = 1
        entropy = 0
        miss = 1
        leaf = []
        total.append(np.sum(leaves[i]))
        for j in range (len(leaves[0])):
            p = leaves[i][j] / total[i]
            leaf.append(p)
            
            gini -= p**2
            if p == 0:
                entropy -= 0
            else:
                entropy -= p * math.log2(p)         
        miss -= max(leaf)
        total_leaves.append(leaf)
        gini_lst.append(gini)
        entropy_lst.append(entropy)
        miss_lst.append(miss)
    
    for k in range (len(leaves)):
        over_gini += (total[k] / sum(total)) * gini_lst[k]
        over_entropy += (total[k] / sum(total)) * entropy_lst[k]
        over_miss += (total[k] / sum(total)) * miss_lst[k]
        
    text = "\nSummary Page"
    print(f"{Fore.GREEN}{Style.BRIGHT}{text}{Style.RESET_ALL}")
    
    print("\nroot gini: " + str(root_gini))
    print("\ngini: " + str(gini_lst))
    print("\noverall gini: " + str(over_gini))
    print("\ngini improvement: " + str(over_gini - root_gini))
    
    print("\n-------")
    print("\nroot entropy: " + str(root_entropy))   
    print("\nentropy: " + str(entropy_lst))
    print("\noverall entropy: " + str(over_entropy))
    print("\nentropy improvement: " + str(over_entropy - root_entropy))
    
    print("\n-------")
    print("\nroot missclass: " + str(root_miss))
    print("\nmissclass: " + str(miss_lst))
    print("\noverall missclass: " + str(over_miss))
    print("\nmiss class improvement: " + str(over_miss - root_miss))
    main()

def gradient_descent():
    text = "REMINDER TO CHANGE EQUATION"
    print(f"{Fore.RED}{Style.BRIGHT}{text}{Style.RESET_ALL}")
    eqn = lambda x : 4*x**3
    x_init = int(input("Initial x value:\n"))
    learning_rate = float(input("Learning rate:\n"))
    iterations = int(input("Number of iterations:\n"))
    
    for i in range (iterations):
        x_new = x_init - learning_rate * eqn(x_init)
        print(x_new)
        x_init = x_new
    main()
    
def polynomial_regression(ridge, classification):
    try:
        det = None
        y_predict = None
        mse_value = None
        print("Please input matrix X")
        X = matrix_converter()
        print("Please input matrix Y")
        y = matrix_converter()
        power = int(input("Polynomial power:\n"))
        test_status = input("Add Test cases? (y/n)\n")
        
        poly=PolynomialFeatures(power)
        X = poly.fit_transform(X)
        
        print(X)
        if ridge == "non-ridge":
            if len(X) > len(X[0]):
                print("\nOverdetermined Case")
                w = np.linalg.inv(X.T @ X)@X.T@y
                print(w)
                
            elif len(X) < len(X[0]):
                print("\nUnderdetermined Case")
                w = X.T @ np.linalg.inv(X @ X.T)@ y
            else:
                print("\nEven-determined case")
                det = np.linalg.det(X)
                if det <= 0.0000000001 and det >= -0.0000000001:
                    det = 0
                    exception_type = "Determinant of X is zero"
                    print(f'{Fore.RED}{Style.BRIGHT}{exception_type}{Style.RESET_ALL}')
                    main()
                w = np.linalg.inv(X) @ y
       
        elif ridge == "ridge":
            lam = float(input("Value for Lambda: \n"))
            if len(X) > len(X[0]):
                print("\nOverdetermined Case... Using Primal Form")
                reg = lam * np.eye(X.shape[1])
                w = np.linalg.inv(X.T@X + reg)@X.T@y
                
            elif len(X) < len(X[0]):
                print("\nUnderdetermined Case... Using Dual Form")
                reg = lam * np.eye(X.shape[0])
                w = X.T @ np.linalg.inv(X @ X.T + reg) @ y
                
            else:
                print("\nEven-determined case")
                det = np.linalg.det(X)
                if det <= 0.0000000001 and det >= -0.0000000001:
                    det = 0
                    exception_type = "Determinant of X is zero"
                    print(f'{Fore.RED}{Style.BRIGHT}{exception_type}{Style.RESET_ALL}')
                    main()
                w = np.linalg.inv(X) @ y
        
        
        if test_status == "y":
            print("Please input your test case:")
            X_test = matrix_converter()
            # Auto add bias as this is a polynomial regression
            X_test = poly.fit_transform(X_test)
            print(X_test)
            y_predict = X_test @ w
            if classification == "classification":
                if (len(y_predict[0]) == 1):
                    print("Performing classification of 2 classes")
                    y_class_predict = np.sign(y_predict)
                else:
                    print("Performing multi-class classification")
                    y_class_predict = [[1 if y == max(x) else 0 for y in x] for x in y_predict ]
            
            mse_status = input("MSE? (y/n)\n")
            if mse_status == "y" and classification == "regression":
                print("Please input y test value:")
                y_test = matrix_converter()
                mse_value = mean_squared_error(y_test, y_predict)
        
        y_predict_train = X @ w
        mse_train = mean_squared_error(y, y_predict_train)
                                       
                                       
        text = "\nSummary Page"
        print(f"{Fore.GREEN}{Style.BRIGHT}{text}{Style.RESET_ALL}")
        print(f"w:\n{w}\n")
        print(f"Determinant of X:\n{det}\n")
        print(f"y_predict:\n{y_predict}\n")
        if classification == "classification":
            print(f"y_class_predict:\n{y_class_predict}")
        print(f"Train MSE:\n{mse_train}")
        print(f"Test MSE:\n{mse_value}\n")
    
        main()

    except EOFError:
        main()
    except KeyboardInterrupt:
        main()

def linear_regression(ridge, classification):
    try:
        det = None
        y_predict = None
        mse_value = None
        print("Please input matrix X")
        X = matrix_converter()
        print("Please input matrix Y")
        y = matrix_converter()
        bias_status = input("Add Bias? (y/n)\n")
        test_status = input("Add Test cases? (y/n)\n")
        
        if bias_status == "y":
            print("Bias is added")
            X = np.append(np.ones((len(X), 1)), X, axis=1)
            print(f'Your new X is:\n{X}\n')
        
        if ridge == "non-ridge":
            if len(X) > len(X[0]):
                print("\nOverdetermined Case")
                w = np.linalg.inv(X.T @ X)@X.T@y
                
            elif len(X) < len(X[0]):
                print("\nUnderdetermined Case")
                w = X.T @ np.linalg.inv(X @ X.T)@ y
            else:
                print("\nEven-determined case")
                det = np.linalg.det(X)
                if det <= 0.0000000001 and det >= -0.0000000001:
                    det = 0
                    exception_type = "Determinant of X is zero"
                    print(f'{Fore.RED}{Style.BRIGHT}{exception_type}{Style.RESET_ALL}')
                    main()
                w = np.linalg.inv(X) @ y
       
        elif ridge == "ridge":
            lam = float(input("Value for Lambda: \n"))
            if len(X) > len(X[0]):
                print("\nOverdetermined Case... Using Primal Form")
                reg = lam * np.eye(X.shape[1])
                w = np.linalg.inv(X.T@X + reg)@X.T@y
                
            elif len(X) < len(X[0]):
                print("\nUnderdetermined Case... Using Dual Form")
                reg = lam * np.eye(X.shape[0])
                w = X.T @ np.linalg.inv(X @ X.T + reg)@ y
                
            else:
                print("\nEven-determined case")
                det = np.linalg.det(X)
                if det <= 0.0000000001 and det >= -0.0000000001:
                    det = 0
                    exception_type = "Determinant of X is zero"
                    print(f'{Fore.RED}{Style.BRIGHT}{exception_type}{Style.RESET_ALL}')
                    main()
                w = np.linalg.inv(X) @ y
        
        if test_status == "y":
            print("Please input your test case:")
            X_test = matrix_converter()
            if bias_status == "y":
                X_test = np.append(np.ones((len(X_test), 1)), X_test, axis=1)
            y_predict = X_test @ w
            
            if classification == "classification":
                if (len(y_predict[0]) == 1):
                    print("Performing classification of 2 classes")
                    y_class_predict = np.sign(y_predict)
                else:
                    print("Performing multi-class classification")
                    y_class_predict = [[1 if y == max(x) else 0 for y in x] for x in y_predict ]
                    
            mse_status = input("MSE? (y/n)\n")
            if mse_status == "y":
                print("Please input y true value:")
                y_true = matrix_converter()
                mse_value = mean_squared_error(y_true, y_predict)
        
        y_predict_train = X @ w
        mse_train = mean_squared_error(y, y_predict_train)
        
        text = "\nSummary Page"
        print(f"{Fore.GREEN}{Style.BRIGHT}{text}{Style.RESET_ALL}")
        print(f"w:\n{w}\n")
        print(f"Determinant of X:\n{det}\n")
        print(f"y_predict:\n{y_predict}\n")
        if classification == "classification":
            print(f"y_class_predict:\n{y_class_predict}")
        
        print(f"Train MSE:\n{mse_train}")
        print(f"Test MSE:\n{mse_value}\n")    
        main()

    except EOFError:
        main()
    except KeyboardInterrupt:
        main()
    
def inverse(X):
    try:
        np.linalg.inv(X)
    except Exception as err:
        exception_type = type(err).__name__
        print(f'{Fore.RED}{Style.BRIGHT}{exception_type}{Style.RESET_ALL}') 
        main()
    return np.linalg.inv(X)

def matrix_converter():
    contents = []
    while True:
        if len(contents) == 0:
            line = input("Please enter your matrix:\n")
        else:
            line = input()
        if line == '':
            break
        line_array = list(map(lambda x: float(x), line.split()))
        contents.append(line_array)
    
    numpy_content = np.array(contents)
    return numpy_content


if __name__ == '__main__':
    main()
