# EE2211 Calculator

This programme was created on AY20/21 Semester 1 as part of an effort to lessen the calculation burden on myself (or other students). If you are using this for future AYs do make sure that the syllabus is still the same.

Functions supported:
- Matrix manipulation
    - Inverse a matrix
    - Transpose
    - Rank of matrix
    - Linear Independence of Matrix

## 1. Setting Up
1. Clone the repository or simply copy paste the code on `final_exam.py` to an empty python file
2. Open the file using Spyder (recommended) or Jupyter Notebook or any other IDE that supports the packages
3. Run the file using the play button on Spyder
4. On the terminal, the programme will ask "What do you want to do?", and your setup is complete

Note: If when running the file you encounter `ModuleNotFoundError: No module named 'name_of_module'` error, simply install the package using `pip3 install name_of_module`

## 2. Guide to using the Program
1. Once you run the program, you will be directed to the main menu function.
2. Enter the ID of function you want to access (full list in Annex A).
3. Follow the prompts given. If you need further clarification, you can read annex B.
    - Note that there is no undo button, once you have entered a value for a prompt, it cannot be deleted. 
    - To return to the main menu, press CTRL-C 
4. Output will be automatically printed and you will be redirected back to the main menu.

## 3. How to Input a Matrix
- The input method is optimised for exam condition, hence no brackets is required.
- Use space to separate items of the same row. 
- To enter elements of the next row, press ENTER.
- Once you are done inputting the matrix, hit ENTER twice.

Example:
```
Example matrix:
[[0,1],
[2,3]]

Input:
0 1
2 3
```


## Annex A: Table of Functions
| ID      | Name |
| ----------- | ----------- |
| 01   | Inverse a matrix|
| 02   | Transpose a matrix|
| 03   | Get rank of a matrix|
| 04   | Get independent rows in a matrix|
| 05   | Matrix multiplication|
| 06   | Get determinant of matrix|
| 1    | Linear Regression|
| 11   | Linear Regression with Ridge|
| 2    | Linear Classification|
| 21   | Linear Classification with Ridge|
| 3    | Polynomial Regression|
| 31   | Polynomial Regression with Ridge|
| 4    | Polynomial Classification|
| 41   | Polynomial Classification with Ridge|
| 5    | Performing Gradient Descent|
| 6    | MSE for Classification|
| 61   | MSE for Regression|
| 7    | Regression Trees|
| 8    | Classification Trees|
| 9    | Evaluation Metric|
| 10   | K Classification|

## Annex B: Function Details

### 01 Matrix Inverse
Input: 
- Any Matrix (see Section 3 for input guide)

Output: 
- Inverse of input matrix
- LinAlgError if input cannot be inverted

### 02 Matrix Transpose
Input: 
- Any Matrix (see Section 3 for input guide)

Output: 
- Transpose of input matrix

### 03 Matrix Rank
Input: 
- Any Matrix (see Section 3 for input guide)

Output: 
- Return the rank of the input matrix

### 04 Linear Independence
Input: 
- Any Matrix

Output: 
- Return the independent rows of the input matrix

### 05 Matrix Multiplication
Input: 
- First matrix
- Second matrix 

Output: 
- Return the multiplication result of the 2 matrices

### 06 Matrix Determinant
Input: 
- Any matrix

Output: 
- Determinant of the matrix (if available)
- Print out 'non-invertible' if matrix is not invertible

### 1 Linear Regression
Input: 
- Matrix X
- Matrix Y
- Bias (optional)
- Test Cases Matrix (optional). Adding test case will automatically perform calculation using the w obtained to the test matrix.
- Calculate MSE (only if test case is selected). Calculate the MSE of training and test result.

Output: 
- Status of matrix X (Under or over determined case)
- w value
- Determinant of matrix X
- Prediction of y (if test case is given)
- Training MSE
- Test MSE (if test case is given)

### 11 Linear Regression with ridge
Input:
- Same as Linear Regression, but with additional input for Lambda

Output:
- Same with Linear Regression

### 2 Linear Classification
Input: 
- Matrix X
- Matrix Y
- Bias (optional)
- Test Cases Matrix (optional). however, for classification you technically should have a test case.
- Calculate MSE (only if test case is selected). Calculate the MSE of training and test result.

Output: 
- Status of matrix X (Under or over determined case)
- Classification status (multi class or 2-class)
- w value
- Determinant of matrix X
- Prediction of y (if test case is given)
- Classification result
- Training MSE
- Test MSE (if test case is given)

### 2 Linear Classification with ridge
Input:
- Same as Linear Classification, but with additional input for Lambda

Output:
- Same with Linear Classification

