import matplotlib.pyplot as plt
import numpy as np

def add_intercept(x):
    #x is a numpy array. This function will add a column of 1s in the very first column of the array

    new_x = np.zeros((x.shape[0],x.shape[1]+1), dtype=x.dtype)
    #x.shape[0] >> No. of rows
    #x.shape[1]+1 >> No. of columns

    new_x[:,0] = 1 #The entire first column is set to 1
    # ':' symbol means all of the elements along this dimension
    new_x[:, 1:] = x
    #All the rows for all the columns except the 0th column (i.e. the first column) will be assigned the elements of x
    return new_x

def load_dataset(csv_path,label_col='y', add_intercept=False):
    
    def add_intercept_fn(x):
        global add_intercept
        return add_intercept(x)
    
    allowed_label_cols = ('y','t')
    if label_col not in allowed_label_cols:
        raise ValueError('Invalid label_col: {}. Expected {}'.format(label_col,allowed_label_cols))
    
    #Open csv file and assign headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')
        #'r' means to open the file in read-only form
        #The very first line of the file is read and the names of the columns are stored in the header array(without the commas)


    #Identifying the indexes of cols that correspond to x and y
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]

    #Extract the data into arrays
    inputs = np.loadtxt(csv_path, delimiter=',',skiprows=1,usecols=x_cols)
    labels = np.loadtxt(csv_path,delimiter=',',skiprows=1,usecols=l_cols)

    #THIS STEP IS STILL NOT CLEAR!!!
    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)
        #This if statement checks if the inputs array has a dimensional`ity of 1. If it does, it means that the data has been loaded as a 1-dimensional array (a single column). To ensure compatibility with later calculations, the np.expand_dims() function is used to add an extra dimension to the inputs array. The -1 argument indicates that the new dimension should be added at the last position. i.e. if the previous shape is (n,) The new shape will be (n,1)

    #Adds intercepts
    if add_intercept:
        inputs = add_intercept_fn(inputs)
    

    return inputs, labels

def plot(x,y,theta,save_path=None,correction=1.0):
    """Plot dataset and fitted logistic regression parameters.
    Args:
        x: Matrix of training examples, one per row.
        y: Vector of labels in {0, 1}.
        theta: Vector of parameters for logistic regression model.
        save_path: Path to save the plot.
        correction: Correction factor to apply (Problem 2(e) only).
    """

    #Plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y==0, -2], x[y==0, -1], 'go', linewidth=2)
    # x[y == 1, -2] selects the values from the second-to-last column (-2) of the x array where the corresponding elements in the y array are equal to 1.
    #WHAT DOES LINEWIDTH SPECIFY????
    #PLOT THE SIGMOID FUNCTION ITSELF
    


    #CLARIFICATION FOR THE DECISION BOUNDARY 
    
    #Our decision threshold is 0.5. The function h(x) gives 0.5 when theta.T.dot(x) is == 0
    #So, for our given features and parameters, the decision boundary is when y = 0
    # ===> theta[0] + theta[1]x1 + theta[2]x2 = 0  where x1 = values on x_axis,   x2 = values on y_axis
    

    # Plot decision boundary (found by solving for theta^T x = 0)
    margin1 = (max(x[:, -2]) - min(x[:, -2]))*0.2
    margin2 = (max(x[:, -1]) - min(x[:, -1]))*0.2
    #Margin refers to the additional space added around the range of values on a specific axis in a plot
    
    x1 = np.arange(min(x[:, -2])-margin1, max(x[:, -2])+margin1, 0.01)
    #This line generates an array of x-axis values (x1) using np.arange(). The arange() function generates values ranging from min(x[:, -2])-margin1 to max(x[:, -2])+margin1 (inclusive) with a step size of 0.01.
    #This line essentially creates a range of x-axis values that covers the extent of the data points plus the calculated margins.

    #The value of x2 is calculated from the above equation using the x1 array.
    x2 = -(theta[0] / theta[2] * correction + theta[1] / theta[2] * x1)
    
    #The decision boundary is then plotted.
    plt.plot(x1, x2, c='red', linewidth=2)

    #.lim() determines the minimum and maximum values for the x-and-y-axis, considering the data points and the calculated margin.
    plt.xlim(x[:, -2].min()-margin1, x[:, -2].max()+margin1)
    plt.ylim(x[:, -1].min()-margin2, x[:, -1].max()+margin2)
    

    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')
    if save_path is not None:
        plt.savefig(save_path)