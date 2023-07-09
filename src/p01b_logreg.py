import numpy as np
import util
from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """

    x_train, y_train = util.load_dataset(train_path,add_intercept=True)

    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)

    log_reg = LogisticRegression()
    log_reg.fit(x_train,y_train)
    prediction = log_reg.predict(x_valid)

    


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***

        m,n = x.shape
        #m is the number of training examples
        #n is the number of features. Number of features == Number of Parameters (including the intercept column of 1)

        #If theta does not have any specific initial value, we will use zero vector with the dimensions of Number of features.
        self.theta = np.zeros(n)

        #In the following mathematical expressions, it is necessary to keep in mind the difference between element-wise multiplication and matrices' dot product.
        def h_x(x, theta):
            return (1/(1+np.exp(-x.dot(theta))))
        def gradient(y,h_x,x,m):
            return ((x.T.dot((h_x - y)))/m)
        def hessian(x,m,hx):
            return (((x.T * hx * (1-hx)).dot(x))/m)
        def new_theta(old_theta, gradient,hessian):
            return(old_theta - np.linalg.inv(hessian).dot(gradient))
        
        diff = False
        while (diff==False ):
            old_theta = np.copy(self.theta)
            hx = h_x(x,self.theta)
            grad_ = gradient(y,hx,x,m)
            hessian_ = hessian(x,m,hx)
            self.theta = new_theta(self.theta, grad_, hessian_)
            diff = np.linalg.norm(self.theta-old_theta, ord=1) < self.eps
        #Norm of a vector: A vector norm, also known as a vector length or vector magnitude, is a mathematical function that assigns a non-negative value to a vector.
            #There are norms of different order. Norm of order 1 is L1 Norm. It adds the absolute value of all elements of a vector.

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return (1/(1+np.exp(-x.dot(self.theta))))
        # *** END CODE HERE ***

