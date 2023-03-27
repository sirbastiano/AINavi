import numpy as np
from scipy.linalg import block_diag
from typing import List

def prompt_F(N: int, dt: float):
    """
    1. The code above creates a 3N + 6 x 3N + 6 identity matrix, which is the offset state vector.
    2. The code above then sets the values in the diagonal of F to 1, with the exception of the velocity components in the diagonal.
    3. The code above then sets the velocity components of the diagonal of F to the time step, dt. 

    Args:
        N (int): State vector dimension
        dt (float): integration time step

    Returns:
        F: the state transition matrix
    """    
    F = np.eye(3 * N + 6)
    iX, iY, iZ, iVx, iVy, iVz = 0, 1, 2, 3, 4, 5
    F[iX, iVx] = dt
    F[iY, iVy] = dt
    F[iZ, iVz] = dt
    return F


def prompt_z_f(x_f: np.array, x_c: np.array) -> np.array:
    """ This function calculates the vector between the crater and the camera center.

    Args:
        x_f (np.array): x,y,z of crater matched|cartesian
        x_c (np.array): x,y,z of camera center|cartesian

    Returns:
        np.array: vector between crater and camera center
    """    
    x_f = np.array(x_f)
    x_c = np.array(x_c)  # TODO: inserisci if
    tmp = x_f - x_c
    return tmp / np.linalg.norm(tmp)


def residual(z_f: np.array, z_f_kminus: np.array):
    """This function calculates the residual between the crater detected and the crater 
    expected based on the pose propagation.

    Args:
        z_f (np.array): vector between crater and camera center
        z_f_kminus (np.array): vector between expected crater and camera center

    Returns:
        np.array: vector between crater detected & expected based on pose propagation
    """    
    z_f = np.array(z_f)  
    z_f_kminus = np.array(z_f_kminus)
    return (z_f - z_f_kminus)  


def prompt_H_f_i(N: int, x_f: np.array, x_c: np.array, index: int):
    """ Function to create the measurement matrix. It calculates a row of the measurement 
    matrix for a single crater.
    
    The code above does the following:
        1. It creates variables to store the 3x3 zeros matrix and 3x3 identity matrix. 
        It's important to note that the variables are created in this way. If you were 
        to write "zeros = np.zeros([3,3])", the function will not be able to access the 
        zeros variable. This is because the variable "zeros" is only local to the function. 
        The "np.zeros([3,3])" function will create a new variable called "zeros" that is 
        only local to the function. The variable "zeros" will be deleted after the function 
        is done running. We use the variable O3 to store the 3x3 zeros matrix. This is 
        because O3 is a global variable.
        
        2. It creates a variable called "z_f" and stores the return value of the prompt_z_f 
        function. The return value of the prompt_z_f function is a 1x3 matrix.
        
        3. It creates a variable called "a" and stores the dot product of the "z_f" matrix
        and the "z_f" matrix. This is because the dot product of a matrix with itself is
        the sum of the squares of the elements in the matrix.
        
        4. It creates a variable called "tmp" and stores the value of the identity matrix
        multiplied by the value of the variable "a" divided by the norm of the difference
        between the camera center and the crater. The norm of the difference between the
        camera center and the crater is the distance between the camera center and the crater.
        
        5. It creates a variable called "H_f_i" and stores the value of the "tmp" variable, 
        which is a 3x3 matrix, concatenated horizontally with the 3x3 zeros matrix. This is because
        the 3x3 zeros matrix is the 3x3 matrix that represents the state of the camera center. 
        The camera center state is represented by a 3x3 matrix because the camera center has 
        3 variables, x, y and z. This is why the identity matrix is multiplied by the value of 
        the variable "a" and divided by the norm of the difference between the camera center 
        and the crater. The value of the variable "a" is the square of the distance between 
        the camera center and the crater. The distance between the camera center and the 
        crater is the norm of the difference between the camera center and the crater. 
        By multiplying the identity matrix by the value of the variable "a" and dividing by 
        the norm of the difference between the camera center and the crater, we get the value 
        of the variable "tmp". The variable "tmp" is a 3x3 matrix. The 3x3 matrix that represents
        the state of the camera center is concatenated horizontally with the 3x3 zeros matrix. 
        This is because the 3x3 zeros matrix represents the state of the craters. The state of 
        the craters are represented by 3x3 matrices because each crater has 3 variables, 
        x, y and z. The 3x3 zeros matrix is concatenated horizontally with the variable "tmp" 
        until the number of times the 3x3 zeros matrix is concatenated horizontally is equal 
        to the number of craters. This is because we need to create a matrix that has the same 
        number of columns as the number of craters.
        
        6. A for loop is created. It runs N times. N is the number of craters. 

    Args:
        N (int): Number of states.
        x_f (np.array): camera center
        x_c (np.array): crater cordinates
        index (int): index of crater

    Returns:
        np.array: row of the measurement matrix for a single crater.
    """    
    x_f = np.array(x_f)
    x_c = np.array(x_c)

    O3 = np.zeros([3, 3])
    I3 = np.eye(3)
    z_f = prompt_z_f(x_f, x_c)
    a = z_f * z_f
    tmp = I3 * a / np.linalg.norm(x_f - x_c)
    H_f_i = np.hstack([-tmp, O3])
    for i in range(N):
        if i == index:
            H_f_i = np.hstack([H_f_i, tmp])
        else:
            H_f_i = np.hstack([H_f_i, O3])
    return H_f_i


def prompt_H(N: int, x_c: np.ndarray, craters_det: List[np.ndarray]) -> np.ndarray:
    """ Prompt the measurement matrix.

    Args:
        N (int): Number of states.
        x_c (np.ndarray): Camera center.
        craters_det (List[np.ndarray]): List of detected craters.

    Returns:
        np.ndarray: _description_
    """    
    x_c = np.array(x_c)  # add h rows zeros
    for i in range(N):
        x_f = np.array(craters_det[i, :])
        H_f_i = prompt_H_f_i(N, x_f, x_c, i)
        if i == 0:
            H = H_f_i
        else:
            H = np.vstack([H, H_f_i])

    R12 = np.zeros([6, 3 * N + 6])
    H = np.vstack([R12, H])
    return H


def prompt_R(N: int, sigma_pix: float) -> np.ndarray:
    """Function to create the measurement noise covariance matrix.

    Args:
        N (int): Number of states.
        sigma_pix (float): Standard deviation of pixel.

    Returns:
        np.ndarray: Matrix of measurement noise covariance.
    """    
    R = np.diag([sigma_pix for i in range(N)])
    return R


def prompt_Q(N: int, dt: float, sigma_acc: float, sigma_dat: float) -> np.ndarray:
    """Function to create the process noise covariance matrix.

    Args:
        N (int): Number of states.
        dt (float): Time step.
        sigma_acc (float): Standard deviation of acceleration.
        sigma_dat (float): Standard deviation of data.

    Returns:
        numpy.ndarray: Process noise covariance matrix.
    """    
    I3 = np.eye(3)
    O3 = np.zeros([3, 3])

    tmp1 = sigma_acc ** 2 * I3 * dt ** 2
    tmp2 = sigma_dat ** 2  # *I3 -->  3*N

    R1 = np.hstack([tmp1, tmp1 / 2 * dt])
    for i in range(N):
        R1 = np.hstack([R1, O3])

    R2 = np.hstack([tmp1 / 2 * dt, tmp1 / 4 * dt ** 2])
    for i in range(N):
        R2 = np.hstack([R2, O3])

    left = np.hstack([O3, O3])
    LEFT = np.vstack([left for i in range(N)])

    RIGHT = np.diag([tmp2 for i in range(3 * N)])

    SUB = np.hstack([LEFT, RIGHT])
    Q = np.vstack([R1, R2, SUB])
    return Q


def state_vector_create(x_initial: np.array, matched_features: np.array) -> np.array:
    """This function creates the state vector after the collection of features.

    Args:
        x_initial (np.array): the initial state vector
        matched_features (np.array): the detected features

    Returns:
        np.array: total state vector
    """    
    N = matched_features.shape[0]
    x = x_initial
    for i in range(N):
        feature = matched_features[i, :]
        x = np.hstack([x, feature])
    return x



if __name__ == "__main__":
    print("Executing main...")

