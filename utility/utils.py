from numba import njit
from astropy.coordinates.funcs import spherical_to_cartesian, cartesian_to_spherical
import numpy as np
import pandas as pd
import cv2
from copy import deepcopy
import glob
import math

from CMA.icp import icp

global km2px, deg2km, px2km, deg2px

km2px = 1/0.118
deg2km = 2*np.pi*1737.4/360
px2km = 0.118
deg2px = 256


def compute_pos_diff(A, B, CAMx, CAMy):
    # Compute the difference between the position of the camera center
    # A: the initial position of the camera center
    # B: the new position of the camera center
    # CAMx: the x coordinate of the camera center
    # CAMy: the y coordinate of the camera center
    hp = A
    x1_a, x2_a, x3_a = float(hp.x1), float(hp.x2), float(hp.x3)
    y1_a, y2_a, y3_a = float(hp.y1), float(hp.y2), float(hp.y3)
    r1_a, r3_a, r3_a = float(hp.r1), float(hp.r2), float(hp.r3)
    # Convert the initial position of the camera center to the absolute coordinates
    A1 = np.hstack([x1_a, y1_a])
    A2 = np.hstack([x2_a, y2_a])
    A3 = np.hstack([x3_a, y3_a])

    A = np.vstack([A1, A2, A3])

    hp = B
    x1_b, x2_b, x3_b = float(hp.lon1), float(hp.lon2), float(hp.lon3)
    y1_b, y2_b, y3_b = float(hp.lat1), float(hp.lat2), float(hp.lat3)
    r1_b, r2_b, r3_b = float(hp.r1), float(hp.r2), float(hp.r3)
    # Convert the new position of the camera center to the absolute coordinates
    x1_b_r, y1_b_r, r1_b_r = absolute2relative([x1_b, y1_b, r1_b], CAMx, CAMy)
    x2_b_r, y2_b_r, r2_b_r = absolute2relative([x2_b, y2_b, r2_b], CAMx, CAMy)
    x3_b_r, y3_b_r, r3_b_r = absolute2relative([x3_b, y3_b, r3_b], CAMx, CAMy)

    B1 = np.hstack([x1_b_r, y1_b_r])
    B2 = np.hstack([x2_b_r, y2_b_r])
    B3 = np.hstack([x3_b_r, y3_b_r])

    B = np.vstack([B1, B2, B3])

    R, t = icp(A, B)
    # Use the ICP algorithm to calculate the difference between the initial position and the new position of the camera center
    xc = t[0]
    yc = t[1]

    pos = [xc, yc]
    return pos


def img_plus_crts(img, craters_det, color="red"):
    # Plot the craters on the image
    # Input: Img:3 chanel, craters_det: np.array
    b = craters_det
    image = img.copy()
    for i in range(b.shape[0]):

        r = b[i][2]
        x_c, y_c = b[i][0], b[i][1]

        center_coordinates = (int(x_c), int(y_c))
        radius = int(r)
        if color == "red":
            color = (255, 0, 0)
        elif color == "green":
            color = (0, 255, 0)

        thickness = 2
        cv2.circle(image, center_coordinates, radius, color, thickness)
    return image


def eu_dist(x: tuple[int, int], y: tuple[int, int]) -> float:
    """ Calculate Euclidean distance between two points """
    x1, y1 = x[0], x[1]
    x2, y2 = y[0], y[1]
    result = ((((x2 - x1)**2) + ((y2-y1)**2))**0.5)
    print("Euclidean distance between {0} and {1} is {2}".format(x, y, result))
    return result


def draw_craters(df, lon_b, lat_b, u=None):
    # Draw the craters on the image
    lon_bounds = lon_b
    lat_bounds = lat_b
    # CAMERA CENTER:
    CAMx, CAMy = (
        (lon_bounds[0] + lon_bounds[1]) / 2,
        (lat_bounds[0] + lat_bounds[1]) / 2,
    )

    if u == None:  # Scale Factor
        u = 256  # ? DEG TO PXS
        span = (abs(lon_b[0]) - abs(lon_b[1])) * 256
        span = abs(int(span))
    # Make the img:
    img = np.zeros((span, span), dtype=int)
    if df is None:
        return img
    else:
        W, H = (
            img.shape[0],
            img.shape[1],
        )  # TODO change the function to non-square shapes
        img = np.ascontiguousarray(img, dtype=np.uint8)
        # Cycle through the dataframe:
        for i in range(df.shape[0]):
            crater = df.iloc[i]
            if crater.Diam < 100:
                # crater center:
                xc, yc = crater.Lon, crater.Lat  # This is in the absolute frame
                # f: Absolute --> f: Relative
                xc = xc - CAMx
                yc = yc - CAMy
                # f: relative --> f: OPENCV
                xc *= u  # Now is in pixel not in lon deg
                yc *= u  # Now is in pixel not in lat deg
                xc = W / 2 + xc
                yc = H / 2 - yc
                center_coordinates = (int(xc), int(yc))
                # ? 1 km = 8.4746 px in our DEM := Merge LOLA - KAGUYA
                KM_to_PX = 8.4746
                radius = int(crater.Diam / 2 * KM_to_PX)
                color = 255
                thickness = 3
                img = cv2.circle(img, center_coordinates,
                                 radius, color, thickness)
        return img


def draw_craters_on_image(df, lon_b, lat_b, img, u=None):
    # Draw the craters on the image with a given image 
    if df is None:
        return img
    else:
        lon_bounds = lon_b
        lat_bounds = lat_b
        # CAMERA CENTER:
        CAMx, CAMy = (
            (lon_bounds[0] + lon_bounds[1]) / 2,
            (lat_bounds[0] + lat_bounds[1]) / 2,
        )

        if u == None:  # Scale Factor
            u = 256  # ? DEG TO PXS
            span = (abs(lon_b[0]) - abs(lon_b[1])) * 256
            span = abs(int(span))
        # Make the img:
        W, H = (
            img.shape[0],
            img.shape[1],
        )  # TODO change the function to non-square shapes
        img = np.ascontiguousarray(img, dtype=np.uint8)
        # Cycle through the dataframe:
        for i in range(df.shape[0]):
            crater = df.iloc[i]
            if crater.Diam < 100:
                # crater center:
                xc, yc = crater.Lon, crater.Lat  # This is in the absolute frame
                # f: Absolute --> f: Relative
                xc = xc - CAMx
                yc = yc - CAMy
                # f: relative --> f: OPENCV
                xc *= u  # Now is in pixel not in lon deg
                yc *= u  # Now is in pixel not in lat deg
                xc = W / 2 + xc
                yc = H / 2 - yc
                center_coordinates = (int(xc), int(yc))
                # ? 1 km = 8.4746 px in our DEM := Merge LOLA - KAGUYA
                KM_to_PX = 8.4746
                radius = int(crater.Diam / 2 * KM_to_PX)
                color = (0, 0, 255)
                thickness = 3
                img = cv2.circle(img, center_coordinates,
                                 radius, color, thickness)
        return img


def cartesian2spherical(x, y, z):
    """ Convert cartesian coordinates to spherical coordinates.

    Args:
        x (float): cartesian x coordinate
        y (float): cartesian y coordinate
        z (float): cartesian z coordinate

    Returns:
        h (np.array): spherical h coordinate in km
        Lat (np.array): spherical h coordinate in deg
        Lon (np.array): spherical h coordinate in deg
    """
    h, Lat, Lon = cartesian_to_spherical(x, y, z)
    R_moon = 1737.4
    h = h - R_moon
    Lon = np.where(Lon.deg < 180, Lon.deg, Lon.deg - 360)
    Lat = np.where(Lat.deg < 90, Lat.deg, Lat.deg - 360)
    return np.array(h), np.array(Lat), np.array(Lon)


def spherical2cartesian(h, Lat, Lon):
    """ Convert spherical coordinates to cartesian coordinates.

    Args:
        h (float):  Altitude (km)
        Lat (float): Latitude (deg)
        Lon (float): Longitude (deg)

    Returns:
        x (np.array): cartesian x coordinate
        y (np.array): cartesian y coordinate
        z (np.array): cartesian z coordinate
    """    
    R_moon = 1737.4 # Moon radius in km
    x, y, z = spherical_to_cartesian(
        h + R_moon, np.deg2rad(Lat), np.deg2rad(Lon))
    return np.array(x), np.array(y), np.array(z)

def CatalogSearch(H, lat_bounds: np.array, lon_bounds: np.array, CAT_NAME):
    """
    This function will search the catalog and return the craters that fall within the specified bounds
    :param H: The catalog
    :param lat_bounds: The latitude bounds
    :param lon_bounds: The longitude bounds
    :param CAT_NAME: The name of the catalog
    :return: A dataframe containing the craters that fall within the specified bounds
    """



    # Choose the columns to use depending on the catalog
    if CAT_NAME == "LROC":
        LATs = np.array(H["Lat"])
        DIAMs = np.array(H["Diameter (km)"])
        LONs = np.array(H["Long"])

    elif CAT_NAME == "HEAD":
        LONs = np.array(H["Lon"])
        LATs = np.array(H["Lat"])
        DIAMs = np.array(H["Diam_km"])
    elif CAT_NAME == "ROBBINS":
        LONs = np.array(H["LON_CIRC_IMG"])
        LATs = np.array(H["LAT_CIRC_IMG"])
        DIAMs = np.array(H["DIAM_CIRC_IMG"])

    elif CAT_NAME == "COMBINED":
        LONs = np.array(H["lon"])
        LATs = np.array(H["lat"])
        DIAMs = np.array(H["diam"])

    # Convert longitudes from [-180, 180] to [0, 360]
        # -180 to 180 // formulation 1
        #   0  to 360 // formulation 2
        # Example: 190 lon //formulation 2 --> -170 lon // formulation 1
        # -10 lon == 350 lon
        # We want to pass from f1 --> f2
    LONs_f1 = np.where(LONs > 180, LONs - 360, LONs)

    cond1 = LONs_f1 < lon_bounds[1]
    cond2 = LONs_f1 > lon_bounds[0]
    cond3 = LATs > lat_bounds[0]
    cond4 = LATs < lat_bounds[1]

    # Extract filtered data
    filt = cond1 & cond2 & cond3 & cond4

    LATs = LATs[filt]
    LONs_f1 = LONs_f1[filt]
    DIAMs = DIAMs[filt]
    
    # If there are no craters in the filtered data, return None
    if LONs_f1 != []:
        craters = np.hstack(
            [np.vstack(LONs_f1), np.vstack(LATs), np.vstack(DIAMs)])
        df = pd.DataFrame(data=craters, columns=["Lon", "Lat", "Diam"])
        return df
    else:
        return None

    

def row(idx, df):
    # Return the row of the dataframe at the given index.
    return df.iloc[idx]


# Print iterations progress
def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def find_dteta(H: float) -> float:
    """Determines the change in latitude of an object on Mars given the height of the object
    above the surface of Mars

    Parameters
    ----------
    H : float
        Height of the object above the surface of Mars (km)

    Returns
    -------
    float
        Change in latitude of an object on Mars (degrees)
    """
    try:
        FOV = np.deg2rad(45)  # Field of View of camera
        d = 2 * H * np.tan(FOV)  # Distance to horizon
        R_m = 1737.1  # Radius of Mars
        dteta = d / R_m  # Change in latitude
        dteta = np.rad2deg(dteta)  # Change in latitude in degrees
        return dteta
    except:
        return None



def remove_items(list: list, item: object) -> list:
    # remove all occurrences of a given item from a list
    # using list comprehension to perform the task
    res = [i for i in list if i != item]
    return res


def remove_multiple_items(indexes: np.ndarray) -> np.ndarray:
    """The code above does the following:
        1. remove all occurrences of a given item from a list
        2. concatenate the two lists of items to be kept 

    Args:
        indexes (np.ndarray): array of indexes

    Returns:
        np.ndarray: _description_
    """    
    # remove all occurrences of a given item from a list
    # print(indexes)
    idx_a = indexes[:, 0]
    idx_b = indexes[:, 1]
    # print(idx_b)
    list_a = []
    list_b = []
    for elem_a, elem_b in zip(idx_a, idx_b):
        # if there is at least one occurrence of the same item in both lists
        if (
            np.count_nonzero(idx_a == elem_a) > 1
            or np.count_nonzero(idx_b == elem_b) > 1
        ):
            # add 0 to the list of items to be removed
            list_a.append(0)
            list_b.append(0)
        else:
            # add the item to the list of items to be kept
            list_a.append(elem_a)
            list_b.append(elem_b)
    # remove all 0 items from the list
    a = remove_items(list_a, 0)
    b = remove_items(list_b, 0)
    a, b = np.vstack(a), np.vstack(b)
    # concatenate the two lists of items to be kept
    if a.shape[0] == 0 or b.shape[0] == 0:
        v = np.array([])
    else:
        v = np.hstack([a, b])
    return v


@njit
def findAngles(a: float, b: float, c: float):
    """ Find the angles of a triangle given the lengths of the sides

    Args:
        a (float): a side of the triangle
        b (float): b side of the triangle
        c (float): c side of the triangle

    Returns:
        A: angle opposite side a
        B: angle opposite side b
        C: angle opposite side c
    """    
    # applied cosine rule
    # find the angle opposite side a
    A = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
    # find the angle opposite side b
    B = np.arccos((a**2 + c**2 - b**2) / (2 * a * c))
    # find the angle opposite side c
    C = np.arccos((b**2 + a**2 - c**2) / (2 * b * a))
    # convert into degrees
    A, B, C = np.rad2deg(A), np.rad2deg(B), np.rad2deg(C)
    return A, B, C




@njit
def compute_K_vet(triplet):
    """ This code computes the angles of a triangle, sorts them, and returns them.
        It takes in a triplet of points and returns a triplet of angles.
        The function names and variable names are important because they describe what the code does.
        The purpose of the code is to compute the angles of a triangle and return them.
        The context of the code is that it is used to compute the angles of a triangle.
        The code is relevant because it is used to compute the angles of a triangle.

    Args:
        triplet (np.array): triplet of points

    Returns:
        K_vet: vector of angles
    """    
    a, b, c = compute_sides(triplet)
    A, B, C = findAngles(a, b, c)
    K_vet = np.sort(np.array([A, B, C]))
    if K_vet is not None:
        return K_vet


@njit
def compute_sides(triplet):
    # Compute the sides of a triangle from three points
    # Compute the length of the first side
    try:
        a = np.linalg.norm(triplet[0][0:2] - triplet[1][0:2])
    except TypeError:
        return np.nan, np.nan, np.nan
    # Compute the length of the second side
    try:
        b = np.linalg.norm(triplet[1][0:2] - triplet[2][0:2])
    except TypeError:
        return np.nan, np.nan, np.nan
    # Compute the length of the third side
    try:
        c = np.linalg.norm(triplet[2][0:2] - triplet[0][0:2])
    except TypeError:
        return np.nan, np.nan, np.nan
    return a, b, c


def find_all_triplets(craters):
    """ This code finds all the triplets of points in a list of points.

    Args:
        craters (np.array): crater points
    """    
    
    def Hstack(K_v, i, j, k, x1, y1, r1, x2, y2, r2, x3, y3, r3):
        """Stacks the provided arguments into a single array.
        
        Arguments:
            K_v: a 3-element array containing the camera intrinsics
            i: the index of the first image
            j: the index of the second image
            k: the index of the third image
            x1: the x-coordinate of the first point
            y1: the y-coordinate of the first point
            r1: the radius of the first point
            x2: the x-coordinate of the second point
            y2: the y-coordinate of the second point
            r2: the radius of the second point
            x3: the x-coordinate of the third point
            y3: the y-coordinate of the third point
            r3: the radius of the third point
        
        Returns:
            A 15-element array containing the provided arguments.
        """
        A = np.zeros(15)
        A[0], A[1], A[2] = K_v[0], K_v[1], K_v[2]
        A[3], A[4], A[5] = i, j, k
        A[6], A[7], A[8] = x1, y1, r1
        A[9], A[10], A[11] = x2, y2, r2
        A[12], A[13], A[14] = x3, y3, r3
        return A

    def eu_dist(x, y):
        # Compute the Euclidean distance between two points x and y
        x1, y1 = x[0], x[1]
        x2, y2 = y[0], y[1]
        result = ((((x2 - x1)**2) + ((y2-y1)**2))**0.5)
        return result

    def concat(a, b, c):
        # initialize the array
        A = np.zeros((3, 3))
        
        # assign the values
        A[0] = a
        A[1] = b
        A[2] = c
        
        return A

    N = craters.shape[0]  # number of craters
    ender = N*N*N  # number of possible triplets
    K = np.zeros((ender, 15))  # matrix to store the values of K for each triplet
    lister = 0  # counter for the iteration
    for i in range(N):
        printProgressBar(i+1, N, printEnd='')
        for j in range(N):
            for k in range(N):
                if (i != j) & (j != k):
                    a = craters[i]
                    b = craters[j]
                    c = craters[k]
                    triplet = concat(a, b, c)
                    x1, y1, r1 = a[0], a[1], a[2]
                    x2, y2, r2 = b[0], b[1], b[2]
                    x3, y3, r3 = c[0], c[1], c[2]

                    C = np.zeros(2)  # centroid
                    C[0] = (x1+x2+x3)/3
                    C[1] = (y1+y2+y3)/3

                    P1, P2, P3 = np.zeros(2), np.zeros(2), np.zeros(2)
                    P1[0] = x1
                    P1[1] = y1
                    P2[0] = x2
                    P2[1] = y2
                    P3[0] = x3
                    P3[1] = y3

                    d1, d2, d3 = eu_dist(P1, C), eu_dist(P2, C), eu_dist(P3, C)
                    d_i, d_j, d_k = d1/r1, d2/r2, d3/r3

                    try:
                        K_v = compute_K_vet(triplet)
                        K[lister] = Hstack(
                            K_v, d_i, d_j, d_k, x1, y1, r1, x2, y2, r2, x3, y3, r3)
                    except ZeroDivisionError:
                        pass

                lister += 1
    return K[np.all(K != 0, axis=1)]


def swap_df_columns(colname_1, colname_2, df):
    # Make a copy of the first column
    tmp = deepcopy(df[colname_1])
    # Overwrite the first column with the second column
    df[colname_1] = df[colname_2]
    # Overwrite the second column with the first column (i.e. the copy)
    df[colname_2] = tmp
    return df


def load_all_images(dt):
    # LOAD ALL IMAGES:
    # Define a container for the images
    container = {}
    # for each image in the folder
    for img in glob.glob(f"DATA/ephemeris sat/inclination zero/{dt} step size/*"):
        # Extract the time of the image
        txt = img             # stringa
        t = txt.split('_')[1]  # numero
        # Add the image to the container
        container[t] = txt
    return container


def absolute2relative(crt, CAMx, CAMy, canvas=[849, 849], km2px=1/0.188):
    """ Transform the absolute coordinates of the crater into the relative ones

    Args:
        crt (np.array): lon (deg), lat (deg), r (km)
        CAMx (_type_): camera center x
        CAMy (_type_): camera center y
        canvas (list, optional): size in px of image. Defaults to [849, 849].
        km2px (float, optional): km to px conversion. Defaults to 1/0.188.

    Returns:
        list: x,y, r(pix)
    """
    # crater center:
    xc, yc, rc = crt[0], crt[1], crt[2]  # This is in the absolute frame
    # f: Absolute --> f: Relative
    xc = xc - CAMx
    yc = yc - CAMy
    # f: relative --> f: OPENCV
    xc *= deg2px  # Now is in pixel not in lon deg
    yc *= deg2px  # Now is in pixel not in lat deg

    xc = float(canvas[0]/2) + xc
    yc = float(canvas[1]/2) - yc
    rc = crt[2] * km2px
    return [xc, yc, rc]

def ecef_to_enu(x, y, z, lat0, lon0, h0):
    """
    This script provides coordinate transformations from Geodetic -> ECEF, ECEF -> ENU
    and Geodetic -> ENU (the composition of the two previous functions). Running the script
    by itself runs tests.
    credits to https://gist.github.com/sbarratt/a72bede917b482826192bf34f9ff5d0b
    """
    a = 1737400 # Mean radius
    b = 1738100 # Equatorial radius
    f = (a - b) / a
    e_sq = f * (2-f)

    lamb = math.radians(lat0)
    phi = math.radians(lon0)
    s = math.sin(lamb)
    N = a / math.sqrt(1 - e_sq * s * s)

    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - e_sq) * N) * sin_lambda

    xd = x - x0
    yd = y - y0
    zd = z - z0

    xEast = -sin_phi * xd + cos_phi * yd
    yNorth = -cos_phi * sin_lambda * xd - sin_lambda * sin_phi * yd + cos_lambda * zd
    zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd

    return xEast, yNorth, zUp



if __name__ == "__main__":
    pass
