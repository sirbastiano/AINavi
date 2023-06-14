# External libraries
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.style as style
import pandas as pd
import astropy
import scipy
from filterpy.kalman import KalmanFilter 
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
from astropy import units as u
from poliastro.bodies import Earth, Mars, Sun, Moon
from poliastro.twobody import Orbit
from poliastro.plotting import OrbitPlotter2D
from poliastro.plotting import OrbitPlotter3D
from sklearn import linear_model, datasets
import glob

# Own Libraries
from utility.utils import *
from EKF.kf import *
from CAMDetector.detect import *
from CMA.pair import *
from CMA.icp import *

style.use('seaborn-paper')

global km2px, deg2km, px2km, deg2px
global DT, TOL1, TOL2, TOL_S1, TOL_S2


def find_solution():

    def check_sol(I,J, tol, mode='natural'):
        # Auxiliary function to check if the solution is correct using the tolerance
        # mode = 'natural' or 'inverse'
        if mode == 'natural':
            row1 = iss[I]
            tmp = S[I].iloc[J]
        elif mode == 'inverse':
            row1 = S[I].iloc[J]
            tmp = iss[I]


        left_id = np.argmin([row1.lon1, row1.lon2, row1.lon3])
        right_id = np.argmax([row1.lon1, row1.lon2, row1.lon3])

        if left_id==0:
            left = [row1.lon1, row1.lat1, row1.r1]
        elif left_id==1:
            left = [row1.lon2, row1.lat2, row1.r2]
        elif left_id==2:
            left = [row1.lon3, row1.lat3, row1.r3]    

        if right_id==0:
            right = [row1.lon1, row1.lat1, row1.r1]
        elif right_id==1:
            right = [row1.lon2, row1.lat2, row1.r2]
        elif right_id==2:
            right = [row1.lon3, row1.lat3, row1.r3] 


        x1,x2,x3 = tmp.x1, tmp.x2, tmp.x3
        y1,y2,y3 = tmp.y1, tmp.y2, tmp.y3
        r1,r2,r3 = tmp.r1, tmp.r2,  tmp.r3

        Left_id = np.argmin([x1,x2,x3])
        Right_id = np.argmax([x1,x2,x3])

        if Left_id==0:
            Left = [x1, y1, r1]
        elif Left_id==1:
            Left = [x2,y2,r2]
        elif Left_id==2:
            Left = [x3,y3,r3]    

        if Right_id==0:
            Right = [x1,y1,r1]
        elif Right_id==1:
            Right = [x2,y2,r2]
        elif Right_id==2:
            Right = [x3,y3,r3]

        a=left[2]/Left[2]
        b=right[2]/Right[2]

        if a-b < tol:
            return True
        else: return False


    def plot_sol(I,J, mode):
        if mode == 'natural':
            row1 = iss[I]
            tmp = S[I].iloc[J]
        elif mode == 'inverse':
            tmp = iss[I]
            row1 = S[I].iloc[J]

        CAMx, CAMy = ((lon_bounds[0] + lon_bounds[1]) / 2,
                      (lat_bounds[0] + lat_bounds[1]) / 2)


        crt1 = np.array([ row1.lon1, row1.lat1, row1.r1  ])
        crt2 = np.array([ row1.lon2, row1.lat2, row1.r2  ])
        crt3 = np.array([ row1.lon3, row1.lat3, row1.r3  ])
        triplet = [crt1, crt2, crt3]


        # img=cv2.imread(filename)
        img=np.zeros((850,850,3))
        deg2px = 256
        for crt in triplet:
            # crater center:
            xc, yc, rc = crt[0], crt[1], crt[2]  # This is in the absolute frame
            # f: Absolute --> f: Relative
            xc = xc - CAMx
            yc = yc - CAMy
            # f: relative --> f: OPENCV
            xc *= deg2px  # Now is in pixel not in lon deg
            yc *= deg2px  # Now is in pixel not in lat deg
            # rc *= u  # Now is in pixel not in lat deg


            xc = 850/2 + xc
            yc = 850/2 - yc
            center_coordinates = (int(xc), int(yc))
            # ? 1 km = 8.4746 px in our DEM := Merge LOLA - KAGUYA
            radius = int(crt[2] * km2px)
            color = (255, 255, 255)
            thickness = 3
            img_prova = cv2.circle(img, center_coordinates, radius, color, thickness)

        plt.figure(dpi=130)
        plt.subplot(121)
        plt.imshow(img_prova)
        plt.xlabel('CAT')
        plt.show()


        cp1 = cv2.imread(filename)
        x1,x2,x3 = tmp.x1, tmp.x2, tmp.x3
        y1,y2,y3 = tmp.y1, tmp.y2, tmp.y3
        r1,r2,r3 = tmp.r1, tmp.r2,  tmp.r3
        cr1 = np.array([x1,y1,r1]) 
        cr2 = np.array([x2,y2,r2]) 
        cr3 = np.array([x3,y3,r3])
        crts = np.vstack([cr1,cr2,cr3])
        plt.subplot(122)
        plt.xlabel('DET')
        IMG1 =  img_plus_crts(cp1, crts, color="red")
        plt.imshow(IMG1)
        plt.show()

    def find_slope(P1:np.array,P2:np.array) -> float:
        slope = (P2[1]-P1[1])/(P2[0]-P1[0])
        return slope



    def check_sol2(I,J, tol, mode):    
        if mode == 'natural':
            B = iss[I]
            A = S[I].iloc[J]
        elif mode == 'inverse':
            B = S[I].iloc[J]
            A = iss[I]

        hp = A
        x1_a, x2_a, x3_a = float(hp.x1), float(hp.x2), float(hp.x3)
        y1_a, y2_a, y3_a = float(hp.y1), float(hp.y2), float(hp.y3)
        r1_a, r2_a, r3_a = float(hp.r1), float(hp.r2), float(hp.r3)

        A1 = np.hstack([x1_a, y1_a, r1_a])
        A2 = np.hstack([x2_a, y2_a, r2_a])
        A3 = np.hstack([x3_a, y3_a, r3_a])

        A = np.vstack([A1, A2, A3])

        hp = B
        x1_b, x2_b, x3_b = float(hp.lon1), float(hp.lon2), float(hp.lon3)
        y1_b, y2_b, y3_b = float(hp.lat1), float(hp.lat2), float(hp.lat3)
        r1_b, r2_b, r3_b = float(hp.r1), float(hp.r2), float(hp.r3)

        x1_b_r, y1_b_r, r1_b_r = absolute2relative([x1_b, y1_b, r1_b], CAMx, CAMy)
        x2_b_r, y2_b_r, r2_b_r = absolute2relative([x2_b, y2_b, r2_b], CAMx, CAMy)
        x3_b_r, y3_b_r, r3_b_r = absolute2relative([x3_b, y3_b, r3_b], CAMx, CAMy)

        B1 = np.hstack([x1_b_r, y1_b_r, r1_b_r])
        B2 = np.hstack([x2_b_r, y2_b_r, r2_b_r])
        B3 = np.hstack([x3_b_r, y3_b_r, r3_b_r])

        B = np.vstack([B1, B2, B3])

        # identifiy points A:
        x1,x2,x3 = A[0][0], A[1][0], A[2][0]
        y1,y2,y3 = A[0][1], A[1][1], A[2][1]
        r1,r2,r3 = A[0][2], A[1][2], A[2][2]
        # Pick the ids:
        Left_id = np.argmin([x1,x2,x3])
        Right_id = np.argmax([x1,x2,x3])
        for id in [0,1,2]: 
            if (id != Left_id) & (id != Right_id): Center_id = id 
        # Reassign relate to ids:
        if Left_id==0:
            Left = [x1, y1, r1]
        elif Left_id==1:
            Left = [x2,y2,r2]
        elif Left_id==2:
            Left = [x3,y3,r3]    

        if Right_id==0:
            Right = [x1,y1,r1]
        elif Right_id==1:
            Right = [x2,y2,r2]
        elif Right_id==2:
            Right = [x3,y3,r3]

        if Center_id==0:
            Center = [x1,y1,r1]
        elif Center_id==1:
            Center = [x2,y2,r2]
        elif Center_id==2:
            Center = [x3,y3,r3]
        # Calculate Orientation:
        alfa1 = find_slope(Left, Center)
        alfa2 = find_slope(Center, Right)
        alfa3 = find_slope(Left, Right)
        # print('\n')
        # print(alfa1,alfa2, alfa3)
        # identifiy points B:
        x1,x2,x3 = B[0][0], B[1][0], B[2][0]
        y1,y2,y3 = B[0][1], B[1][1], B[2][1]
        r1,r2,r3 = B[0][2], B[1][2], B[2][2]
        # Pick the ids:
        Left_id = np.argmin([x1,x2,x3])
        Right_id = np.argmax([x1,x2,x3])
        for id in [0,1,2]: 
            if (id != Left_id) & (id != Right_id): Center_id = id 
        # Reassign relate to ids:
        if Left_id==0:
            Left = [x1, y1, r1]
        elif Left_id==1:
            Left = [x2,y2,r2]
        elif Left_id==2:
            Left = [x3,y3,r3]    

        if Right_id==0:
            Right = [x1,y1,r1]
        elif Right_id==1:
            Right = [x2,y2,r2]
        elif Right_id==2:
            Right = [x3,y3,r3]

        if Center_id==0:
            Center = [x1,y1,r1]
        elif Center_id==1:
            Center = [x2,y2,r2]
        elif Center_id==2:
            Center = [x3,y3,r3]
        # Calculate Orientation:
        beta1 = find_slope(Left, Center)
        beta2 = find_slope(Center, Right)
        beta3 = find_slope(Left, Right)
        # print('\n')
        # print(beta1,beta2,beta3)
        # R, t = icp(A,B)

        # sinteta = R[1,0]
        # costeta = R[0,0]
        # tanteta = sinteta/costeta
        # teta = np.arctan(tanteta)
        # teta = np.rad2deg(teta)

        # if abs(teta) < tol: return True
        # else: return False

        if (abs(alfa1-beta1) < tol) & (abs(alfa2-beta2) < tol) & (abs(alfa3-beta3) < tol): return True
        else: return False


    def filter_quartile(Xs):
        X = pd.DataFrame(Xs)
        # z = abs(stats.zscore(Z))
        # print(z)
        Q1 = X.quantile(0.48)
        Q3 = X.quantile(0.52)
        IQR = Q3 - Q1
        X = X[np.logical_not((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR)))]
        X = X.dropna()
        return np.array(X)
################################################################################################
#                                   MAIN
################################################################################################




# def find_solution(TOL_S1, TOL_S2, TOL1, TOL2):
    ZF = []
    CRATERS_CAT, CRATERS_DET = [], []
    for idx in range(61):
        # Loading All Images:
        dict = load_all_images(dt=DT)
        # Img:
        filename = dict[str(idx+1)]
        img=cv2.imread(filename)
        # Detection:
        t1 = time.time()
        craters_det = detect(img)
        # Removing minor craters:
        craters_det = craters_det[craters_det[:,2] > 15]
        t2 = time.time()
        save_craters_det = craters_det.shape[0]
        print(f'Detection Time:{t2-t1:.2f}\n')
        # Pandas DataFrame:
        df_craters_det = sort_mat(craters_det)
        # Find all triplets:
        t1 = time.time()
        triplets = find_all_triplets(craters_det)
        triplets_det= pd.DataFrame(triplets, columns=['Angle1','Angle2','Angle3','des1','des2','des3','x1','y1','r1','x2','y2','r2','x3',   'y3',  'r3'])
        triplets_det.shape
        t2 = time.time()
        print('\n')
        print(f'Total craters founded:{craters_det.shape[0]}')
        print(f'Number of total combinations:{triplets_det.shape[0]}\nComputational time: {t2-t1:.2f} s')



        # Opening Database:
        DB = pd.read_csv('DATA/H_L_combined.csv')
        # DB = pd.read_csv('DATA/lunar_crater_database_robbins_2018.csv')
        # Filtering:
        span = 3.29/2 * 1. 
        lat_bounds=[-span, span]
        get_lon = float(filename.split('_')[-1].split('jpg')[0][:-2])
        lon_bounds=[get_lon-span,get_lon+span]

        craters_cat = CatalogSearch(DB, lat_bounds, lon_bounds, CAT_NAME='COMBINED')
        if craters_cat is not None:
            km2deg = 1/deg2km
            craters_cat = craters_cat[(craters_cat.Diam < 40)&(craters_cat.Diam > 2.5)]
            craters_cat['Diam']*=0.5*km2deg # km --- > deg
            save_craters_cat = craters_cat.shape[0]

            craters_cat_m = np.array(craters_cat)

            t1 = time.time()
            triplets_cat_m = find_all_triplets(craters_cat_m)
            triplets_cat = pd.DataFrame(triplets_cat_m, columns=['Angle1','Angle2','Angle3','des1','des2','des3','lon1','lat1', 'r1','lon2',        'lat2','r2','lon3','lat3','r3'])
            triplets_cat['r1'] *= deg2km
            triplets_cat['r2'] *= deg2km
            triplets_cat['r3'] *= deg2km
            t2 = time.time()
            print(f'Total craters catalogued:{craters_cat.shape[0]+1}')
            print(f'Number of total combinations:{triplets_cat.shape[0]}\nComputational time: {t2-t1:.2f} s')
        else:
            print('No craters in cat!')


        if VERBOSE:
            #img1
            plt.figure(dpi=200, tight_layout=True)
            cp1 = deepcopy(img)
            img_det = img_plus_crts(img, craters_det)
            plt.subplot(122)
            plt.xticks([0,848/2,848],[f'{lon_bounds[0]:.2f}°',f'{(lon_bounds[1]+lon_bounds[0])/2:.2f}°',f'{lon_bounds[1]:.2f}°'])
            plt.yticks([0,848/2,848],[f'{lat_bounds[0]:.2f}°',f'{(lat_bounds[1]+lat_bounds[0])/2:.2f}°',f'{lat_bounds[1]:.2f}°'])
            plt.imshow(img_det)
            plt.xlabel('LON')
            plt.ylabel('LAT')
            plt.show()

            # FIG.2
            cp1 = deepcopy(img)
            # DB = pd.read_csv('DATA/lunar_crater_database_robbins_2018.csv')
            DB = pd.read_csv('DATA/H_L_combined.csv')
            df = CatalogSearch(DB, lat_bounds, lon_bounds, CAT_NAME='COMBINED')
            image_with_craters = draw_craters_on_image(df,  lon_bounds, lat_bounds, cp1, u=None)

            plt.subplot(121)
            plt.imshow(image_with_craters)
            plt.xticks([0,850/2,850],[f'{lon_bounds[0]:.2f}°',f'{(lon_bounds[1]+lon_bounds[0])/2:.2f}°',f'{lon_bounds[1]:.2f}°'])
            plt.yticks([0,850/2, 850],[f'{lat_bounds[0]:.2f}°',f'{(lat_bounds[1]+lat_bounds[0])/2:.2f}°',f'{lat_bounds[1]:.2f}°'])
            plt.xlabel('LON')
            plt.ylabel('LAT')
            plt.show()



        tol1 = TOL_S1

        t1 = time.time()
        QUERY1 = triplets_cat
        QUERY2 = triplets_det
        QUERY1 = dropduplicates(QUERY1)
        QUERY2 = dropduplicates(QUERY2) 

        if QUERY1.shape[0]<QUERY2.shape[0]:
            mode = 'natural'
            joins, items = inner_join(QUERY1, QUERY2, tol1)
        else:
            mode = 'inverse'
            joins, items = inner_join(QUERY2, QUERY1, tol1)
        print(f'Mode:{ mode}')
        t2 = time.time()
        print(f'Computational time: {t2-t1:.2f} s\nPossible list Combinations: {len(items)}')


        t1 = time.time()
        tol2 = TOL_S2
        S, iss = [], []
        for i in range(len(joins)):
            join = joins[i]
            des1, des2, des3 = items[i].des1, items[i].des2, items[i].des3
            s=join[ (abs(join.des1 - des1) < tol2) & (abs(join.des2 - des2) < tol2) & (abs(join.des3 - des3) < tol2)\
                  | (abs(join.des1 - des2) < tol2) & (abs(join.des2 - des3) < tol2) & (abs(join.des3 - des1) < tol2)\
                  | (abs(join.des1 - des3) < tol2) & (abs(join.des2 - des1) < tol2) & (abs(join.des3 - des2) < tol2)]

            if s.shape[0] > 0:
                S.append(s)
                iss.append(items[i])
        t2 = time.time()
        print(f'Computational time: {t2-t1:.2f} s\nPossible list Combinations: {len(S)}')



        # TEST
        CAMx, CAMy = ( (lon_bounds[0] + lon_bounds[1]) / 2, (lat_bounds[0] + lat_bounds[1]) / 2) # Location Absolute
        if mode == 'natural':
            for I in range(len(iss)):
                row1 = iss[I]
                J = 0
                for J in range(S[I].shape[0]):
                    tmp = S[I].iloc[J]

                    diff = compute_pos_diff(tmp, row1, CAMx, CAMy)
                    diff = np.array(diff) # Is in pixel
                    q = diff*px2km
                    if np.all( abs(q) < 2):
                        print(q, I,J)
                    J+=1
        else:
            for I in range(len(S)):
                row1 = iss[I]
                J = 0
                for J in range(S[I].shape[0]):
                    tmp = S[I].iloc[J]

                    diff = compute_pos_diff(row1,tmp,CAMx, CAMy)
                    diff = np.array(diff) # Is in pixel
                    q = diff*px2km
                    if np.all( abs(q) < 2):
                        print(q, I,J)
                    J+=1
        # VERIFICA:
        Is, Js = [], []
        for I in range(len(iss)):
            row1 = iss[I]
            for J in range(S[I].shape[0]):
                if check_sol(I,J, TOL1, mode): 
                    if check_sol2(I,J, TOL2, mode):
                        Is.append(I)
                        Js.append(J)
                        print(I,J)
        Is = np.array(Is)
        Js = np.array(Js)



        Ts = []
        for i, j in zip(Is,Js):
            if mode == 'natural':
                tc = iss[i]
                td = S[i].iloc[j]
            elif mode == 'inverse':
                td = iss[i]
                tc = S[i].iloc[j]

            hp = td
            x1_a, x2_a, x3_a = float(hp.x1), float(hp.x2), float(hp.x3)
            y1_a, y2_a, y3_a = float(hp.y1), float(hp.y2), float(hp.y3)
            r1_a, r3_a, r3_a = float(hp.r1), float(hp.r2), float(hp.r3)

            A1 = np.hstack([x1_a, y1_a])
            A2 = np.hstack([x2_a, y2_a])
            A3 = np.hstack([x3_a, y3_a])

            A = np.vstack([A1, A2, A3])

            hp = tc
            x1_b, x2_b, x3_b = float(hp.lon1), float(hp.lon2), float(hp.lon3)
            y1_b, y2_b, y3_b = float(hp.lat1), float(hp.lat2), float(hp.lat3)
            r1_b, r2_b, r3_b = float(hp.r1), float(hp.r2), float(hp.r3)

            x1_b_r, y1_b_r, r1_b_r = absolute2relative([x1_b, y1_b, r1_b], CAMx, CAMy)
            x2_b_r, y2_b_r, r2_b_r = absolute2relative([x2_b, y2_b, r2_b], CAMx, CAMy)
            x3_b_r, y3_b_r, r3_b_r = absolute2relative([x3_b, y3_b, r3_b], CAMx, CAMy)

            B1 = np.hstack([x1_b_r, y1_b_r])
            B2 = np.hstack([x2_b_r, y2_b_r])
            B3 = np.hstack([x3_b_r, y3_b_r])

            B = np.vstack([B1, B2, B3])

            R, t = icp(A,B)
            Ts.append(t)
        print(len(Ts))



        # Reallocate points
        Xs, Ys = [], []
        for t in Ts:
            Xs.append(t[0])
            Ys.append(t[1])

        # Calculate Error on position
        if len(Ts)>3:
            Xs = filter_quartile(Xs)
            Ys = filter_quartile(Ys)
            Xs = np.mean(Xs)
            Ys = np.mean(Ys)
            Z = np.hstack([Xs,Ys])
            ZF.append(abs(Z*px2km*1000))
            CRATERS_CAT.append(save_craters_cat)
            CRATERS_DET.append(save_craters_det)

        elif len(Ts)>0:
            Xs = np.mean(Xs)
            Ys = np.mean(Ys)
            Z = np.hstack([Xs,Ys])
            ZF.append(abs(Z*px2km*1000))
            CRATERS_CAT.append(save_craters_cat)
            CRATERS_DET.append(save_craters_det)
        else:
            print('No combination Found, impossible to estimate position...')
            ZF.append(np.array([-1,-1]))
            CRATERS_CAT.append(-1)
            CRATERS_DET.append(-1)

    CRATERS_CAT= np.array(CRATERS_CAT)
    CRATERS_DET= np.array(CRATERS_DET)

    CRATERS_CAT= CRATERS_CAT[CRATERS_CAT>0]
    CRATERS_DET= CRATERS_DET[CRATERS_DET>0]

    X,Y = [], []
    for i in ZF:
        if np.all( i > 0):
            X.append(i[0])
            Y.append(i[1])
    
    X = np.array(X)
    Y = np.array(Y)

    X_mean = np.mean(filter_quartile(X))
    Y_mean = np.mean(filter_quartile(Y))
    if len(X) > 0:
        plt.figure(dpi=300)
        plt.scatter(range(len(X)),X)
        plt.scatter(range(len(Y)),Y)
        plt.legend(['Error-X','Error-Y'])
        plt.title(f'TOL_S1:{TOL_S1},            TOL_S2:{TOL_S2},            TOL1:{TOL1},           TOL2:{TOL2}')
        plt.ylabel('m')
        plt.xlabel('Estimation completed')
        plt.ylim([0, 2000])
        plt.savefig(f'A_Mx_{X_mean:.2f}_My_{Y_mean:.2f}_Comp_{X.shape[0]:.2f}.jpg')
        plt.show()

        plt.figure(dpi=300)
        plt.scatter(range(len(CRATERS_CAT)), CRATERS_CAT)
        plt.scatter(range(len(CRATERS_DET)), CRATERS_DET)
        plt.legend(['Craters Cat','Craters Det'])
        plt.title(f'TOL_S1:{TOL_S1},            TOL_S2:{TOL_S2},            TOL1:{TOL1},           TOL2:{TOL2}')
        plt.savefig(f'B_Mx_{X_mean:.2f}_My_{Y_mean:.2f}_Comp_{X.shape[0]:.2f}.jpg')
        plt.show()


if __name__ == '__main__':

       VERBOSE=False
       DT = 10

       for TOL1 in [0.08,0.1,0.12,0.14]:
              for TOL2 in [0.8,0.10,0.12,0.14,0.16]:
                     for TOL_S1 in [3,4,5,6,7]:
                            for TOL_S2 in [0.8,1,1.2,1.4]:
                                   find_solution()
                                   plt.close('all')