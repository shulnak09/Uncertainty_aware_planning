import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from shapely.geometry import Polygon, Point
from shapely.affinity import translate, rotate
from math import atan2, degrees, sqrt
import torch
# from shapely.figures import SIZE, GREEN, GRAY, set_limits


def minkowski_sum(PA, PB):
    """Computes the Minkowski sum of two general polygons.

    Args:
        PA (ndarray or Polygon): An n x 2 array of x,y coordinates traversing a polygon or a Shapely Polygon object.
        PB (ndarray or Polygon): An m x 2 array of x,y coordinates traversing a polygon or a Shapely Polygon object.

    Returns:
        Polygon: A Shapely Polygon object that represents the resulting Minkowski sum.
    """
    # Convert input to Shapely Polygon objects
    if isinstance(PA, np.ndarray):
        PA = Polygon(PA)
    if isinstance(PB, np.ndarray):
        PB = Polygon(PB)

    # Triangulate each polygon
    Atri = Delaunay(PA.exterior.coords)
    Btri = Delaunay(PB.exterior.coords)

    # Loop over each pair of triangles in the triangulations
    ind12 = np.array([[0,0], [0,1], [1,0], [1,1], [2,0], [2,1], [0,2], [1,2], [2,2]])
    for iA in range(Atri.nsimplex):
        for iB in range(Btri.nsimplex):
            # Form the Minkowski sum of the two triangles iA and iB from each
            # triangulated polygon
            tAxy = Atri.points[Atri.simplices[iA], :]
            tBxy = Btri.points[Btri.simplices[iB], :]
            Hxy = tAxy[ind12[:, 0], :] + tBxy[ind12[:, 1], :]

            # Form the convex hull to resolve the resulting hexagon
            hull = ConvexHull(Hxy)
            nhull = len(hull.vertices)
            if nhull > 3:
                # Remove collinear points to get rid of a warning from Shapely
                tolxy = np.linalg.norm(np.finfo(float).eps * 100 * (np.max(Hxy, axis=0) - np.min(Hxy, axis=0)))
                D = lambda xy1, xy2: np.sqrt(np.sum((xy1 - xy2) ** 2, axis=1))
                collflag = -D(Hxy[np.roll(hull.vertices, 1), :], Hxy[np.roll(hull.vertices, -1), :]) \
                           + D(Hxy[hull.vertices, :], Hxy[np.roll(hull.vertices, 1), :]) \
                           + D(Hxy[hull.vertices, :], Hxy[np.roll(hull.vertices, -1), :])
                hull.vertices = hull.vertices[collflag > tolxy]

            HexShape = Polygon(Hxy[hull.vertices, :])

            # Union the resulting shape with the previous shapes
            if iA == 0 and iB == 0:
                Psum = HexShape
            else:
                Psum = Psum.union(HexShape)

    return Psum




def create_ellipse(mean, cov, n_points=100):
    
    # Compute the eigenvectors and eigenvalues of the covariance matrix
    eigvals, eigvecs = np.linalg.eig(cov)
    
    # Compute the angle between the x-axis and the first eigenvector
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    
    # Define the ellipse
    ellipse = (mean, np.sqrt(eigvals), angle)
    
    # Create an array of angles
    angles = np.linspace(0, 360, n_points)[:, np.newaxis]
    
    # Create a circle with radius 1 centered at the origin
    circle = np.column_stack([np.cos(np.radians(angles)), np.sin(np.radians(angles))])
    
    # Scale and rotate the circle based on the covariance matrix and mean
    transform = np.column_stack([eigvecs @ np.diag(eigvals) @ circle.T, np.array(mean).reshape(2, 1)])
    
    # Create a shapely Polygon from the transformed circle points
    poly = Polygon(transform.T)
    return poly

# Compute the ADE and FDE:
# Evaluate the prediction metrics : ADE, FDE
# ADE : Average Displacement Error , FDE : Final Displacement Error


def average_displacement_error(forward_pred, y_pred, y_true):

  
    final_ADE = torch.zeros(y_pred.shape[0])
    for N in range(y_pred.shape[0]):
        ADE = 0.0 # Initialize ADE for each trajectory as 0
        for n in range(forward_pred):
              ADE += torch.mean(torch.sqrt((y_pred[N,:,n,0] - y_true[:,:,n,0])**2 + (y_pred[N,:,n,1] - y_true[:,:,n,1])**2))
        ADE = ADE/forward_pred 
        final_ADE[N] = ADE
#     print(final_ADE)
  
#     mean, std = torch.mean(final_ADE), torch.std(final_ADE)
    return final_ADE


def final_displacement_error(forward_pred, y_pred, y_true):
    
    final_FDE = torch.zeros(y_pred.shape[0])
    for N in range(y_pred.shape[0]):
        FDE = torch.mean(torch.sqrt((y_pred[N,:,-1,0] - y_true[:,:,-1,0])**2 + (y_pred[N,:,-1,1] - y_true[:,:,-1,1])**2))
        final_FDE[N] = FDE

#     print(final_FDE)
#     mean, std = torch.mean(final_FDE), torch.std(final_FDE)
    return final_FDE
    
    
# Performance Prediction Interval:
# PICP : Prediction interval coverage probability ( 1 - alpha)
# MPIW : Mean Predictive Interval Width

# Coverage plot for prediction interval:

def checkpoint(h, k, theta, x, y, a, b):
 
    # checking the equation of
    # ellipse with the given point
    p = (((x - h) * np.cos(theta) + (y-k) * np.sin(theta))**2 / a ** 2) + \
         (((y - k) * np.sin(theta) - (x - h) * np.cos(theta)) ** 2 / b**2)
 
    return p


def coverage_probability(y_pred, y_true, y_pred_var):
    
    num_fea = y_pred.shape[3]
    sigmas = np.exp(y_pred_var)
    mu_ens = np.mean(y_pred, axis=0)
    # sigma_ens = torch.sqrt((torch.sum(torch.square(mu_preds) + torch.square(sigma_preds),axis=0))/sigma_preds.shape[0] - torch.square(mu_ens))
#     var_ens = np.mean((sigmas + mus ** 2 ), axis = 0) - mu_ens**2
    var_aleatoric = np.mean(sigmas[:,:,:,:2], axis = 0)
    var_epistemic = np.mean(y_pred[:,:,:,:2]**2, axis = 0) - mu_ens[:,:,:2]**2
    var_ens = var_aleatoric  + var_epistemic
#     - mu_ens[:,:,:4]**2
#     var_epistemic = np.var(mus, axis = 0)
    var_state_unc = (np.mean((y_pred[:,:,:,2:4]), axis=0)) 

    cnt = []
    for id in range(y_pred.shape[1]):
        flag = 0
        nugget = 1e-4
    
        for pred in range(y_pred.shape[2]):
            
            mean = np.squeeze(mu_ens[id, pred, :])
    #         print(mean[0],mean[1])

            # Total Variance:
    #         var_ens = np.squeeze(var_aleatoric[id_no, pred,:]).reshape(2,2) + np.diag(np.squeeze(var_epistemic[id_no, pred,:]))
            # Total Variance:
    #         cov_epistemic = np.squeeze(var_epistemic[id_no, pred,:]))
            cov_pred = np.squeeze(np.squeeze(np.diag(var_ens[id,pred,:])))
#             print('cov_total', cov_pred)

            # Total Variance:
            cov_state = np.squeeze(var_state_unc[id, pred,:2])
            cov_state = np.diag(np.squeeze(cov_state))
#             print('cov_state',cov_state)

            lambda_tot, v_tot = np.linalg.eig(cov_pred)
            lambda_tot = np.sqrt(lambda_tot)
    #         print(lambda_tot)

            lambda_ale, v_ale = np.linalg.eig(cov_state)
            lambda_ale = np.sqrt(lambda_ale)
            
            h = mean[0]
            k = mean[1]
            theta = np.rad2deg(np.arccos((v_tot[0,0])))
            a = lambda_tot[0]  + 2* lambda_ale[0]
            b = lambda_tot[1]  + 2* lambda_ale[1]
            x = y_true[id,pred,0]
            y = y_true[id,pred,1]
            
            p = checkpoint(h, k, theta, x, y, a, b)
            
            
#             if (mean[id, n, axis] - 1*(sigma_ens[id, n, axis])) < (y_true[id,n,axis]) \
#                     < (mean[id,n,axis] + 1*(sigma_ens[id,n,axis])):
            if p <= 1:
                flag += 1
            else:
                pass
  
        cnt.append(flag/12.0)
    cnt = np.array(cnt)
    return cnt



# Prediction Interval Width:

def prediction_interval_width(y_pred, y_true,  y_pred_var):
    
    
    num_fea = y_pred.shape[3]
    sigmas = np.exp(y_pred_var)
    mu_ens = np.mean(y_pred, axis=0)
    # sigma_ens = torch.sqrt((torch.sum(torch.square(mu_preds) + torch.square(sigma_preds),axis=0))/sigma_preds.shape[0] - torch.square(mu_ens))
#     var_ens = np.mean((sigmas + mus ** 2 ), axis = 0) - mu_ens**2
    var_aleatoric = np.mean(sigmas[:,:,:,:2], axis = 0)
    var_epistemic = np.mean(y_pred[:,:,:,:2]**2, axis = 0) - mu_ens[:,:,:2]**2
    var_ens = var_aleatoric  + var_epistemic
#     - mu_ens[:,:,:4]**2
#     var_epistemic = np.var(mus, axis = 0)
    var_state_unc = (np.mean((y_pred[:,:,:,2:4]), axis=0)) 

    width = []
    for id in range(y_pred.shape[1]):
        flag = 0
        nugget = 1e-4
    
        for pred in range(y_pred.shape[2]):
            
            mean = np.squeeze(mu_ens[id, pred, :])
    #         print(mean[0],mean[1])

            # Total Variance:
    #         var_ens = np.squeeze(var_aleatoric[id_no, pred,:]).reshape(2,2) + np.diag(np.squeeze(var_epistemic[id_no, pred,:]))
            # Total Variance:
    #         cov_epistemic = np.squeeze(var_epistemic[id_no, pred,:]))
            cov_pred = np.squeeze(np.squeeze(np.diag(var_ens[id,pred,:])))
#             print('cov_total', cov_pred)

            # Total Variance:
            cov_state = np.squeeze(var_state_unc[id, pred,:2])
            cov_state = np.diag(np.squeeze(cov_state))
#             print('cov_state',cov_state)

            lambda_tot, v_tot = np.linalg.eig(cov_pred)
            lambda_tot = np.sqrt(lambda_tot)
    #         print(lambda_tot)

            lambda_ale, v_ale = np.linalg.eig(cov_state)
            lambda_ale = np.sqrt(lambda_ale)
            
            h = mean[0]
            k = mean[1]
            a = 2*(lambda_tot[0] + lambda_ale[0] )
            b = 2*(lambda_tot[1] + lambda_ale[1])
        width.append([a,b])
    width = np.array(width)
    return width
        
def reshape_data(X):
    X = X.reshape(X.shape[0] * X.shape[1], X.shape[2], X.shape[3])
    return X

def split_data(X, look_back =8, forward_pred=12):
    X_train = X[:,:look_back,:]
    y_train = X[:,look_back:,:]
    return X_train, y_train   

from scipy.stats import multivariate_normal

'''

def plot_multivariate_normal(test_x, test_y, mean, cov):
    
    plt.style.use('seaborn-white')
    plt.rcParams['figure.figsize']= 6,4
    fig = plt.figure()

    # Initializing the random seed
    # random_seed=1000

    # List containing the variance
    # covariance values
    # cov_val = [-0.8, 0, 0.8]

    # For a specific ID:
    id_no = 120

    for pred in range(forward_pred):


        # Setting mean of the distributino
        # to be at (0,0)
        mean = mu_ens[id_no, pred, :2].detach().cpu().numpy()

        # Storing density function values for
        # further analysis
        pdf_list = []




        # Initializing the covariance matrix
        cov = torch.diag(sigma_ens[id_no, pred, :2]).detach().cpu().numpy()

    #     print(mean, cov)
        # Generating a Gaussian bivariate distribution
        # with given mean and covariance matrix
        distr = multivariate_normal(mean = mean, cov = cov, 
                     seed = np.random.seed(1))

        # Generating a meshgrid complacent with
        # the 3-sigma boundary
        mean_1, mean_2 = mean[0], mean[1]
        sigma_1, sigma_2 = cov[0,0], cov[1,1]

        x = np.linspace(mean_1-1*sigma_1, mean_1+1*sigma_1, num=100)
        y = np.linspace(mean_2-1*sigma_2, mean_2+1*sigma_2, num=100)
        X, Y = np.meshgrid(x,y)

        # Generating the density function
        # for each point in the meshgrid
        pdf = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pdf[i,j] = distr.pdf([X[i,j], Y[i,j]])

        # Plotting the density function values
        # key = 131
        # ax = fig.add_subplot(key, projection = '3d')
        # ax.plot_surface(X, Y, pdf, cmap = 'viridis')
        # plt.xlabel("x1")
        # plt.ylabel("x2")
        # plt.title(f'Covariance between x1 and x2 = {val}')
        pdf_list.append(pdf)
        # ax.axes.zaxis.set_ticks([])

        # plt.tight_layout()
        # plt.show()


      # Plotting contour plots
        for idx, val in enumerate(pdf_list):
            cntr = plt.contourf(X, Y, val, cmap='viridis')

       # cbar = fig.colorbar(cntr)
       # plt.colorbar()

        proxy = [plt.Rectangle((0,0),1,0,fc = pc.get_facecolor()[0]) 
            for pc in cntr.collections]
        legend1 = plt.legend(proxy,["$\mu$, $\Sigma_{x}$, $\Sigma_{y}$"], loc = "lower right", fontsize =12 )

    #     yhat_do_mean = yhat_do.mean(axis=0)[:,:,0:2]
    
    test_x = test_x.detach().cpu().numpy()
    test_y = test_y.detach().cpu().numpy()
    
    plt.scatter(test_x[id_no,:,0], test_x[id_no,:,1], color='g', label="input trajectory", marker='o')
    plt.scatter(test_y[id_no,:,0], test_y[id_no,:,1], color='b', label="Ground Truth", alpha=0.9, marker='^')
    # plt.plot(yhat_do_mean[id_no,:,0], yhat_do_mean[id_no,:,1], color='r', label="1D-CNN", marker='s', alpha = 0.5)
    plt.xlabel("x1")
    plt.ylabel("x2")
#     # plt.title(f'Covariance between x1 and x2 ')
#     plt.ylim([3,7])
    # plt.xlim([-2,10])
    plt.gca().set_aspect('equal', adjustable='box')

    plt.tight_layout()

    plt.xlabel('x', fontsize=16)
    plt.ylabel('y',fontsize=16)
    plt.yticks(np.arange(3,7.1,2))
    # plt.xticks(np.arange(0,10.1,2))
    # legend2 = plt.legend(loc=4, fontsize = 12)
    plt.grid("on", alpha=0.5)
    plt.tick_params(axis='both', which='major', labelsize=14)
    filename = "CNN-LSTM_Bivariate_Normal_Distribution_uncertainty_states_FP_4.8_secs_ID_58_do_0.4"
    filename1 = filename + ".png"
    filename2 = filename + ".pdf"
    plt.legend()
    plt.title("")
    plt.savefig(filename1, dpi=300)
    plt.savefig(filename2, format='pdf')
    plt.show()

plot_multivariate_normal(X_test_KF, y_test_KF, mu_ens, sigma_ens)


'''