def gaussian_nll( ytrue, ypreds):

    """Keras implmementation of multivariate Gaussian negative loglikelihood loss function. 
    This implementation implies diagonal covariance matrix.

    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, n_dims]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, n_dims*2]
        predicted mu and logsigma values (e.g. by your neural network)
        
    Returns
    -------
    neg_log_likelihood: float
        negative loglikelihood averaged over samples
        
    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=gaussian_nll, optimizer='Adam') 

    """

    n_dims = int(int(ypreds.shape[2]/2))
    mu = ypreds[:, :, 0:n_dims]
    logsigma = ypreds[:, :, n_dims:]


    mse = -0.5*K.sum(K.square((ytrue-mu)/K.exp(2*logsigma)))
    sigma_trace = -K.sum(logsigma)
    log2pi = -0.5*n_dims*np.log(2*np.pi)

    log_likelihood = mse + sigma_trace + log2pi

    return K.mean(-log_likelihood)
  

def gaussian_nll_loss(mean_values, var_values, y):
  loss = 0.5*tf.reduce_mean(tf.math.log(var_values)) +\
    0.5*tf.reduce_mean(tf.divide(tf.math.square(tf.math.subtract(y,mean_values)),var_values)) + \
    0.5*tf.math.log(2*np.pi)
  return loss 


def custom_loss(sigma):
    def gaussian_loss(y_true, y_pred):
        return tf.reduce_mean(tf.math.log(sigma) + 0.5*tf.divide(tf.math.square(y_true - y_pred), sigma)) + 1e-6
    return gaussian_loss