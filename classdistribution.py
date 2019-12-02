import numpy as np


class GMM:

    def __init__(self, mu, isigma, sigma, weight):
        self.mu = mu
        self.isigma = isigma
        self.sigma = sigma
        self.weight = weight

    def pdfunnormalized(self, x):
        # x = x.reshape(-1)
        mu = self.mu
        isigma = self.isigma
        weight = self.weight
        size = mu.shape
        if len(mu.shape) == 1:
            nmode = 1
        else:
            nmode = size[0]
        # d = size[1]
        pdf = 0
        for i in range(nmode):
            x0 = (x - mu[i, :])
            s0 = isigma[i, :, :]
            xsinv = np.sum(np.dot(x0, isigma[i]) * x0, axis=1)  # x_square_inverse
            pdf += weight[i] * np.exp(-0.5 * xsinv)
        # pdf += 10**(-100)
        return pdf

    def pdfunnormalized_theano(self, x):
        import theano.tensor as tt
        # x = x.reshape(-1)
        tiny = 1e-200
        mu = self.mu
        isigma = self.isigma
        weight = self.weight
        if len(mu.shape) == 1:
            nmode = 1
        else:
            nmode = mu.shape[0]
        # nmode, d = mu.shape
        y = 0
        for i in range(nmode):
            # x0 = (x - mu[i, :])
            # xsinv = tt.dot(tt.dot(x0, isigma[i]), x0)  # x_square_inverse
            # pdf = pdf + weight[i] * tt.exp(-0.5 * xsinv)
            x0 = x - mu[i, :]
            xsinv = tt.dot(tt.dot(x0, isigma[i]), x0)
            y = y + weight[i] * tt.exp(-0.5 * xsinv)
        return y

    def expectation(self):
        mu = self.mu
        weight = self.weight
        sigma = self.sigma
        isigma = self.isigma
        size = mu.shape
        nmode = size[0]
        d = size[1]
        z = 0  # nomralization constant
        expec = np.zeros((1, d))  # expectation: non-normalized
        for i in range(nmode):
            z = z + weight[i] * np.sqrt(
                (2 * np.pi) ** d * np.linalg.det(sigma[i, :, :]))
            expec = expec + weight[i] * np.sqrt(
                (2 * np.pi) ** d * np.linalg.det(sigma[i, :, :])) * mu[i, :]
        expec = expec / z
        return expec, z

    def derivativelogp(self, x):
        mu = self.mu
        weight = self.weight
        isigma = self.isigma
        size = mu.shape
        nmode = size[0]
        d = size[1]
        dlogp_numerator = np.zeros((1, d))
        for i in range(nmode):
            x0 = x - mu[i, :]
            s0 = isigma[i, :, :]
            xs_inv = np.linalg.multi_dot([x0, s0, x0])  # x_square_inverse
            dlogp_numerator = dlogp_numerator + ((isigma[i, :, :].dot((-x0))) *
                                                 weight[i] * np.exp(-0.5 *
                                                                    xs_inv))
        dlogp = dlogp_numerator / self.pdfunnormalized(x)
        return dlogp

    def vderivativelogp(self, x):

        if len(x.shape) == 1:
            dlogp = self.derivativelogp(x)
        else:
            dlogp = np.zeros(np.shape(x))
            for i in range(x.shape[0]):
                dlogp[i, :] = self.derivativelogp(x[i, :])
        return dlogp


def multivariate_normal_pdf(x, mean, cov):
    d = len(mean)
    x0 = x - mean
    xsinv = tt.dot(tt.dot(x0, np.linalg.inv(cov)), x0)
    pdf = tt.exp(-0.5 * xsinv) / tt.sqrt((2 * np.pi) ** d * np.linalg.det(cov))
    return pdf











