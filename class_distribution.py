import numpy as np
import theano.tensor as tt


class GMMBeta:

    def __init__(self, mu, isigma, sigma, weight, prior_mean, prior_cov):
        self.mu = mu
        self.isigma = isigma
        self.sigma = sigma
        self.weight = weight
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov

    def pdfunnormalized(self, x, beta):
        mu = self.mu
        isigma = self.isigma
        weight = self.weight
        size = mu.shape
        if len(mu.shape) == 1:
            nmode = 1
        else:
            nmode = size[0]
        x0 = x - mu
        x1 = np.reshape(x0, (nmode, 1, -1))
        x2 = np.reshape(x0, (nmode, -1, 1))
        xsinv = np.sum(np.sum((x1 * x2) * isigma, axis=2), axis=1)
        pdf = np.sum(weight * np.exp(-0.5 * xsinv))**beta * self.multivariate_normal_pdf(x)**(1-beta)
        return pdf

    def pdfunnormalized_theano(self, x, beta):
        mu = self.mu
        isigma = self.isigma
        weight = self.weight
        size = mu.shape
        if len(mu.shape) == 1:
            nmode = 1
        else:
            nmode = size[0]
        x0 = x - mu
        x1 = tt.reshape(x0, (nmode, 1, -1))
        x2 = tt.reshape(x0, (nmode, -1, 1))
        xsinv = tt.sum(tt.sum((x1 * x2) * isigma, axis=2), axis=1)
        pdf = tt.sum(weight * tt.exp(-0.5 * xsinv))**beta * self.multivariate_normal_pdf_theano(x)**(1-beta)
        return pdf

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

    def derivativelogp(self, x, beta):
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
            dlogp_numerator = dlogp_numerator + ((isigma[i, :, :].dot((-x0))) * weight[i] * np.exp(-0.5 * xs_inv))
        dlogp = beta*dlogp_numerator / self.pdfunnormalized(x, beta=1) + (1-beta) * (
            np.linalg.inv(self.prior_cov).dot(-(x - self.prior_mean)))
        return dlogp

    def vderivativelogp(self, x, beta):

        if len(x.shape) == 1:
            dlogp = self.derivativelogp(x, beta)
        else:
            dlogp = np.zeros(np.shape(x))
            for i in range(x.shape[0]):
                dlogp[i, :] = self.derivativelogp(x[i, :], beta)
        return dlogp

    def multivariate_normal_pdf(self, x):
        mean = self.prior_mean
        cov = self.prior_cov
        d = len(mean)
        x0 = x - mean
        xsinv = np.dot(np.dot(x0, np.linalg.inv(cov)), x0)
        pdf = np.exp(-0.5 * xsinv) / np.sqrt((2 * np.pi) ** d * np.linalg.det(cov))
        return pdf

    def multivariate_normal_pdf_theano(self, x):
        mean = self.prior_mean
        cov = self.prior_cov
        d = len(mean)
        x0 = x - mean
        xsinv = tt.dot(tt.dot(x0, np.linalg.inv(cov)), x0)
        pdf = tt.exp(-0.5 * xsinv) / np.sqrt((2 * np.pi) ** d * np.linalg.det(cov))
        return pdf









