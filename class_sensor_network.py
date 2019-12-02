import theano.tensor as tt
import theano
import numpy as np


class SNL:

    def __init__(self, num_sensor, area_length, anchor_node, distance_measured, observ):
        self.length = area_length
        self.n = num_sensor
        self.d = distance_measured
        self.rsigma = 0.3 * self.length
        self.sigma = 0.02 * self.length
        self.anchor_node = anchor_node
        self.observ = observ
    #
    # def log_pdf(self, x_short):
    #     tiny = 1e-10
    #     x_short = np.reshape(x_short, (self.n, -1))
    #     x = np.concatenate((self.anchor_node, x_short), axis=0)
    #     xg1, xg2 = np.meshgrid(x[:, 0], x[:, 0])
    #     yg1, yg2 = np.meshgrid(x[:, 1], x[:, 1])
    #
    #     xg = np.square(xg1 - xg2)
    #     yg = np.square(yg1 - yg2)
    #
    #     x_norm = np.sqrt(xg + yg)
    #
    #     not_observ = 1 - self.observ
    #     logpdf = (
    #             -np.sum(self.observ*(xg + yg))/(2*self.rsigma**2) -
    #             np.sum(self.observ*np.square((self.d - x_norm))
    #                    )/(2*self.sigma**2) + np.sum(not_observ * np.log(1 + tiny - np.exp((xg + yg)/(-2*self.rsigma**2))))
    #               )
    #     return logpdf

    def log_pdf_beta(self, x_short, beta=1):
        tiny = 1e-10
        x_short = np.reshape(x_short, (self.n, -1))
        x = np.concatenate((self.anchor_node, x_short), axis=0)
        xg1, xg2 = np.meshgrid(x[:, 0], x[:, 0])
        yg1, yg2 = np.meshgrid(x[:, 1], x[:, 1])

        xg = np.square(xg1 - xg2)
        yg = np.square(yg1 - yg2)

        x_norm = np.sqrt(xg + yg)

        not_observ = 1 - self.observ
        logpdf1 = (
                -np.sum(self.observ*np.triu((xg + yg), k=1))/(2*self.rsigma**2) -
                np.sum(self.observ*np.square(np.triu((self.d - x_norm), k=1)))/(2*self.sigma**2)
        )
        logpdf2 = np.sum(not_observ*np.triu(np.log(1 + tiny - np.exp((xg + yg)/(-2*self.rsigma**2))), k=1))
        return (logpdf1 + logpdf2) * beta

    # def gradlog_pdf(self, x_short):
    #     x_short = np.reshape(x_short, (self.n, -1))
    #     x = np.concatenate((self.anchor_node, x_short), axis=0)
    #     x = x.reshape(-1)
    #     glogpdf = np.zeros(x.shape)
    #     for i in range(int(x.shape[0] / 2)):
    #         temp = np.zeros(2)
    #         for j in range(int(x.shape[0] / 2)):
    #             if i != j:
    #                 temp += (-(x[i*2:(i+1)*2] - x[j*2:(j+1)*2]) / (self.rsigma**2)
    #                          + (x[i*2:(i+1)*2] - x[j*2:(j+1)*2]) / np.linalg.norm((x[i*2:(i+1)*2] - x[j*2:(j+1)*2]))
    #                          * (self.d[i, j] - np.linalg.norm((x[i*2:(i+1)*2] - x[j*2:(j+1)*2])))/(self.sigma**2)
    #                          )
    #         glogpdf[i*2:(i+1)*2] = temp
    #     return glogpdf[6:]

    def gradlog_pdf_beta(self, x_short, beta=1):
        x_short = np.reshape(x_short, (self.n, -1))
        x = np.concatenate((self.anchor_node, x_short), axis=0)
        x = x.reshape(-1)
        glogpdf = np.zeros(x.shape)
        for i in range(int(x.shape[0] / 2)):
            temp = np.zeros(2)
            for j in range(int(x.shape[0] / 2)):
                if i != j:
                    exp = np.exp(np.linalg.norm((x[i*2:(i+1)*2] - x[j*2:(j+1)*2]))**2/ (-2*self.rsigma**2))
                    temp += (self.observ[i, j] * (
                            -(x[i*2:(i+1)*2] - x[j*2:(j+1)*2]) / (self.rsigma**2)
                            + (x[i*2:(i+1)*2] - x[j*2:(j+1)*2]) / np.linalg.norm((x[i*2:(i+1)*2] - x[j*2:(j+1)*2]))
                            * (self.d[i, j] - np.linalg.norm((x[i*2:(i+1)*2] - x[j*2:(j+1)*2])))/(self.sigma**2)
                             )
                            + (1-self.observ[i, j]) * ((x[i*2:(i+1)*2] - x[j*2:(j+1)*2])*exp/self.rsigma**2)/(1-exp)
                             )
            glogpdf[i*2:(i+1)*2] = temp
        return glogpdf[6:] * beta

    def log_pdf_theano(self, x_short):
        tiny = 1e-15
        x_long = tt.reshape(x_short, (self.n, -1))
        x = tt.concatenate((self.anchor_node, x_long), axis=0)
        xg2 = tt.tile(tt.reshape(x[:, 0], (-1, 1)), self.n+3)
        xg1 = tt.transpose(xg2)
        yg2 = tt.tile(tt.reshape(x[:, 1], (-1, 1)), self.n+3)
        yg1 = tt.transpose(yg2)

        xg = (xg1 - xg2)**2
        yg = (yg1 - yg2)**2

        x_norm = tt.sqrt(tt.maximum(xg + yg, 0.0000000001))

        not_observ = 1 - self.observ
        logpdf = (
                -tt.sum(self.observ*tt.triu((xg + yg), k=1))/(2*self.rsigma**2) -
                tt.sum(self.observ*tt.square(tt.triu((self.d - x_norm), k=1))
                       )/(2*self.sigma**2)
                + tt.sum(not_observ*tt.triu(tt.log(1 + tiny - tt.exp((xg + yg)/(-2*self.rsigma**2))), k=1))
        )
        return logpdf

    def pdfunnormalized(self, x):
        return np.exp(self.log_pdf_beta(x))

    def derivativelogp(self, x_short):
        theano.config.compute_test_value = 'ignore'
        y = tt.vector('y')
        logpdf_tt = theano.function([y], tt.grad(self.log_pdf_theano(y), y))
        return logpdf_tt(x_short)

    def vderivativelogp(self, x):
        y = tt.vector('y')
        logpdf_tt = theano.function([y], tt.grad(self.log_pdf_theano(y), y))
        if len(x.shape) == 1:
            dlogp = logpdf_tt(x)
        else:
            dlogp = np.zeros(np.shape(x))
            for i in range(x.shape[0]):
                dlogp[i, :] = logpdf_tt(x[i, :])
        return dlogp



