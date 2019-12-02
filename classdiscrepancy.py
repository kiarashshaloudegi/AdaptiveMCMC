import numpy as np

"""
IQ kernel:
            h = free_parameter;
            c = constant;
            beta = Beta;
            k = (c^2+(1/h)*norm(x-y)^2)^(-beta);
RBF kernel:
            s = free_param_linear
            k = exp(-0.5*norm(x-y)^2/s^2)
"""


class StainMethod:
    def __init__(self, free_param, constant, beta):
        self.free_param = free_param
        self.constant = constant
        self.Beta = beta

    def iq_kernelp(self, x, y, gradlogpx, gradlogpy):
        h = self.free_param
        c2 = self.constant ** 2
        beta = self.Beta
        d = x.size
        z = x - y
        r2 = np.linalg.norm(z) ** 2
        base = c2 + r2 / h
        base_beta = base ** (-beta)
        base_beta1 = base_beta / base
        coeffk = gradlogpx.dot(gradlogpy)
        coeffgrad = -2 * beta * base_beta1 * (1 / h)
        kterm = coeffk * base_beta
        gradandgradgradterms = coeffgrad * (
                    (gradlogpy.dot(z) - gradlogpx.dot(z)) + (
                        -1 * d + 2 * (beta + 1) * (1 / h) * r2 / base))
        kp = kterm + gradandgradgradterms
        return kp

    def iq_kernelp_matrix(self, X, grad_logp):
        l = X.shape[0]
        kp = np.zeros((l, l))
        for i in range(l):
            for j in range(l):
                if j > i:
                    temp = self.iq_kernelp(X[i, :], X[j, :], grad_logp[i, :],
                                           grad_logp[j, :])
                    kp[i, j] = temp
                    kp[j, i] = temp
                elif i == j:
                    temp = self.iq_kernelp(X[i, :], X[j, :], grad_logp[i, :],
                                           grad_logp[j, :])
                    kp[i, j] = temp
        return kp

