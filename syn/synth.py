import numpy as np
import matplotlib.pyplot as plt


class Adam:

    def __init__(self, x0, amsgrad=False, aamsgrad=False, alpha=0.01, beta1=0.9, beta2=0.999,
                 F=None, epsilon=1e-8, decay=False):
        self.x = x0

        self.amsgrad = amsgrad
        self.aamsgrad = aamsgrad
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.F = F
        self.epsilon = epsilon
        self.decay = decay

        self.m = 0
        self.v = 0
        self.t = 0
        self.pre_x = 0
        self.vmax = 0

    def step(self, grad, pre_grad):
        self.t = self.t + 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2

        m = self.m / (1 - self.beta1 ** self.t)
        v = self.v / (1 - self.beta2 ** self.t)

        if self.amsgrad:
            self.vmax = max(self.vmax, v)
            m = self.m
            v = self.vmax

        lr = self.alpha / np.sqrt(self.t) if self.decay else self.alpha

        x_updated = self.x - lr * m / (np.sqrt(v) + self.epsilon)

        if self.aamsgrad:
            if grad * pre_grad < 0:
                self.m = grad
                self.v = grad * grad
                x_updated = self.x - lr * grad
            else:
                x_updated = self.x - lr * m / (np.sqrt(v) + self.epsilon) + self.beta1 * (self.x - self.pre_x)

        self.pre_x = self.x
        self.x = x_updated if self.F is None else np.clip(x_updated, self.F[0], self.F[1])
        return self.x


def experiment(name="", iterations=int(1e8), F=(-1, 1), max_grad=1010, min_grad=-10, stochastic_params=(0.1, True)):
    adam_s = Adam(1, F=F, alpha=stochastic_params[0], decay=stochastic_params[1])
    amsg_s = Adam(1, amsgrad=True, F=F, alpha=stochastic_params[0], decay=stochastic_params[1])
    aamsg_s = Adam(1, amsgrad=True, aamsgrad=True, F=F, alpha=stochastic_params[0], decay=stochastic_params[1])

    ts = [1]
    xs_adam_s, xs_amsg_s, xs_aamsg_s = [1], [1], [1]

    pre_grad = -10
    for t in range(4, iterations + 3):
        if name == "stochastic":
            grad_s = (max_grad if np.random.random() < 0.01 else min_grad)
        else:
            grad_s = (max_grad if t % 101 == 1 else min_grad)

        x_adam_s = adam_s.step(grad_s, pre_grad)
        x_amsg_s = amsg_s.step(grad_s, pre_grad)
        x_aamsg_s = aamsg_s.step(grad_s, pre_grad)

        xs_adam_s.append(x_adam_s)
        xs_amsg_s.append(x_amsg_s)
        xs_aamsg_s.append(x_aamsg_s)
        ts.append(t - 2)

        pre_grad = grad_s
    if name == "stochastic":
        plt.figure(1)
    else:
        plt.figure(2)
    plt.xlabel('iterations')
    plt.ylabel('x')
    plt.grid()
    plt.plot(ts, xs_adam_s, label='ADAM')
    plt.plot(ts, xs_amsg_s, label='AMSGRAD')
    plt.plot(ts, xs_aamsg_s, label='ACADG')
    plt.legend()
    plt.savefig("./" + name + ".png")


experiment(name="stochastic")
experiment(name="online")
