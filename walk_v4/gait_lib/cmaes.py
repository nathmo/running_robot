"""Minimal (mu/mu_w, lambda)-CMA-ES (Hansen's tutorial defaults), numpy, batched evaluation.

    es = CMAES(x0, sigma0, popsize=16, bounds=(lo, hi))
    while not es.done:
        X = es.ask()                 # [popsize, n]
        f = objective(X)             # [popsize]  (evaluate the whole population at once: GPU)
        es.tell(X, f)
    es.best_x, es.best_f
"""
import numpy as np


class CMAES:
    def __init__(self, x0, sigma0, popsize=16, bounds=None, max_evals=2000, seed=0, ftol=1e-9):
        self.n = len(x0)
        self.mean = np.asarray(x0, float).copy()
        self.sigma = float(sigma0)
        self.lam = int(popsize)
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        n = self.n
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.bounds = None if bounds is None else (np.asarray(bounds[0], float), np.asarray(bounds[1], float))
        self.rng = np.random.default_rng(seed)
        self.max_evals = int(max_evals)
        self.evals = 0
        self.gen = 0
        self.best_x, self.best_f = self.mean.copy(), np.inf
        self.ftol = ftol
        self.hist = []

    @property
    def done(self):
        if self.evals >= self.max_evals:
            return True
        if len(self.hist) > 10 and (max(self.hist[-10:]) - min(self.hist[-10:])) < self.ftol:
            return True
        return self.sigma * self.D.max() < 1e-8

    def ask(self):
        z = self.rng.standard_normal((self.lam, self.n))
        self._y = (z * self.D) @ self.B.T
        X = self.mean + self.sigma * self._y
        if self.bounds is not None:
            X = np.clip(X, self.bounds[0], self.bounds[1])
            self._y = (X - self.mean) / self.sigma
        return X

    def tell(self, X, f):
        f = np.asarray(f, float)
        f = np.where(np.isfinite(f), f, np.inf)
        self.evals += len(f)
        self.gen += 1
        order = np.argsort(f)
        if f[order[0]] < self.best_f:
            self.best_f, self.best_x = float(f[order[0]]), X[order[0]].copy()
        self.hist.append(float(f[order[0]]))
        y = self._y[order[:self.mu]]
        y_w = self.w @ y
        self.mean = self.mean + self.sigma * y_w
        # C^{-1/2} y_w = B D^{-1} B^T y_w
        c_inv_half_yw = self.B @ ((self.B.T @ y_w) / self.D)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * c_inv_half_yw
        hsig = float(np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chiN
                     < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w
        rank_mu = sum(wi * np.outer(yi, yi) for wi, yi in zip(self.w, y))
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * rank_mu)
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        D2, B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(D2, 1e-20))
        self.B = B
