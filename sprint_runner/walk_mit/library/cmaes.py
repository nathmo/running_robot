"""A small (mu/mu_w, lambda)-CMA-ES with box constraints -- no third-party dependency on the cluster.

Hansen's standard formulation (rank-mu update, cumulative step-size adaptation), enough for the
17-dim gait-library solves and the closed-loop search of artifact §09. Deterministic for a seed.

    es = CMAES(x0, sigma0, lo, hi, popsize=16, seed=0)
    while not es.done(max_evals):
        X = es.ask()                     # [popsize, n] candidates inside [lo, hi]
        es.tell(X, [f(x) for x in X])    # minimise f
    es.best_x, es.best_f
"""
import numpy as np


class CMAES:
    def __init__(self, x0, sigma0, lo=None, hi=None, popsize=None, seed=0):
        self.n = int(len(x0))
        self.mean = np.asarray(x0, dtype=float).copy()
        self.sigma = float(sigma0)
        self.lo = None if lo is None else np.asarray(lo, dtype=float)
        self.hi = None if hi is None else np.asarray(hi, dtype=float)
        self.rng = np.random.default_rng(seed)
        n = self.n
        self.lam = int(popsize) if popsize else 4 + int(3 * np.log(n))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.evals = 0
        self.gen = 0
        self.best_x = self.mean.copy()
        self.best_f = np.inf
        self.history = []

    def _clip(self, x):
        if self.lo is not None:
            x = np.maximum(x, self.lo)
        if self.hi is not None:
            x = np.minimum(x, self.hi)
        return x

    def ask(self):
        z = self.rng.standard_normal((self.lam, self.n))
        y = (z * self.D) @ self.B.T
        X = self.mean + self.sigma * y
        self._last_y = y
        Xc = np.array([self._clip(x) for x in X])
        # repaired samples: keep the search geometry consistent with what was evaluated
        self._last_y = (Xc - self.mean) / self.sigma
        return Xc

    def tell(self, X, f):
        f = np.asarray(f, dtype=float)
        self.evals += len(f)
        self.gen += 1
        idx = np.argsort(f)
        if f[idx[0]] < self.best_f:
            self.best_f = float(f[idx[0]])
            self.best_x = np.asarray(X[idx[0]], dtype=float).copy()
        self.history.append((self.evals, float(f[idx[0]]), float(np.median(f))))
        y = self._last_y[idx[:self.mu]]
        yw = self.w @ y
        old_mean = self.mean.copy()
        self.mean = self._clip(old_mean + self.sigma * yw)
        # step-size path (in the whitened frame)
        invsqrtC = self.B @ np.diag(1.0 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (invsqrtC @ yw)
        hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chiN
                < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        rank_mu = sum(wi * np.outer(yi, yi) for wi, yi in zip(self.w, y))
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * rank_mu)
        self.sigma *= float(np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1)))
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        Dsq, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(Dsq, 1e-20))

    def done(self, max_evals, tol_sigma=1e-6):
        return self.evals >= max_evals or self.sigma * float(self.D.max()) < tol_sigma


if __name__ == "__main__":
    # self-test: a shifted Rosenbrock in a box
    def rosen(x):
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))
    es = CMAES(np.full(6, -0.5), 0.5, lo=np.full(6, -2.0), hi=np.full(6, 2.0), popsize=16, seed=1)
    while not es.done(6000):
        X = es.ask()
        es.tell(X, [rosen(x) for x in X])
    print(f"cmaes self-test: f_best {es.best_f:.2e} after {es.evals} evals, x {np.round(es.best_x, 3)}")
    assert es.best_f < 1e-2, es.best_f
    print("cmaes self-test OK")
