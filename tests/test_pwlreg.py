import numpy as np
import numpy.testing as nptest
from sklearn.linear_model import LinearRegression

import pwlreg as pw


class TestPWLReg:
    x_small = np.array([0.0, 1.0, 1.5, 2.0])
    y_small = np.array([0.0, 1.0, 1.1, 1.5])
    x_sin = np.linspace(0, 10, num=100)
    y_sin = np.sin(x_sin * np.pi / 2)
    y = np.array([0.00000000e+00, 9.69801700e-03, 2.94350340e-02,
                  4.39052750e-02, 5.45343950e-02, 6.74104940e-02,
                  8.34831790e-02, 1.02580042e-01, 1.22767939e-01,
                  1.42172312e-01, 0.00000000e+00, 8.58600000e-06,
                  8.31543400e-03, 2.34184100e-02, 3.39709150e-02,
                  4.03581990e-02, 4.53545600e-02, 5.02345260e-02,
                  5.55253360e-02, 6.14750770e-02, 6.82125120e-02,
                  7.55892510e-02, 8.38356810e-02, 9.26413070e-02,
                  1.02039790e-01, 1.11688258e-01, 1.21390666e-01,
                  1.31196948e-01, 0.00000000e+00, 1.56706510e-02,
                  3.54628780e-02, 4.63739040e-02, 5.61442590e-02,
                  6.78542550e-02, 8.16388310e-02, 9.77756110e-02,
                  1.16531753e-01, 1.37038283e-01, 0.00000000e+00,
                  1.16951050e-02, 3.12089850e-02, 4.41776550e-02,
                  5.42877590e-02, 6.63321350e-02, 8.07655920e-02,
                  9.70363280e-02, 1.15706975e-01, 1.36687642e-01,
                  0.00000000e+00, 1.50144640e-02, 3.44519970e-02,
                  4.55907760e-02, 5.59556700e-02, 6.88450940e-02,
                  8.41374060e-02, 1.01254006e-01, 1.20605073e-01,
                  1.41881288e-01, 1.62618058e-01])
    x = np.array([0.00000000e+00, 8.82678000e-03, 3.25615100e-02,
                  5.66106800e-02, 7.95549800e-02, 1.00936330e-01,
                  1.20351520e-01, 1.37442010e-01, 1.51858250e-01,
                  1.64433570e-01, 0.00000000e+00, -2.12600000e-05,
                  7.03872000e-03, 1.85494500e-02, 3.00926700e-02,
                  4.17617000e-02, 5.37279600e-02, 6.54941000e-02,
                  7.68092100e-02, 8.76596300e-02, 9.80525800e-02,
                  1.07961810e-01, 1.17305210e-01, 1.26063930e-01,
                  1.34180360e-01, 1.41725010e-01, 1.48629710e-01,
                  1.55374770e-01, 0.00000000e+00, 1.65610200e-02,
                  3.91016100e-02, 6.18679400e-02, 8.30997400e-02,
                  1.02132890e-01, 1.19011260e-01, 1.34620080e-01,
                  1.49429370e-01, 1.63539960e-01, -0.00000000e+00,
                  1.01980300e-02, 3.28642800e-02, 5.59461900e-02,
                  7.81388400e-02, 9.84458400e-02, 1.16270210e-01,
                  1.31279040e-01, 1.45437090e-01, 1.59627540e-01,
                  0.00000000e+00, 1.63404300e-02, 4.00086000e-02,
                  6.34390200e-02, 8.51085900e-02, 1.04787860e-01,
                  1.22120350e-01, 1.36931660e-01, 1.50958760e-01,
                  1.65299640e-01, 1.79942720e-01])

    def test_pwlr(self):
        m = pw.AutoPiecewiseRegression(n_segments=2)
        m.fit(self.x_small, self.y_small)

    def test_matrix_assembly(self):
        x0 = self.x_small.copy()
        A = pw.pwlreg._assemble_regression_matrix(x0, breaks=x0)
        Atruth = np.array(
            [
                [1.,            0.,            0.,            0.],
                [1., x0[1] - x0[0],            0.,            0.],
                [1., x0[2] - x0[0], x0[2] - x0[1],            0.],
                [1., x0[3] - x0[0], x0[3] - x0[1], x0[3] - x0[2]],
            ]
        )
        assert np.allclose(A, Atruth)

    def test_fit(self):
        m = pw.AutoPiecewiseRegression(n_segments=4)
        m.fit(self.x_small, self.y_small)
        assert np.isclose(m.ssr_, 0.)

    def test_fit_rng(self):
        seed = 1234
        m = pw.AutoPiecewiseRegression(n_segments=2, random_state=np.random.RandomState(seed=seed))
        coefs1 = m.fit(self.x_small, self.y_small).coef_
        m = pw.AutoPiecewiseRegression(n_segments=2, random_state=np.random.RandomState(seed=seed))
        coefs2 = m.fit(self.x_small, self.y_small).coef_
        nptest.assert_equal(coefs1, coefs2)

    def test_predict_nobreaks(self):
        m = pw.PiecewiseLinearRegression(breakpoints=[np.min(self.x_small), np.max(self.x_small)])
        m.fit(self.x_small, self.y_small)
        y_pred = m.predict(self.x_small)

        m2 = LinearRegression()
        X = np.reshape(self.x_small, (-1, 1))
        m2.fit(X, self.y_small)
        y2_pred = m2.predict(X)

        assert np.allclose(y_pred, y2_pred)
