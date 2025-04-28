"""Test suite for pwlreg."""

import numpy as np
import numpy.testing as nptest
import pytest
from sklearn.linear_model import LinearRegression

import pwlreg as pw


class TestPWLReg:
    """Regression tests."""

    x_small = np.array([0.0, 1.0, 1.5, 2.0])
    y_small = np.array([0.0, 1.0, 1.1, 1.5])

    y = np.array(
        [
            0.00000000e00,
            9.69801700e-03,
            2.94350340e-02,
            4.39052750e-02,
            5.45343950e-02,
            6.74104940e-02,
            8.34831790e-02,
            1.02580042e-01,
            1.22767939e-01,
            1.42172312e-01,
            0.00000000e00,
            8.58600000e-06,
            8.31543400e-03,
            2.34184100e-02,
            3.39709150e-02,
            4.03581990e-02,
            4.53545600e-02,
            5.02345260e-02,
            5.55253360e-02,
            6.14750770e-02,
            6.82125120e-02,
            7.55892510e-02,
            8.38356810e-02,
            9.26413070e-02,
            1.02039790e-01,
            1.11688258e-01,
            1.21390666e-01,
            1.31196948e-01,
            0.00000000e00,
            1.56706510e-02,
            3.54628780e-02,
            4.63739040e-02,
            5.61442590e-02,
            6.78542550e-02,
            8.16388310e-02,
            9.77756110e-02,
            1.16531753e-01,
            1.37038283e-01,
            0.00000000e00,
            1.16951050e-02,
            3.12089850e-02,
            4.41776550e-02,
            5.42877590e-02,
            6.63321350e-02,
            8.07655920e-02,
            9.70363280e-02,
            1.15706975e-01,
            1.36687642e-01,
            0.00000000e00,
            1.50144640e-02,
            3.44519970e-02,
            4.55907760e-02,
            5.59556700e-02,
            6.88450940e-02,
            8.41374060e-02,
            1.01254006e-01,
            1.20605073e-01,
            1.41881288e-01,
            1.62618058e-01,
        ]
    )
    x = np.array(
        [
            0.00000000e00,
            8.82678000e-03,
            3.25615100e-02,
            5.66106800e-02,
            7.95549800e-02,
            1.00936330e-01,
            1.20351520e-01,
            1.37442010e-01,
            1.51858250e-01,
            1.64433570e-01,
            0.00000000e00,
            -2.12600000e-05,
            7.03872000e-03,
            1.85494500e-02,
            3.00926700e-02,
            4.17617000e-02,
            5.37279600e-02,
            6.54941000e-02,
            7.68092100e-02,
            8.76596300e-02,
            9.80525800e-02,
            1.07961810e-01,
            1.17305210e-01,
            1.26063930e-01,
            1.34180360e-01,
            1.41725010e-01,
            1.48629710e-01,
            1.55374770e-01,
            0.00000000e00,
            1.65610200e-02,
            3.91016100e-02,
            6.18679400e-02,
            8.30997400e-02,
            1.02132890e-01,
            1.19011260e-01,
            1.34620080e-01,
            1.49429370e-01,
            1.63539960e-01,
            -0.00000000e00,
            1.01980300e-02,
            3.28642800e-02,
            5.59461900e-02,
            7.81388400e-02,
            9.84458400e-02,
            1.16270210e-01,
            1.31279040e-01,
            1.45437090e-01,
            1.59627540e-01,
            0.00000000e00,
            1.63404300e-02,
            4.00086000e-02,
            6.34390200e-02,
            8.51085900e-02,
            1.04787860e-01,
            1.22120350e-01,
            1.36931660e-01,
            1.50958760e-01,
            1.65299640e-01,
            1.79942720e-01,
        ]
    )

    x_wt = np.array(
        [
            65.43888437,
            25.47311425,
            61.85749766,
            17.5333924,
            21.37950291,
            7.91211261,
            16.19834165,
            21.34177323,
            64.59330943,
            17.66453689,
            29.54741018,
            40.86134423,
            95.49950279,
            95.50400314,
            89.27108334,
            88.77585348,
            91.28000404,
            74.35087072,
            72.67818409,
            95.72369409,
            68.98457571,
            89.561734,
            89.15085463,
            87.16359338,
            68.98453132,
            99.26638603,
            81.48540368,
            84.57563571,
            67.10336549,
            75.29181448,
            95.33018443,
            81.03484559,
            91.2820268,
            97.42742615,
            72.06464765,
            99.74455457,
            73.01694883,
            98.02372578,
            69.86714082,
            82.4509536,
            94.3566436,
            76.27472468,
            95.68001987,
            99.22174695,
            94.7765507,
            81.81282881,
            79.22677818,
            82.92809142,
            70.04199105,
            74.48054247,
            84.70668581,
            91.19664449,
            81.81405603,
            77.0815254,
            98.26286691,
            91.7841166,
            82.57759918,
            82.09010403,
            91.62914751,
            95.22819597,
        ]
    )
    y_wt = np.array(
        [
            8.97880375,
            10.848107,
            10.49808261,
            9.91555836,
            10.20249333,
            9.83619422,
            10.83705971,
            9.28756034,
            8.82585013,
            10.47526795,
            11.73739481,
            9.86335634,
            33.23462012,
            23.78659682,
            27.89500429,
            24.9981975,
            24.56649716,
            15.64591645,
            11.28936398,
            24.8824468,
            8.95420215,
            13.4774194,
            25.04721503,
            19.4818676,
            12.29725968,
            26.89833196,
            14.78996983,
            29.3125192,
            11.45011255,
            14.53061505,
            24.28185342,
            22.14938715,
            20.64677086,
            33.69656426,
            13.68423099,
            18.55806064,
            14.42253021,
            31.82904077,
            12.21504628,
            20.92386072,
            30.69449996,
            13.46260248,
            36.29032943,
            27.05361143,
            28.19597908,
            12.31542127,
            18.9774047,
            20.41061017,
            10.67105297,
            15.1893247,
            20.70096165,
            26.7345677,
            19.71340063,
            13.64807196,
            25.94367206,
            14.37849188,
            17.86041382,
            15.86200352,
            27.33210288,
            31.14143882,
        ]
    )
    wts = np.array(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            5.27492542,
            5.27560047,
            4.3406625,
            4.26637802,
            4.64200061,
            2.10263061,
            1.85172761,
            5.30855411,
            1.29768636,
            4.3842601,
            4.32262819,
            4.02453901,
            1.2976797,
            5.8399579,
            3.17281055,
            3.63634536,
            1.01550482,
            2.24377217,
            5.24952767,
            3.10522684,
            4.64230402,
            5.56411392,
            1.75969715,
            5.91168319,
            1.90254233,
            5.65355887,
            1.43007112,
            3.31764304,
            5.10349654,
            2.3912087,
            5.30200298,
            5.83326204,
            5.1664826,
            3.22192432,
            2.83401673,
            3.38921371,
            1.45629866,
            2.12208137,
            3.65600287,
            4.62949667,
            3.22210841,
            2.51222881,
            5.68943004,
            4.71761749,
            3.33663988,
            3.2635156,
            4.69437213,
            5.2342294,
        ]
    )

    def test_matrix_assembly_deg1(self):
        """Matrix is assembled correctly for the one degree case."""
        x0 = self.x_small.copy()
        breaks = [x0.min(), 1.25, x0.max()]
        A = pw.pwlreg._assemble_regression_matrix(self.x_small, breaks, degree=[1, 1])
        Atruth = np.array(
            [
                [1.0, x0[0], 0.0, 0.0],
                [1.0, x0[1], 0.0, 0.0],
                [0.0, 0.0, 1.0, x0[2]],
                [0.0, 0.0, 1.0, x0[3]],
            ]
        )
        assert np.allclose(A, Atruth)

    def test_matrix_assembly_deg2(self):
        """Matrix is assembled correctly for the two degree case."""
        x0 = self.x_small.copy()
        breaks = [x0.min(), 1.25, x0.max()]
        A = pw.pwlreg._assemble_regression_matrix(self.x_small, breaks, degree=[2, 2])
        Atruth = np.array(
            [
                [1.0, x0[0], x0[0] ** 2, 0.0, 0.0, 0.0],
                [1.0, x0[1], x0[1] ** 2, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, x0[2], x0[2] ** 2],
                [0.0, 0.0, 0.0, 1.0, x0[3], x0[3] ** 2],
            ]
        )
        assert np.allclose(A, Atruth)

    def test_continuity_constraints_deg1(self):
        """Continuity constraints are correct for the one degree case."""
        breaks = [np.min(self.x_small), 1.25, 1.75, np.max(self.x_small)]
        C = pw.pwlreg._assemble_continuity_constraints(breaks, degree=[1, 1, 1])
        Ctruth = np.array(
            [
                [1.0, breaks[1], -1.0, -breaks[1], 0.0, 0.0],
                [0.0, 0.0, 1.0, breaks[2], -1.0, -breaks[2]],
            ]
        )
        assert np.allclose(C, Ctruth)

    def test_continuity_constraints_deg2(self):
        """Continuity constraints are correct for the two degree case."""
        b = [np.min(self.x_small), 1.25, 1.75, np.max(self.x_small)]
        C = pw.pwlreg._assemble_continuity_constraints(b, degree=[2, 2, 2])
        Ctruth = np.array(
            [
                [1.0, b[1], b[1] ** 2, -1.0, -b[1], -(b[1] ** 2), 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, b[2], b[2] ** 2, -1.0, -b[2], -(b[2] ** 2)],
            ]
        )
        assert np.allclose(C, Ctruth)

    def test_fit(self):
        """Fit method executes."""
        m = pw.AutoPiecewiseRegression(n_segments=3)
        m.fit(self.x, self.y)
        assert pytest.approx(m.ssr_, rel=1e-2) == 2.438e-4

    def test_fit_rng(self):
        """Fit method is deterministic given a seed."""
        seed = 1234
        m = pw.AutoPiecewiseRegression(n_segments=2, random_state=np.random.RandomState(seed=seed))
        coefs1 = m.fit(self.x_small, self.y_small).coef_
        m = pw.AutoPiecewiseRegression(n_segments=2, random_state=np.random.RandomState(seed=seed))
        coefs2 = m.fit(self.x_small, self.y_small).coef_
        nptest.assert_equal(coefs1, coefs2)

    def test_predict_nobreaks_equals_sklearn_regression(self):
        """Fit aligns with sklearn regression results."""
        m = pw.PiecewiseLinearRegression()
        m.fit(self.x_small, self.y_small)
        y_pred = m.predict(self.x_small)

        m2 = LinearRegression()
        X = np.reshape(self.x_small, (-1, 1))
        m2.fit(X, self.y_small)
        y2_pred = m2.predict(X)

        assert np.allclose(y_pred, y2_pred)

    def test_pw_degree_shortform(self):
        """Degree shorthand expands properly."""
        m = pw.PiecewiseLinearRegression(
            breakpoints=[np.min(self.x), 0.04, np.max(self.x)],
            degree=1,
        )
        m.fit(self.x, self.y)
        assert m.degree == [1, 1]

    def test_pw_wrong_number_of_degrees(self):
        """Wrong number of degrees fails gracefully."""
        m = pw.PiecewiseLinearRegression(degree=[1, 2, 3])
        with pytest.raises(
            ValueError,
            match="The number of degree values must match the number of segments",
        ):
            m.fit(self.x, self.y)

    def test_pw_wrong_continuity(self):
        """Wrong continuity value fails gracefully."""
        m = pw.PiecewiseLinearRegression(continuity="c1000")
        with pytest.raises(ValueError, match="Continuity must be one of"):
            m.fit(self.x, self.y)

    def test_pw_weights(self):
        """Weighted regression runs without problems."""
        m = pw.PiecewiseLinearRegression(
            breakpoints=[np.min(self.x_wt), 65, np.max(self.x_wt)],
            degree=[0, 1],
        )
        m.fit(self.x_wt, self.y_wt, weights=1 / self.wts)
        assert pytest.approx(m.ssr_, rel=1e-2) == 862.5051

    def test_pw_weights_2d_fails(self):
        """Weights with incorrect dimensions fails gracefully."""
        m = pw.PiecewiseLinearRegression(
            breakpoints=[np.min(self.x_wt), 65, np.max(self.x_wt)],
            degree=[0, 1],
        )
        wts = np.column_stack((1 / self.wts, 1 / self.wts))
        with pytest.raises(ValueError, match="must be 1D"):
            m.fit(self.x_wt, self.y_wt, weights=wts)

    def test_pw_weights_single_value(self):
        """Single value for weights expands properly."""
        m = pw.PiecewiseLinearRegression(
            breakpoints=[np.min(self.x_wt), 65, np.max(self.x_wt)],
            degree=[0, 1],
        )
        m.fit(self.x_wt, self.y_wt, weights=1)
        assert pytest.approx(m.ssr_, abs=1e-2) == 849.0556

    def test_pw_wrong_dim_weights_fails(self):
        """Weights array with incorrect dimensions fails gracefully."""
        m = pw.PiecewiseLinearRegression(
            breakpoints=[np.min(self.x_wt), 65, np.max(self.x_wt)],
            degree=[0, 1],
        )
        with pytest.raises(ValueError, match="weights.shape =="):
            m.fit(self.x_wt, self.y_wt, weights=1 / self.wts[:10])

    def test_autopw_wrong_solver(self):
        """Unrecognized solver fails gracefully."""
        m = pw.AutoPiecewiseRegression(n_segments=3, solver="blarg")
        with pytest.raises(ValueError, match="Valid solvers are"):
            m.fit(self.x, self.y)

    def test_autopw_lbfgsb(self):
        """Auto regression L-BFGS-B solver runs."""
        m = pw.AutoPiecewiseRegression(n_segments=3, solver="L-BFGS-B")
        m.fit(self.x, self.y)
        assert pytest.approx(m.ssr_, rel=1e-2) == 2.438e-4

    def test_autopw_neldermead(self):
        """Auto regression Nelder-Mead solver runs."""
        m = pw.AutoPiecewiseRegression(n_segments=3, solver="Nelder-Mead")
        m.fit(self.x, self.y)
        assert pytest.approx(m.ssr_, rel=1e-2) == 2.438e-4

    def test_autopw_powell(self):
        """Auto regression Powell solver runs."""
        m = pw.AutoPiecewiseRegression(n_segments=3, solver="Powell")
        m.fit(self.x, self.y)
        assert pytest.approx(m.ssr_, rel=1e-2) == 2.438e-4

    def test_autopw_one_segment(self):
        """Auto regression single segment expands correctly."""
        m = pw.AutoPiecewiseRegression(n_segments=1)
        m.fit(self.x, self.y)
        assert pytest.approx(m.ssr_, rel=1e-2) == 3.368e-3

    def test_autopw_predict(self):
        """Auto regression prediction runs."""
        m = pw.AutoPiecewiseRegression(n_segments=3)
        m.fit(self.x, self.y)
        preds = m.predict(self.x)
        assert pytest.approx(np.mean(preds), rel=1e-2) == 6.583e-2
