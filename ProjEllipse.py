import numpy as np
class ProjEllipse:
    def __init__(self):
        self.coef_ = None

    def project(self, lt, rt, bl, br):
        # (-1,  1, 1) ( 1,  1, 1)
        # (-1, -1, 1) ( 1, -1, 1)
        # (W0, W1, 1) (X0, X1, 1)
        # (Z0, Z1, 1) (Y0, Y1, 1) 
        W0, W1 = lt
        X0, X1 = rt
        Z0, Z1 = bl
        Y0, Y1 = br
        A =  np.float64(X0*Y0*Z1 - W0*Y0*Z1 - X0 *Y1* Z0 + W0* Y1* Z0 - W0* X1* Z0 + W1* X0* Z0 + W0* X1* Y0 - W1* X0* Y0)
        B =  np.float64(W0*Y0*Z1 - W0*X0*Z1 - X0 *Y1* Z0 + X1* Y0* Z0 - W1* Y0* Z0 + W1* X0* Z0 + W0* X0* Y1 - W0* X1* Y0)
        C =  np.float64(X0*Y0*Z1 - W0*X0*Z1 - W0 *Y1* Z0 - X1* Y0* Z0 + W1* Y0* Z0 + W0* X1* Z0 + W0* X0* Y1 - W1* X0* Y0)
        D =  np.float64(X1*Y0*Z1 - W1*Y0*Z1 - W0 *X1* Z1 + W1* X0* Z1 - X1* Y1* Z0 + W1* Y1* Z0 + W0* X1* Y1 - W1* X0* Y1)
        E = np.float64(-X0*Y1*Z1 + W0*Y1*Z1 + X1 *Y0* Z1 - W0* X1* Z1 - W1* Y1* Z0 + W1* X1* Z0 + W1* X0* Y1 - W1* X1* Y0)
        F =  np.float64(X0*Y1*Z1 - W0*Y1*Z1 + W1 *Y0* Z1 - W1* X0* Z1 - X1* Y1* Z0 + W1* X1* Z0 + W0* X1* Y1 - W1* X1* Y0)
        G = np.float64( X0*Z1 - W0* Z1 - X1* Z0 + W1* Z0 - X0* Y1 + W0* Y1 + X1* Y0 - W1* Y0)
        H = np.float64( Y0*Z1 - X0* Z1 - Y1* Z0 + X1* Z0 + W0* Y1 - W1* Y0 - W0* X1 + W1* X0)
        I = np.float64(Y0*Z1 - W0* Z1 - Y1* Z0 + W1* Z0 + X0* Y1 - X1* Y0 + W0* X1 - W1* X0)

        T = np.array([[A, B, C],
                    [D, E, F],
                    [G, H, I]], dtype=np.float64)
        s = np.matmul(T, np.array([[1],
                                [1],
                                [1]]))[2]
        T /=s
        T_inv = np.linalg.inv(T)
        J = T_inv[0, 0]
        K = T_inv[0, 1]
        L = T_inv[0, 2]
        M = T_inv[1, 0]
        N = T_inv[1, 1]
        O = T_inv[1, 2]
        P = T_inv[2, 0]
        Q = T_inv[2, 1]
        R = T_inv[2, 2]
        
        # ax2 + bxy + cy2 + dx + ey + f = 0
        # ax2 + 2bxy + cy2 + 2dx + 2fy + g = 0
        a = J**2 + M**2 - P**2
        b = 2 * (J*K + M*N - P*Q)
        c = K**2 + N**2 - Q**2
        d = 2 * (J*L + M*O - P*R)
        e = 2* (K*L + N*O - Q*R)
        f = L**2 + O**2 - R**2

        self.coef_ = np.array([a,b,c,d,e,f]).reshape(-1,1)

        return self

    @property
    def coefficients(self):

        return tuple(c for c in self.coef_.ravel())

    def as_parameters(self):
        """Returns the definition of the fitted ellipse as localized parameters

        Returns
        _______
        center : tuple
            (x0, y0)
        width : float
            Total length (diameter) of horizontal axis.
        height : float
            Total length (diameter) of vertical axis.
        phi : float
            The counterclockwise angle [radians] of rotation from the x-axis to the semimajor axis
        """

        # Eigenvectors are the coefficients of an ellipse in general form
        # the division by 2 is required to account for a slight difference in
        # the equations between (*) and (**)
        # a*x^2 +   b*x*y + c*y^2 +   d*x +   e*y + f = 0  (*)  Eqn 1
        # a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0  (**) Eqn 15
        # We'll use (**) to follow their documentation
        a = self.coefficients[0]
        b = self.coefficients[1] / 2.
        c = self.coefficients[2]
        d = self.coefficients[3] / 2.
        f = self.coefficients[4] / 2.
        g = self.coefficients[5]

        # Finding center of ellipse [eqn.19 and 20] from (**)
        x0 = (c*d - b*f) / (b**2 - a*c)
        y0 = (a*f - b*d) / (b**2 - a*c)
        center = (x0, y0)

        # Find the semi-axes lengths [eqn. 21 and 22] from (**)
        numerator = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
        denominator1 = (b**2 - a*c) * ( np.sqrt((a-c)**2+4*b**2) - (c+a))  # noqa: E201
        denominator2 = (b**2 - a*c) * (-np.sqrt((a-c)**2+4*b**2) - (c+a))
        height = np.sqrt(numerator / denominator1)
        width = np.sqrt(numerator / denominator2)

        # Angle of counterclockwise rotation of major-axis of ellipse to x-axis
        # [eqn. 23] from (**)
        # w/ trig identity eqn 9 form (***)
        if b == 0 and a > c:
            phi = 0.0
        elif b == 0 and a < c:
            phi = np.pi/2
        elif b != 0 and a > c:
            phi = 0.5 * np.arctan(2*b/(a-c))
        elif b != 0 and a < c:
            phi = 0.5 * (np.pi + np.arctan(2*b/(a-c)))
        elif a == c:
            phi = 0.0
        else:
            raise RuntimeError("Unreachable")

        return center, width, height, phi

    def return_samples(self, n_points=None, t=None):
        

        if n_points is None and t is None:
            raise AttributeError("A value for `n_points` or `t` must be ",
                                 "provided")

        if t is None:
            t = np.linspace(0, 2*np.pi, n_points)

        center, width, height, phi = self.as_parameters()

        x = (center[0] + width * np.cos(t) * np.cos(phi) - height * np.sin(t) * np.sin(phi))
        y = (center[1] + width * np.cos(t) * np.sin(phi) + height * np.sin(t) * np.cos(phi))

        return np.c_[x, y]
if __name__ == "__main__":
    p = ProjEllipse()
    lt = np.array([100, 400])
    rt = np.array([400, 400])
    bl = np.array([100, 100])
    br = np.array([400, 100])
    p.project(lt, rt, bl, br)
    print(p.as_parameters())
    p.return_samples(200)