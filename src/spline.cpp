#include <iostream>

#include <spline_solver/spline.hpp>
#include <spline_solver/constants.hpp>

using namespace std;
using namespace Eigen;

double Spline1::interpolate(double t) const
{
    const double t1 = t;
    const double t2 = t * t1;
    const double t3 = t * t2;
    const double t4 = t * t3;
    const double t5 = t * t4;

    return coeffs[0] + coeffs[1] * t1 + coeffs[2] * t2 + coeffs[3] * t3 + coeffs[4] * t4 + coeffs[5] * t5;
}

double Spline1::derivative(double t) const
{
    const double t1 = t;
    const double t2 = t * t1;
    const double t3 = t * t2;
    const double t4 = t * t3;

    return coeffs[1] + 2 * coeffs[2] * t1 + 3 * coeffs[3] * t2 + 4 * coeffs[4] * t3 + 5 * coeffs[5] * t4;
}

Spline1 Spline1::calculate(const double *a, const double *b)
{
    Spline1 out;
    out.coeffs[0] = a[0];
    out.coeffs[1] = a[1];
    out.coeffs[2] = 0.5 * a[2];
    out.coeffs[3] = -10 * a[0] - 6 * a[1] - 1.5 * a[2] + 10 * b[0] - 4 * b[1] + 0.5 * b[2];
    out.coeffs[4] =  15 * a[0] + 8 * a[1] + 1.5 * a[2] - 15 * b[0] + 7 * b[1] - 1.0 * b[2];
    out.coeffs[5] = - 6 * a[0] - 3 * a[1] - 0.5 * a[2] +  6 * b[0] - 3 * b[1] + 0.5 * b[2];

    return out;
}

template<unsigned int Dims>
typename Spline<Dims>::VectorNd Spline<Dims>::interpolate(double t) const
{
    VectorNd p;

    for (int k = 0; k < Dims; ++k)
    {
        p[k] = _dims[k].interpolate(t);
    }

    return p;
}

template<unsigned int Dims>
typename Spline<Dims>::VectorNd Spline<Dims>::derivative(double t) const
{
    VectorNd d;

    for (int k = 0; k < Dims; ++k)
    {
        d[k] = _dims[k].derivative(t);
    }

    return d;
}

template<unsigned int Dims>
void Spline<Dims>::walk(const double deltatau, void (*fn)(VectorNd, double, const Spline<Dims>&, void *), void *payload, double a, double b) const
{
    double tau = a;

    while (tau <= b)
    {
        if (b - tau < deltatau)
        {
            tau = b;
        }

        fn(interpolate(tau), tau, *this, payload);

        tau += deltatau;
    }
}

template<unsigned int Dims>
Spline<Dims> Spline<Dims>::calculate(Matrix<double, Dims, 3, RowMajor> A, Matrix<double, Dims, 3, RowMajor> B)
{
    Spline<Dims> out;
    const double *dataA = &A(0);
    const double *dataB = &B(0);

    for (int k = 0; k < Dims; ++k)
    {
        out._dims[k] = Spline1::calculate(dataA + 3 * k, dataB + 3 * k);
    }

    out.calculateLength();

    return out;
}

template<unsigned int Dims>
void Spline<Dims>::calculateLength()
{
    // 7-point Gauss-Quadrature for solving length integral
    length = 0.0;

    for (auto &c : GAUSS_LENGENDRE_COEFFICIENTS)
    {
        length += derivative(c(1)).norm() * c(0);
    }

    length *= 0.5;
}

template<unsigned int Dims>
typename SplinePath<Dims>::VectorNd SplinePath<Dims>::interpolate(double s)
{
    int m = 0;
    double s_ = s;

    while (s_ > children[m].length && m < children.size())
    {
        s_ -= children[m].length;
    }

    if (s_ >= 1.0)
    {
        s_ = 1.0;
    }

    return children[m].interpolate(s_ / children[m].length);
}

template<unsigned int Dims>
void SplinePath<Dims>::walk(const double deltatau, void (*fn)(VectorNd, SplinePath<Dims>&, double, Spline<Dims>&, void *), void *payload)
{
    double tau = 0.0;
    int m = 0;

    while (m < children.size())
    {
        double reltau = tau / children[m].length;
        VectorNd pt = children[m].interpolate(reltau);

        fn(pt, *this, reltau, children[m], payload);
        tau += deltatau;

        if (tau > children[m].length)
        {
            tau = 0.0;
            m++;
        }
    }
}

template<unsigned int Dims>
void SplinePath<Dims>::add(Spline<Dims> sp)
{
    children.push_back(sp);
    length += sp.length;
}

template<unsigned int Dims>
SplinePath<Dims> SplinePath<Dims>::calculate(MatrixNXd q, MatrixNXd v, MatrixNXd a)
{
    const int N = q.cols()-1;
    SplinePath<Dims> result;

    assert(q.cols() == N+1 && v.cols() == N+1 && a.cols() == N+1);

    Matrix<double, Dims, 3, RowMajor> A;
    Matrix<double, Dims, 3, RowMajor> B;

    A.col(0) = q.col(0);
    A.col(1) = v.col(0);
    A.col(2) = a.col(0);

    for (int k = 0; k < N; ++k)
    {
        B.col(0) = q.col(k+1);
        B.col(1) = v.col(k+1);
        B.col(2) = a.col(k+1);

        result.add(Spline<Dims>::calculate(A, B));

        A = B;
    }

    return result;
}

#define last N-1

template<unsigned int Dims>
SplinePath<Dims> SplineSolver<Dims>::solve(vector<VectorNd> points, VectorNd vel_start, VectorNd vel_end, VectorNd accel_start, VectorNd accel_end)
{
    const int N = points.size();
    Matrix<double, Dims, Dynamic, RowMajor> q(Dims, N);
    for (int j = 0; j < points.size(); ++j)
    {
        q.col(j) = points[j];
    }

    Matrix<double, Dims, Dynamic, RowMajor> v(Dims, N), a(Dims, N);
    VectorXd vv(N), va(N);

    for (int k = 0; k < Dims; ++k)
    {
        vv.setZero();
        va.setZero();

        vv(0) = vel_start(k);
        vv(last) = vel_end(k);
        va(0) = accel_start(k);
        va(last) = accel_end(k);

        if (!find_params_1d(q.row(k).transpose(), vv, va))
        {
            return SplinePath<Dims>();
        }

        // copy values
        v.row(k) = vv;
        a.row(k) = va;
    }

    return SplinePath<Dims>::calculate(q, v, a);
}

template<unsigned int Dims>
bool SplineSolver<Dims>::find_params_1d(VectorXd q, VectorXd &v, VectorXd &a)
{
    if (q.rows() > max_n)
    {
        const int N = q.rows();

        for (int k = 0; k <= N - max_n; k++)
        {
            int l;
            // from start
            if (k % 2 == 0)
            {
                l = k / 2;
            }
            // from end
            else
            {
                l = N - max_n - (k-1) / 2;
            }

            VectorXd vseg = v.segment(l, max_n);
            VectorXd aseg = a.segment(l, max_n);

            if (!find_params_1d_n(q.segment(l, max_n), vseg, aseg))
            {
                return false;
            }

            v.segment(l, max_n) = vseg;
            a.segment(l, max_n) = aseg;
        }

        return true;
    }
    else
    {
        return find_params_1d_n(q, v, a);
    }
}

template<unsigned int Dims>
bool SplineSolver<Dims>::find_params_1d_n(VectorXd q, VectorXd &v, VectorXd &a)
{
    assert(q.rows() > 2);
    const int N = q.rows();
    const int M = 2*(N-2);

    // Assemble b vector
    VectorXd b(M);
    for (int k = 0; k<N-2; ++k)
    {
        b(k) = 15*q(k+2) - 15*q(k);
        b(N-2+k) = -20*q(k+2) + 40*q(k+1) - 20*q(k);
    }

    b(0) += -7*v(0)-a(0);
    b(N-2-1) += -7*v(last)+a(last);

    b(N-2) += -8*v(0) - a(0);
    b(2*(N-2)-1) += 8*v(last) - a(last);

    // Assemble 2(N-2) matrix
    if (A.rows() != M)
    {
        if (!build_solver(N, M))
        {
            cout << "inverse failed" << endl;
            return false;
        }
    }

    VectorXd sol = solver.solve(b);

    if(solver.info()!=Success)
    {
        // solving failed
        cout << "solving failed" << endl;
        return false;
    }

    v.segment(1, N-2) = sol.segment(0, N-2);
    a.segment(1, N-2) = sol.segment(N-2, N-2);

    return true;
}

template<unsigned int Dims>
bool SplineSolver<Dims>::build_solver(const int N, const int M)
{
    A = SparseMatrix<double>(M, M);
    A.reserve(VectorXi::Constant(M, 6));

    for (int k = 0; k<N-2; ++k)
    {
        A.insert(k, k) = 16;
        A.insert(N-2+k, k) = 0;
        A.insert(k, N-2+k) = 0;
        A.insert(N-2+k, N-2+k) = -6;

        if (k > 0)
        {
            A.insert(k, k-1) = 7;
            A.insert(N-2+k, k-1) = 8;
            A.insert(k, N-2+k-1) = 1;
            A.insert(N-2+k, N-2+k-1) = 1;
        }

        if (k < N-3)
        {
            A.insert(k, k+1) = 7;
            A.insert(N-2+k, k+1) = -8;
            A.insert(k, N-2+k+1) = -1;
            A.insert(N-2+k, N-2+k+1) = 1;
        }
    }

    A.makeCompressed();
    solver.compute(A);

    if (solver.info() != Success)
    {
        cout << "inverse failed" << endl;
        return false;
    }

    return true;
}

template class Spline<1>;
template class SplinePath<1>;
template class SplineSolver<1>;

template class Spline<2>;
template class SplinePath<2>;
template class SplineSolver<2>;

template class Spline<3>;
template class SplinePath<3>;
template class SplineSolver<3>;

