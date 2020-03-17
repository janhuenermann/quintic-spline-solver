#include <iostream>

#include <spline_solver/spline.hpp>
#include <spline_solver/constants.hpp>

using namespace std;
using namespace Eigen;

template<unsigned int Order>
double UnitBoundedPolynomial1<Order>::interpolate(double t) const
{
    double y = 0.0, tt = 1;

    for (int k = 0; k <= Order; ++k)
    {
        y += coeffs[k] * tt;
        tt *= t;
    }

    return y;
}

template<unsigned int Order>
double UnitBoundedPolynomial1<Order>::derivative(double t) const
{
    double dy = 0.0, tt = 1;

    for (int k = 1; k <= Order; ++k)
    {
        dy += coeffs[k] * tt * (double)k;
        tt *= t;
    }

    return dy;
}

template<unsigned int Order>
double UnitBoundedPolynomial1<Order>::derivative2(double t) const
{
    double dy = 0.0, tt = 1;

    for (int k = 2; k <= Order; ++k)
    {
        dy += coeffs[k] * tt * (double)(k * (k-1));
        tt *= t;
    }

    return dy;
}

template<>
UnitBoundedPolynomial1<3> UnitBoundedPolynomial1<3>::fit(const Matrix<double, RequiredValues, 1> a, 
                                                         const Matrix<double, RequiredValues, 1> b)
{
    UnitBoundedPolynomial1<3> out;
    out.coeffs[0] = a[0];
    out.coeffs[1] = a[1];
    out.coeffs[2] = -3 * a[0] - 2 * a[1] + 3 * b[0] - 1 * b[1];
    out.coeffs[3] =  2 * a[0] + 1 * a[1] - 2 * b[0] + 1 * b[1];

    return out;
}

template<>
UnitBoundedPolynomial1<5> UnitBoundedPolynomial1<5>::fit(const Matrix<double, RequiredValues, 1> a, 
                                                         const Matrix<double, RequiredValues, 1> b)
{
    UnitBoundedPolynomial1<5> out;
    out.coeffs[0] = a[0];
    out.coeffs[1] = a[1];
    out.coeffs[2] = 0.5 * a[2];
    out.coeffs[3] = -10 * a[0] - 6 * a[1] - 1.5 * a[2] + 10 * b[0] - 4 * b[1] + 0.5 * b[2];
    out.coeffs[4] =  15 * a[0] + 8 * a[1] + 1.5 * a[2] - 15 * b[0] + 7 * b[1] - 1.0 * b[2];
    out.coeffs[5] = - 6 * a[0] - 3 * a[1] - 0.5 * a[2] +  6 * b[0] - 3 * b[1] + 0.5 * b[2];

    return out;
}

template<unsigned int Order, unsigned int Dims>
typename UnitBoundedPolynomial<Order, Dims>::VectorNd UnitBoundedPolynomial<Order, Dims>::interpolate(double t) const
{
    VectorNd p;

    for (int k = 0; k < Dims; ++k)
    {
        p[k] = _dims[k].interpolate(t);
    }

    return p;
}

template<unsigned int Order, unsigned int Dims>
typename UnitBoundedPolynomial<Order, Dims>::VectorNd UnitBoundedPolynomial<Order, Dims>::derivative(double t) const
{
    VectorNd d;

    for (int k = 0; k < Dims; ++k)
    {
        d[k] = _dims[k].derivative(t);
    }

    return d;
}

template<unsigned int Order, unsigned int Dims>
void UnitBoundedPolynomial<Order, Dims>::walk(const double deltatau, void (*fn)(VectorNd, double, const UnitBoundedPolynomial<Order, Dims>&, void *), void *payload, double a, double b) const
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

template<unsigned int Order, unsigned int Dims>
UnitBoundedPolynomial<Order, Dims> UnitBoundedPolynomial<Order, Dims>::fit(Matrix<double, Dims, Poly1::RequiredValues> A, 
                                                                           Matrix<double, Dims, Poly1::RequiredValues> B)
{
    UnitBoundedPolynomial<Order, Dims> out;

    for (int k = 0; k < Dims; ++k)
    {
        out._dims[k] = Poly1::fit(A.row(k).transpose(), B.row(k).transpose());
    }

    out.calculateLength();

    return out;
}

template<unsigned int Order, unsigned int Dims>
void UnitBoundedPolynomial<Order, Dims>::calculateLength()
{
    // N-point Gauss-Quadrature for solving length integral
    length = 0.0;

    for (auto &c : GAUSS_LENGENDRE_COEFFICIENTS)
    {
        length += derivative(c(1)).norm() * c(0);
    }

    length *= 0.5;
}

template<unsigned int Order, unsigned int Dims>
typename HermiteSpline<Order, Dims>::VectorNd HermiteSpline<Order, Dims>::interpolate(double s)
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

template<unsigned int Order, unsigned int Dims>
void HermiteSpline<Order, Dims>::walk(const double deltatau, void (*fn)(VectorNd, HermiteSpline<Order, Dims>&, double, UnitBoundedPolynomial<Order, Dims>&, void *), void *payload)
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

template<unsigned int Order, unsigned int Dims>
void HermiteSpline<Order, Dims>::add(UnitBoundedPolynomial<Order, Dims> sp)
{
    children.push_back(sp);
    length += sp.length;
}

template<unsigned int Order, unsigned int Dims>
HermiteSpline<Order, Dims> HermiteSpline<Order, Dims>::fit(array<MatrixNXd, Polynomial1::RequiredValues> values)
{
    const int N = values[0].cols()-1; // number of splines
    HermiteSpline<Order, Dims> result;

    Matrix<double, Dims, Polynomial1::RequiredValues, RowMajor> A;
    Matrix<double, Dims, Polynomial1::RequiredValues, RowMajor> B;

    for (int k = -1; k < N; ++k)
    {
        for (int j = 0; j < Polynomial1::RequiredValues; ++j)
        {
            B.col(j) = values[j].col(k+1);
        }

        if (k >= 0)
        {
            result.add(UnitBoundedPolynomial<Order, Dims>::fit(A, B));
        }

        A = B;
    }

    return result;
}

#define last N-1

template<unsigned int Order, unsigned int Dims>
HermiteSpline<Order, Dims> BaseSplineSolver<Order, Dims>::solve(vector<VectorNd> points, 
                                                                Matrix<double, Dims, Polynomial1::RequiredValues - 1> start, 
                                                                Matrix<double, Dims, Polynomial1::RequiredValues - 1> end)
{
    const int N = points.size();
    // one polynomial per dimension, we go over each dimension one by one
    // values[0] ^= point value
    // values[1] ^= 1st derivative
    // values[2] ^= 2nd derivative
    // ...
    array<MatrixNXd, Polynomial1::RequiredValues> values;

    // copy values into temp matrix
    for (int j = 0; j < Polynomial1::RequiredValues; ++j)
    {
        values[j] = MatrixNXd(Dims, N);

        if (j > 0)
        {
            values[j].col(0) = start.col(j-1);
            values[j].col(last) = end.col(j-1);
        }
    }

    // set values
    for (int j = 0; j < N; ++j)
    {
        values[0].col(j) = points[j];
    }

    RowXpr *params = (RowXpr *)malloc(Polynomial1::RequiredValues * sizeof(RowXpr));

    for (int k = 0; k < Dims; ++k)
    {
        for (int j = 0; j < Polynomial1::RequiredValues; ++j)
        {
            RowXpr row = values[j].row(k);
            memmove(&params[j], &row, sizeof(RowXpr));
        }

        if (!find_params_1d(params))
        {
            return HermiteSpline<Order, Dims>();
        }
    }

    free(params);

    return HermiteSpline<Order, Dims>::fit(values);
}

#define last N-1

template<unsigned int Dims>
bool SplineSolver<5, Dims>::find_params_1d(typename SplineSolver<5, Dims>::RowXpr params[3])
{
    RowXpr &q = params[0];
    RowXpr &v = params[1];
    RowXpr &a = params[2];

    assert(q.cols() > 2);
    const int N = q.cols();
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
    if (A.cols() != M)
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
bool SplineSolver<5, Dims>::build_solver(const int N, const int M)
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

#define INSTANTIATE_SPLINE(Order, Dims) \
template class UnitBoundedPolynomial<Order, Dims>;\
template class HermiteSpline<Order, Dims>;\
template class BaseSplineSolver<Order, Dims>;\
template class SplineSolver<Order, Dims>;


INSTANTIATE_SPLINE(5, 1)
INSTANTIATE_SPLINE(5, 2)
INSTANTIATE_SPLINE(5, 3)