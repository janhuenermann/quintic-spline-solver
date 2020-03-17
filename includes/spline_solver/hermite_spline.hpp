#ifndef SPLINE_HPP
#define SPLINE_HPP

#include <array>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>

using namespace std;
using namespace Eigen;

/**
 * A 1D polynomial of type f(t) = a + b t + c t^2 + d t^3 + ... + z t^(order)
 */
template<unsigned int Order>
struct UnitBoundedPolynomial1
{
    static_assert(Order % 2 == 1, "Order must be odd");
    static_assert(Order > 0);

    /**
     * The number of derivates (including 0-th derivative) 
     * required to fit this polynomial to start and end point
     */
    static constexpr const int RequiredValues = (Order + 1) / 2;

    double coeffs[Order+1];

    /**
     * Returns point at t on polynomial
     * @param  t in [0;1]
     * @return   double
     */
    double interpolate(double t) const;

    /**
     * Returns first derivative of polynomial at t
     * @param  t in [0;1]
     * @return   double
     */
    double derivative(double t) const;

    /**
     * Returns second derivative of polynomial at t
     * @param  t in [0;1]
     * @return   double
     */
    double derivative2(double t) const;

    /**
     * Fits the polynomial to go through start and end point
     * with specified derivatives.
     */
    static UnitBoundedPolynomial1<Order> fit(const Matrix<double, RequiredValues, 1> a, const Matrix<double, RequiredValues, 1> b);
};

template<unsigned int Order, unsigned int Dims>
struct UnitBoundedPolynomial
{
    static_assert(Order > 0);
    static_assert(Dims > 0);

    typedef Matrix<double, Dims, 1> VectorNd;
    typedef UnitBoundedPolynomial1<Order> Poly1;

    UnitBoundedPolynomial1<Order> _dims[Dims];
    double length;

    VectorNd interpolate(double t) const;
    VectorNd derivative(double t) const;

    /**
     * Walks on the polynomial. Calls fn many times with points on the function.
     * Resolution is specified using deltatau.
     * @param deltatau  Resolution with which to walk 
     * @param fn        Callback function
     * @param payload   Gets passed to callback
     * @param a         Optional, start point, can be used for extrapolation
     * @param b         Optional, end point, can be used for extrapolation
     */
    void walk(const double deltatau, void (*fn)(VectorNd, double, const UnitBoundedPolynomial<Order, Dims>&, void *), void *payload = nullptr, double a = 0.0, double b = 1.0) const;

    /**
     * Calculates a spline.
     * @param  A Matrix of point a with A[:, 0] = point, A[:, 1] = 1st derivative, A[:, 2] = 2nd derivative
     * @param  B Matrix of point b with B[:, 0] = point, B[:, 1] = 1st derivative, B[:, 2] = 2nd derivative
     * @return   Spline
     */
    static UnitBoundedPolynomial<Order, Dims> fit(Matrix<double, Dims, Poly1::RequiredValues> A, 
                                                  Matrix<double, Dims, Poly1::RequiredValues> B);

private:
    void calculateLength();
};

template<unsigned int Order, unsigned int Dims>
class SplineSolver;

/**
 * A hermite spline. 
 * Order and dimensionality can vary. 
 * Currently only quintic and cubic splines are supported.
 */
template<unsigned int Order, unsigned int Dims>
struct HermiteSpline
{
    static_assert(Order > 0);
    static_assert(Dims > 0);

    using Solver = SplineSolver<Order, Dims>;

    typedef Matrix<double, Dims, 1> VectorNd;
    typedef Matrix<double, Dims, Dynamic> MatrixNXd;

    typedef UnitBoundedPolynomial1<Order> Polynomial1;
    typedef UnitBoundedPolynomial<Order, Dims> Polynomial;

    vector<Polynomial> children;
    double length;

    HermiteSpline() : length(0)
    {
    }

    VectorNd interpolate(double s);

    void walk(const double deltatau, void (*fn)(VectorNd, HermiteSpline<Order, Dims>&, double, Polynomial&, void *), void *payload = nullptr);
    void add(Polynomial sp);

    /**
     * Calculates the hermite spline.
     * @param values    An array of required values (i.e. point, derivative, 2nd derivative, ..) x dimensions x number of points.
     * @return HermiteSpline
     */
    static HermiteSpline<Order, Dims> fit(array<MatrixNXd, Polynomial1::RequiredValues> values);
};

template<unsigned int Order, unsigned int Dims>
class BaseSplineSolver
{
public:

    typedef typename Eigen::Matrix<double, Dims, 1> VectorNd;
    typedef typename Eigen::Matrix<double, Dims, Dynamic> MatrixNXd;
    typedef typename MatrixNXd::RowXpr RowXpr;

    typedef UnitBoundedPolynomial1<Order> Polynomial1;
    typedef UnitBoundedPolynomial<Order, Dims> Polynomial;

    typedef HermiteSpline<Order, Dims> Spline;

    HermiteSpline<Order, Dims> solve(vector<VectorNd> points, 
                                     Matrix<double, Dims, Polynomial1::RequiredValues - 1> start, 
                                     Matrix<double, Dims, Polynomial1::RequiredValues - 1> end);

protected:

    virtual bool find_params_1d(RowXpr params[Polynomial1::RequiredValues]) = 0;

};

template<unsigned int Order, unsigned int Dims>
class SplineSolver : public BaseSplineSolver<Order, Dims>
{};


// Quintic solver

template<unsigned int Dims>
class SplineSolver<5, Dims> : public BaseSplineSolver<5, Dims>
{
public:

    using RowXpr = typename BaseSplineSolver<5, Dims>::RowXpr;

protected:

    SparseMatrix<double> A;
    SparseLU<SparseMatrix<double>,  COLAMDOrdering<int>> solver;

    bool find_params_1d(RowXpr params[3]);
    bool build_solver(const int N, const int M);

};

// Cubic solver

template<unsigned int Dims>
class SplineSolver<3, Dims> : public BaseSplineSolver<3, Dims>
{
public:

    using RowXpr = typename BaseSplineSolver<3, Dims>::RowXpr;

protected:

    SparseMatrix<double> A;
    SparseLU<SparseMatrix<double>,  COLAMDOrdering<int>> solver;

    bool find_params_1d(RowXpr params[2]);
    bool build_solver(const int N, const int M);

};

template <unsigned int Dims>
using CubicHermiteSpline = HermiteSpline<3, Dims>;

template <unsigned int Dims>
using QuinticHermiteSpline = HermiteSpline<5, Dims>;

#endif
