#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>

#include <opencv2/opencv.hpp>

#include "constants.hpp"

using namespace std;
using namespace Eigen;
using namespace cv;

struct Spline1
{
    double coeffs[6];

    double interpolate(double t) const; 
    double derivative(double t) const;

    static Spline1 calculate(const double *a, const double *b);
};


/**
 * A quintic hermite spline
 */
template<unsigned int Dims>
struct Spline
{
    typedef Matrix<double, Dims, 1> VectorNd;

    Spline1 _dims[Dims];
    double length;

    VectorNd interpolate(double t) const;
    VectorNd derivative(double t) const;

    /**
     * Calculates a spline.
     * @param  A Matrix of point a with A[:, 0] = point, A[:, 1] = 1st derivative, A[:, 2] = 2nd derivative
     * @param  B Matrix of point b with B[:, 0] = point, B[:, 1] = 1st derivative, B[:, 2] = 2nd derivative
     * @return   Spline
     */
    static Spline calculate(Matrix<double, Dims, 3, RowMajor> A, Matrix<double, Dims, 3, RowMajor> B);

private:
    void calculateLength();
};

template<unsigned int Dims>
struct SplinePath
{
    typedef Matrix<double, Dims, 1> VectorNd;
    typedef Matrix<double, Dims, Dynamic> MatrixNXd;

    vector<Spline<Dims>> children;
    double length;

    SplinePath() : length(0)
    {}

    VectorNd interpolate(double s);

    void walk(const double deltatau, void (*fn)(VectorNd, SplinePath<Dims>&, double, Spline<Dims>&));
    void draw(Mat3f img, Vector2d scale);
    void add(Spline<Dims> sp);

    static SplinePath<Dims> calculate(MatrixNXd q, MatrixNXd v, MatrixNXd a);
};

#define last N-1

template<unsigned int Dims>
class SplineSolver
{    
public:

    typedef Matrix<double, Dims, 1> VectorNd;

    SplinePath<Dims> solve(vector<VectorNd> points, VectorNd vel_start, VectorNd vel_end, VectorNd accel_start, VectorNd accel_end);

private:
    int max_n = 64;
    SparseMatrix<double> A;
    SparseQR<SparseMatrix<double>,  Eigen::COLAMDOrdering<int>> solver;

    bool find_params_1d(VectorXd q, VectorXd &v, VectorXd &a);
    bool find_params_1d_n(VectorXd q, VectorXd &v, VectorXd &a);
    bool build_solver(const int N, const int M);

};