#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <chrono>

#include <spline_solver/hermite_spline.hpp>
#include <spline_solver/draw.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;

Mat3f frame(1080, 1920);
const double scale = (double)max(frame.cols, frame.rows);

#define WINDOW1 "w1"


vector<Vector2d> points;

Vector2i last_click_point;
chrono::time_point<std::chrono::steady_clock> last_click_time;

template<typename Spline>
Spline fit_and_draw_spline()
{
    Spline sp;

    frame.setTo(Scalar(0.0f,0.0f, 0.0f)); // clear frame

    if (points.size() >= 3)
    {
        typename Spline::Solver solver;

        Matrix<double, 2, Spline::Polynomial1::RequiredValues - 1> start;
        Matrix<double, 2, Spline::Polynomial1::RequiredValues - 1> end;

        start.setZero();
        end.setZero();

        sp = solver.solve(points, start, end);
        draw_spline(sp, frame, Vector2d(scale, scale));
    }

    for (int k = 0; k < points.size(); ++k)
    {
        circle(frame, Point((int)(points[k].x() * scale), (int)(points[k].y() * scale)), 5.0, Scalar(0.0f, 1.0f, 0.0f), FILLED);
    }

    imshow(WINDOW1, frame);

    return sp;
}

void click_callback(int event, int x, int y, int flags, void* userdata)
{
    if (flags != (EVENT_FLAG_CTRLKEY | EVENT_FLAG_LBUTTON))
    {
        return ;
    }

    Vector2i click_point(x, y);
    std::chrono::duration<double> diff = std::chrono::steady_clock::now() - last_click_time;

    // Debounce
    if (last_click_point == click_point || diff.count() < 0.05)
    {
        return ;
    }

    cout << "Left mouse button is clicked while pressing CTRL key - position (" << x << ", " << y << ")" << endl;

    points.push_back(click_point.cast<double>() / scale);
    fit_and_draw_spline<QuinticHermiteSpline<2>>();

    last_click_time = std::chrono::steady_clock::now();
    last_click_point = click_point;
}

int main(int argc, char **argv)
{
    namedWindow(WINDOW1, 1);
    setMouseCallback(WINDOW1, click_callback, NULL);

    cout << "Click on a spot with the left mouse button and hold down CTRL to add a point to the spline." << endl;

    imshow(WINDOW1, frame);
    waitKey(0);

    fit_and_draw_spline<CubicHermiteSpline<2>>();
    waitKey(0);

    return 0;
}
