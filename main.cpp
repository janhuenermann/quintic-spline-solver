#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <chrono>

#include <spline_solver/spline.hpp>
#include <spline_solver/draw.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;

Mat3f frame(1080, 1920);
Mat3f base(1080, 1920);

#define WINDOW1 "w1"

vector<Vector2d> points;
SplineSolver<2> solver;
SplinePath<2> path;
chrono::time_point<std::chrono::steady_clock> last_click_time;

SplinePath<2> fit_spline()
{
    if (points.size() < 3)
    {
        return SplinePath<2>();
    }

    return solver.solve(points, Vector2d(0,0), Vector2d(0,0), Vector2d(0,0), Vector2d(0,0));
}

void click_callback(int event, int x, int y, int flags, void* userdata)
{
    if (flags != (EVENT_FLAG_CTRLKEY + EVENT_FLAG_LBUTTON))
    {
        return ;
    }

    Vector2i click_point(x, y);
    std::chrono::duration<double> diff = std::chrono::steady_clock::now() - last_click_time;

    // Debounce
    if (diff.count() < 0.02)
    {
        return ;
    }

    last_click_time = std::chrono::steady_clock::now();
    base.setTo(Scalar(0.0f,0.0f, 0.0f)); // clear frame

    cout << "Left mouse button is clicked while pressing CTRL key - position (" << x << ", " << y << ")" << endl;

    double scale = (double)max(frame.cols, frame.rows);
    points.push_back(click_point.cast<double>() / scale);

    path = fit_spline();
    draw_spline(path, base, Vector2d(scale, scale));

    for (int k = 0; k < points.size(); ++k)
    {
        circle(base, Point((int)(points[k].x() * scale), (int)(points[k].y() * scale)), 5.0, Scalar(0.0f, 1.0f, 0.0f), FILLED);
    }

    base.copyTo(frame);
    imshow(WINDOW1, frame);
}

int main(int argc, char **argv)
{
    namedWindow(WINDOW1, 1);
    setMouseCallback(WINDOW1, click_callback, NULL);

    cout << "Click on a spot with the left mouse button and hold down CTRL to add a point to the spline." << endl;

    imshow(WINDOW1, frame);
    waitKey(0);

    return 0;
}
