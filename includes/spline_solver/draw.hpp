#ifndef DRAW_SPLINE_HPP
#define DRAW_SPLINE_HPP

#include <opencv2/core.hpp>

#include <spline_solver/spline.hpp>

using namespace cv;

// helper to store img and scale
struct _drawSplinePayload { Mat3f &img; Vector2d &scale; };

// gets called for every point
inline void draw_spline_callback(Vector2d pt, SplinePath<2>& path, double tau, Spline<2>& sp, void *payload)
{
    _drawSplinePayload *payl = reinterpret_cast<_drawSplinePayload *>(payload);

    pt = pt.cwiseProduct(payl->scale);

    circle(payl->img, Point(pt.x(), pt.y()), 4.0, Scalar(0.0f, 0.0f, 1.0f), FILLED);
}

/**
 * Draws the spline onto the given image with given scale.
 * @param path  Spline
 * @param img   Image
 * @param scale Resizes spline by given factor
 */
inline void draw_spline(SplinePath<2> path, Mat3f img, Vector2d scale = Vector2d(1,1))
{
    const Vector2d size(img.cols, img.rows);
    const double deltatau = 1.0 / scale.norm(); // 1.0 / hypot((double)img.cols, (double)img.rows);

    _drawSplinePayload payl = { img, scale };
    path.walk(deltatau, &draw_spline_callback, reinterpret_cast<void *>(&payl));
}

#endif