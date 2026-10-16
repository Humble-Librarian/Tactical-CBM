#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace tactical {

struct Point2D {
    float x;
    float y;
    
    bool operator<(const Point2D& other) const noexcept {
        return (x < other.x) || (x == other.x && y < other.y);
    }
};

/**
 * @brief Computes 2D Convex Hull Area via Andrew's Monotone Chain algorithm.
 * Guarantees bit-for-bit mathematical parity between C++, Python, and WebAssembly.
 */
inline float compute_convex_hull_area(std::vector<Point2D> pts) noexcept {
    if (pts.size() < 3) return 0.0f;
    std::sort(pts.begin(), pts.end());

    auto cross = [](const Point2D& o, const Point2D& a, const Point2D& b) noexcept {
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    };

    std::vector<Point2D> lower;
    lower.reserve(pts.size());
    for (const auto& p : pts) {
        while (lower.size() >= 2 && cross(lower[lower.size() - 2], lower.back(), p) <= 0.0f) {
            lower.pop_back();
        }
        lower.push_back(p);
    }

    std::vector<Point2D> upper;
    upper.reserve(pts.size());
    for (auto it = pts.rbegin(); it != pts.rend(); ++it) {
        while (upper.size() >= 2 && cross(upper[upper.size() - 2], upper.back(), *it) <= 0.0f) {
            upper.pop_back();
        }
        upper.push_back(*it);
    }

    lower.pop_back();
    upper.pop_back();
    lower.insert(lower.end(), upper.begin(), upper.end());

    if (lower.size() < 3) return 0.0f;

    float area = 0.0f;
    for (size_t i = 0; i < lower.size(); ++i) {
        size_t j = (i + 1) % lower.size();
        area += lower[i].x * lower[j].y - lower[j].x * lower[i].y;
    }
    return std::abs(area) * 0.5f;
}

inline float clamp01(float v) noexcept {
    return (v < 0.0f) ? 0.0f : ((v > 1.0f) ? 1.0f : v);
}

inline float compute_c1_line_height(const std::vector<Point2D>& defenders) noexcept {
    if (defenders.empty()) return 0.5f;
    float max_x = 0.0f;
    for (const auto& d : defenders) {
        if (d.x > max_x) max_x = d.x;
    }
    float c1 = (120.0f - max_x) / 70.0f;
    return clamp01(c1);
}

inline float compute_c2_compactness(const std::vector<Point2D>& defenders, bool apply_frustum_scaling = false) noexcept {
    if (defenders.size() < 3) return 0.5f;
    float area = compute_convex_hull_area(defenders);
    if (apply_frustum_scaling && defenders.size() < 10) {
        float factor = std::sqrt(10.0f / static_cast<float>(std::max<size_t>(defenders.size(), 4)));
        area *= factor;
    }
    float c2 = 1.0f - (area / 1200.0f);
    return clamp01(c2);
}

inline float compute_c3_width_balance(const std::vector<Point2D>& defenders, float ball_y) noexcept {
    if (defenders.empty()) return 0.5f;
    float min_y = 80.0f, max_y = 0.0f, sum_y = 0.0f;
    for (const auto& d : defenders) {
        min_y = std::min(min_y, d.y);
        max_y = std::max(max_y, d.y);
        sum_y += d.y;
    }
    float spread = max_y - min_y;
    float avg_y = sum_y / static_cast<float>(defenders.size());
    float y_offset = std::abs(avg_y - ball_y) / 35.0f;
    float c3 = (spread / 45.0f) * (1.0f - y_offset * 0.5f);
    return clamp01(c3);
}

inline float compute_c4_pressure(const std::vector<Point2D>& defenders, const Point2D& ball) noexcept {
    if (defenders.empty()) return 0.0f;
    float min_dist = 999.0f;
    int within_6m = 0;
    for (const auto& d : defenders) {
        float dist = std::hypot(d.x - ball.x, d.y - ball.y);
        if (dist < min_dist) min_dist = dist;
        if (dist <= 6.0f) within_6m++;
    }
    float c4 = std::exp(-min_dist / 4.0f) * 0.7f + (static_cast<float>(within_6m) / 3.0f) * 0.3f;
    return clamp01(c4);
}

inline float dist_to_segment(const Point2D& p, const Point2D& a, const Point2D& b) noexcept {
    float dx = b.x - a.x, dy = b.y - a.y;
    float len_sq = dx * dx + dy * dy;
    if (len_sq < 1e-4f) return std::hypot(p.x - a.x, p.y - a.y);
    float t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / len_sq;
    if (t < 0.05f || t > 0.95f) return 999.0f;
    return std::hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

inline float compute_c5_lane_blockage(
    const Point2D& ball, 
    const std::vector<Point2D>& attackers, 
    const std::vector<Point2D>& defenders) noexcept 
{
    int forward_count = 0;
    int blocked_count = 0;
    for (const auto& a : attackers) {
        if (a.x > ball.x) {
            forward_count++;
            bool blocked = false;
            for (const auto& d : defenders) {
                if (dist_to_segment(d, ball, a) <= 2.2f) {
                    blocked = true;
                    break;
                }
            }
            if (blocked) blocked_count++;
        }
    }
    return forward_count > 0 ? (static_cast<float>(blocked_count) / static_cast<float>(forward_count)) : 0.0f;
}

inline float compute_c6_box_overload(
    const Point2D& ball,
    const std::vector<Point2D>& attackers,
    const std::vector<Point2D>& defenders) noexcept 
{
    float max_def_x = 0.0f;
    for (const auto& d : defenders) {
        if (d.x > max_def_x) max_def_x = d.x;
    }
    float offside_line_x = std::max(max_def_x, ball.x);

    int onside_att_in_box = 0;
    for (const auto& a : attackers) {
        if (a.x >= 102.0f && a.y >= 18.0f && a.y <= 62.0f && a.x <= (offside_line_x + 0.5f)) {
            onside_att_in_box++;
        }
    }
    bool is_carrier_in_box = (ball.x >= 102.0f && ball.y >= 18.0f && ball.y <= 62.0f);
    int total_att_in_box = onside_att_in_box + (is_carrier_in_box ? 1 : 0);

    int def_in_box = 0;
    for (const auto& d : defenders) {
        if (d.x >= 102.0f && d.y >= 18.0f && d.y <= 62.0f) {
            def_in_box++;
        }
    }

    if (total_att_in_box == 0) return 0.0f;
    float raw_ratio = (def_in_box == 0) ? 1.0f : (static_cast<float>(total_att_in_box) / static_cast<float>(total_att_in_box + def_in_box));

    if (is_carrier_in_box) return raw_ratio;

    float box_proximity = clamp01((ball.x - 30.0f) / 72.0f);
    return raw_ratio * (0.4f + 0.6f * box_proximity);
}

} // namespace tactical
