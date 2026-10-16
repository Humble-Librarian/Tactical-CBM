#include "../include/geometry.hpp"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

extern "C" {

/**
 * @brief Evaluates all 6 tactical concepts (C1 - C6) directly via WebAssembly.
 * 
 * @param def_coords Pointer to array of [x, y] defender coordinates
 * @param num_defenders Number of defenders
 * @param att_coords Pointer to array of [x, y] attacker coordinates
 * @param num_attackers Number of attackers
 * @param ball_x Ball X coordinate
 * @param ball_y Ball Y coordinate
 * @param out_concepts Output pointer for 6 float concepts [c1, c2, c3, c4, c5, c6]
 */
EMSCRIPTEN_KEEPALIVE
void compute_tactical_concepts_wasm(
    const float* def_coords, int num_defenders,
    const float* att_coords, int num_attackers,
    float ball_x, float ball_y,
    float* out_concepts
) {
    std::vector<tactical::Point2D> defenders;
    defenders.reserve(num_defenders);
    for (int i = 0; i < num_defenders; ++i) {
        defenders.push_back({def_coords[i * 2 + 0], def_coords[i * 2 + 1]});
    }

    std::vector<tactical::Point2D> attackers;
    attackers.reserve(num_attackers);
    for (int i = 0; i < num_attackers; ++i) {
        attackers.push_back({att_coords[i * 2 + 0], att_coords[i * 2 + 1]});
    }

    tactical::Point2D ball{ball_x, ball_y};

    out_concepts[0] = tactical::compute_c1_line_height(defenders);
    out_concepts[1] = tactical::compute_c2_compactness(defenders, false);
    out_concepts[2] = tactical::compute_c3_width_balance(defenders, ball_y);
    out_concepts[3] = tactical::compute_c4_pressure(defenders, ball);
    out_concepts[4] = tactical::compute_c5_lane_blockage(ball, attackers, defenders);
    out_concepts[5] = tactical::compute_c6_box_overload(ball, attackers, defenders);
}

}
