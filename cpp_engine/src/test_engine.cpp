#include "../include/shard_format.hpp"
#include "../include/ort_engine.hpp"
#include "../include/geometry.hpp"

#include <iostream>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "Running C++ Engine Structural & Numerical Invariant Tests...\n";

    // 1. Binary layout invariants
    static_assert(sizeof(FrameRecord) == 2588, "FrameRecord size must be 2588 bytes");
    std::cout << "  [PASS] FrameRecord binary struct layout static asserts verified.\n";

    tactical::PitchInferenceState state;
    assert(reinterpret_cast<uintptr_t>(state.node_buf.data()) % 32 == 0);
    assert(reinterpret_cast<uintptr_t>(state.adj_buf.data()) % 32 == 0);
    std::cout << "  [PASS] alignas(32) AVX2 stack alignment verified.\n";

    // 2. Geometry C++ Engine Tests
    std::vector<tactical::Point2D> tight_def = {
        {40.0f, 35.0f}, {40.0f, 45.0f}, {50.0f, 35.0f}, {50.0f, 45.0f}
    };
    float hull_area = tactical::compute_convex_hull_area(tight_def);
    assert(std::abs(hull_area - 100.0f) < 1e-3f);
    std::cout << "  [PASS] Andrew's Monotone Chain 2D Convex Hull area (100 m^2) verified.\n";

    float c1 = tactical::compute_c1_line_height(tight_def);
    assert(c1 >= 0.0f && c1 <= 1.0f);
    std::cout << "  [PASS] C1 Line Height verified.\n";

    float c2 = tactical::compute_c2_compactness(tight_def, false);
    assert(c2 > 0.85f && c2 <= 1.0f);
    std::cout << "  [PASS] C2 Compactness verified.\n";

    tactical::Point2D ball{55.0f, 40.0f};
    float c4 = tactical::compute_c4_pressure(tight_def, ball);
    assert(c4 > 0.0f && c4 <= 1.0f);
    std::cout << "  [PASS] C4 Gegenpress Pressure verified.\n";

    // 3. Stage 5 Concept Intervention Test
    std::array<float, 6> weights = {-1.2f, -0.15f, -0.5f, -0.3f, -0.2f, 2.0f};
    float bias = 0.3823f;
    std::array<float, 6> c_safe = {0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.1f};
    std::array<float, 6> c_danger = {0.2f, 0.2f, 0.2f, 0.1f, 0.1f, 0.8f};

    float threat_safe = tactical::evaluate_concept_intervention(c_safe, weights, bias);
    float threat_danger = tactical::evaluate_concept_intervention(c_danger, weights, bias);
    assert(threat_safe < threat_danger);
    std::cout << "  [PASS] Concept sensitivity & monotonicity verified (Overload increases danger).\n";

    std::cout << "\nALL C++ INVARIANTS AND TESTS PASSED SUCCESSFULLY!\n";
    return 0;
}
