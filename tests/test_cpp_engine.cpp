#include "../cpp_engine/include/ort_engine.hpp"
#include "../cpp_engine/include/shard_format.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "Running C++ Engine Structural & Numerical Invariant Tests...\n";

    // 1. Assert FrameRecord exact size & offsets
    static_assert(sizeof(FrameRecord) == 2588, "FrameRecord size must be 2588 bytes");
    static_assert(offsetof(FrameRecord, match_id) == 0, "match_id offset must be 0");
    static_assert(offsetof(FrameRecord, frame_id) == 4, "frame_id offset must be 4");
    static_assert(offsetof(FrameRecord, node_features) == 8, "node_features offset must be 8");
    static_assert(offsetof(FrameRecord, adj_mask) == 536, "adj_mask offset must be 536");
    static_assert(offsetof(FrameRecord, player_mask) == 2472, "player_mask offset must be 2472");
    static_assert(offsetof(FrameRecord, c_true) == 2560, "c_true offset must be 2560");
    static_assert(offsetof(FrameRecord, y_threat) == 2584, "y_threat offset must be 2584");
    std::cout << "  [PASS] FrameRecord binary struct layout static asserts verified.\n";

    // 2. Assert Stack Alignment
    tactical::PitchInferenceState state;
    assert(reinterpret_cast<uintptr_t>(state.node_buf.data()) % 32 == 0);
    assert(reinterpret_cast<uintptr_t>(state.adj_buf.data()) % 32 == 0);
    assert(reinterpret_cast<uintptr_t>(state.player_mask.data()) % 32 == 0);
    assert(reinterpret_cast<uintptr_t>(state.concepts.data()) % 32 == 0);
    std::cout << "  [PASS] alignas(32) AVX2 stack alignment verified.\n";

    // 3. Test evaluate_concept_intervention accuracy
    std::array<float, 6> weights = {-1.2000f, -0.1500f, -0.5000f, -0.3000f, -0.2000f, 2.0000f};
    float bias = 0.3823f;

    // Zero vector test: sigmoid(bias)
    std::array<float, 6> zeros = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float threat_zero = tactical::evaluate_concept_intervention(zeros, weights, bias);
    float expected_zero = 1.0f / (1.0f + std::exp(-bias));
    assert(std::abs(threat_zero - expected_zero) < 1e-6f);
    std::cout << "  [PASS] Zero concept vector evaluation matches expected sigmoid.\n";

    // Monotonicity test 1: Increasing box overload (positive weight) MUST strictly increase threat
    std::array<float, 6> c_low = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.1f};
    std::array<float, 6> c_high = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.9f};
    float t_low = tactical::evaluate_concept_intervention(c_low, weights, bias);
    float t_high = tactical::evaluate_concept_intervention(c_high, weights, bias);
    assert(t_high > t_low);
    std::cout << "  [PASS] Concept sensitivity & monotonicity verified (Overload increases danger).\n";

    // Monotonicity test 2: Increasing pressure (negative weight) MUST strictly decrease threat
    std::array<float, 6> c_press_low = {0.5f, 0.5f, 0.5f, 0.1f, 0.5f, 0.5f};
    std::array<float, 6> c_press_high = {0.5f, 0.5f, 0.5f, 0.9f, 0.5f, 0.5f};
    float t_press_low = tactical::evaluate_concept_intervention(c_press_low, weights, bias);
    float t_press_high = tactical::evaluate_concept_intervention(c_press_high, weights, bias);
    assert(t_press_high < t_press_low);
    std::cout << "  [PASS] Concept sensitivity & monotonicity verified (Pressure reduces danger).\n";

    // Monotonicity test 3: Pushing high line (negative weight) MUST strictly decrease threat
    std::array<float, 6> c_line_low = {0.1f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    std::array<float, 6> c_line_high = {0.9f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    float t_line_low = tactical::evaluate_concept_intervention(c_line_low, weights, bias);
    float t_line_high = tactical::evaluate_concept_intervention(c_line_high, weights, bias);
    assert(t_line_high < t_line_low);
    std::cout << "  [PASS] Concept sensitivity & monotonicity verified (High Line suppresses threat).\n";

    // Monotonicity test 4: Lane blockage (negative weight) MUST strictly decrease threat
    std::array<float, 6> c_lane_low = {0.5f, 0.5f, 0.5f, 0.5f, 0.1f, 0.5f};
    std::array<float, 6> c_lane_high = {0.5f, 0.5f, 0.5f, 0.5f, 0.9f, 0.5f};
    float t_lane_low = tactical::evaluate_concept_intervention(c_lane_low, weights, bias);
    float t_lane_high = tactical::evaluate_concept_intervention(c_lane_high, weights, bias);
    assert(t_lane_high < t_lane_low);
    std::cout << "  [PASS] Concept sensitivity & monotonicity verified (Passing lanes cut reduces threat).\n";

    std::cout << "\nALL C++ INVARIANTS AND TESTS PASSED SUCCESSFULLY!\n";
    return 0;
}
