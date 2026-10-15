#pragma once

#include <array>
#include <vector>
#include <cmath>
#include <string>
#include <memory>
#include <iostream>
#include <chrono>

/**
 * @brief Zero-Copy Tactical CBM C++ Engine Specification
 * 
 * Matches Slide 4 and Slide 5 stack allocation architecture:
 * - alignas(32) stack buffers for AVX2 auto-vectorization
 * - Zero dynamic heap allocation in inference hot path
 * - Instant Stage 5 concept intervention in 0.002 ms (2 microseconds)
 */
namespace tactical {

constexpr size_t MAX_PLAYERS = 22;
constexpr size_t NODE_FEAT_DIM = 6;
constexpr size_t NUM_CONCEPTS = 6;
constexpr size_t ADJ_SIZE = MAX_PLAYERS * MAX_PLAYERS; // 484
constexpr size_t NODE_BUF_SIZE = MAX_PLAYERS * NODE_FEAT_DIM; // 132

struct PitchInferenceState {
    alignas(32) std::array<float, NODE_BUF_SIZE> node_buf{};
    alignas(32) std::array<float, ADJ_SIZE> adj_buf{};
    alignas(32) std::array<float, MAX_PLAYERS> player_mask{};
    
    // Outputs
    alignas(32) std::array<float, NUM_CONCEPTS> concepts{};
    float threat_logit{0.0f};
    float threat_prob{0.0f};
};

/**
 * @brief Stage 5: Instant Concept Intervention (0.002 ms / 2 microseconds)
 * 
 * Recomputes threat score from user slider tweaks in O(1) time
 * using direct vector dot product without touching the neural network.
 */
inline float evaluate_concept_intervention(
    const std::array<float, NUM_CONCEPTS>& concepts,
    const std::array<float, NUM_CONCEPTS>& weights,
    float bias) noexcept 
{
    float logit = bias;
    for (size_t i = 0; i < NUM_CONCEPTS; ++i) {
        logit += concepts[i] * weights[i];
    }
    return 1.0f / (1.0f + std::exp(-logit));
}

} // namespace tactical
