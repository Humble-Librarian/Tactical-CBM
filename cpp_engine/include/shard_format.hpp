#pragma once

#include <cstdint>
#include <cstddef>
#include <array>

#pragma pack(push, 1)

/**
 * @brief Binary Training Shard Frame Layout (Fixed 2,588 bytes per frame)
 * 
 * Guaranteed byte alignment for zero-copy memory mapping between C++ and Python.
 */
struct FrameRecord {
    int32_t match_id;                                     // 4 bytes  (offset 0)
    int32_t frame_id;                                     // 4 bytes  (offset 4)
    std::array<float, 22 * 6> node_features;              // 528 bytes (offset 8)
    std::array<float, 22 * 22> adj_mask;                  // 1936 bytes (offset 536)
    std::array<float, 22> player_mask;                    // 88 bytes (offset 2472)
    std::array<float, 6> c_true;                          // 24 bytes (offset 2560)
    float y_threat;                                       // 4 bytes  (offset 2584)
};                                                        // Total: 2588 bytes

#pragma pack(pop)

static_assert(sizeof(FrameRecord) == 2588, "FrameRecord size must exactly equal 2588 bytes");
static_assert(offsetof(FrameRecord, node_features) == 8, "node_features offset mismatch");
static_assert(offsetof(FrameRecord, adj_mask) == 536, "adj_mask offset mismatch");
static_assert(offsetof(FrameRecord, player_mask) == 2472, "player_mask offset mismatch");
static_assert(offsetof(FrameRecord, c_true) == 2560, "c_true offset mismatch");
static_assert(offsetof(FrameRecord, y_threat) == 2584, "y_threat offset mismatch");
