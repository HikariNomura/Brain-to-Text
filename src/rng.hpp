#pragma once
// Reimplementation of numpy's `default_rng(0)` PCG64 bit generator and its
// `.permutation(n)` (Fisher-Yates shuffle) algorithm, hardcoded for seed=0
// (the only seed the original Python script ever uses:
// `np.random.default_rng(0)  # 0 is seed`).
//
// Verified bit-for-bit against numpy 2.5.1:
//   np.random.default_rng(0).permutation(np.arange(100))
// reproduces exactly the same sequence as permutation(100) below.
//
// If a different seed is ever needed, numpy's generic `SeedSequence` entropy
// mixing algorithm would need to be implemented too; that is not done here
// since the original script always uses seed 0.
#include <cstdint>
#include <vector>
#include <algorithm>

namespace mtrf {

using u128 = unsigned __int128;

inline u128 mk128(uint64_t hi, uint64_t lo) {
    return (u128(hi) << 64) | u128(lo);
}
inline uint64_t rotr64(uint64_t v, unsigned rot) {
    rot &= 63;
    return (v >> rot) | (v << ((64 - rot) & 63));
}

class PCG64Seed0 {
public:
    PCG64Seed0() {
        // state/inc for numpy's PCG64(0), captured from
        // np.random.PCG64(0).state immediately after construction.
        state_ = mk128(0x1aa1b5345996452dULL, 0x09585eb7a69561e3ULL);
        inc_ = mk128(0x418ddadb3af71a82ULL, 0x588133bc447873a9ULL);
        mult_ = mk128(0x2360ed051fc65da4ULL, 0x4385df649fccf645ULL);
    }

    uint64_t next_uint64() {
        state_ = state_ * mult_ + inc_;
        uint64_t hi = (uint64_t)(state_ >> 64);
        uint64_t lo = (uint64_t)(state_);
        unsigned rot = (unsigned)(state_ >> 122);
        return rotr64(hi ^ lo, rot);
    }

    uint32_t next_uint32() {
        if (!has_uint32_) {
            uint64_t next = next_uint64();
            has_uint32_ = true;
            uinteger_ = (uint32_t)(next >> 32);
            return (uint32_t)(next & 0xffffffffu);
        } else {
            has_uint32_ = false;
            return uinteger_;
        }
    }

private:
    u128 state_, inc_, mult_;
    bool has_uint32_ = false;
    uint32_t uinteger_ = 0;
};

// numpy's `random_interval`: uniform integer in [0, max] via mask-and-reject.
inline uint64_t random_interval(PCG64Seed0& bg, uint64_t max) {
    if (max == 0) return 0;
    uint64_t mask = max;
    mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4;
    mask |= mask >> 8; mask |= mask >> 16; mask |= mask >> 32;
    uint64_t value;
    if (max <= 0xffffffffULL) {
        do { value = bg.next_uint32() & mask; } while (value > max);
    } else {
        do { value = bg.next_uint64() & mask; } while (value > max);
    }
    return value;
}

// Equivalent to: np.random.default_rng(0).permutation(np.arange(n))
inline std::vector<int> permutation_seed0(int n) {
    PCG64Seed0 bg;
    std::vector<int> a(n);
    for (int i = 0; i < n; i++) a[i] = i;
    for (int i = n - 1; i >= 1; i--) {
        uint64_t j = random_interval(bg, (uint64_t)i);
        std::swap(a[i], a[(size_t)j]);
    }
    return a;
}

// Equivalent to: np.array_split(shuffled, k)
inline std::vector<std::vector<int>> array_split(const std::vector<int>& a, int k) {
    int n = (int)a.size();
    int base = n / k, extra = n % k;
    std::vector<std::vector<int>> groups;
    groups.reserve(k);
    int pos = 0;
    for (int g = 0; g < k; g++) {
        int sz = base + (g < extra ? 1 : 0);
        groups.emplace_back(a.begin() + pos, a.begin() + pos + sz);
        pos += sz;
    }
    return groups;
}

} // namespace mtrf
