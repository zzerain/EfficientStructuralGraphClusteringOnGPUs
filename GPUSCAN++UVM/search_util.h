#pragma once

#include <cstdint>


template<typename T, typename B>
B LinearSearch(T* array, B offset_beg, B offset_end, T val) {
    // linear search fallback
    for (auto offset = offset_beg; offset < offset_end; offset++) {
        if (array[offset] >= val) {
            return offset;
        }
    }
    return offset_end;
}

template<typename T, typename B>
B BranchFreeBinarySearch(T* a, B offset_beg, B offset_end, T x) {
    T n = offset_end - offset_beg;
    //using I = uint32_t;
    const T* base = a + offset_beg;
    while (n > 1) {
        B half = n / 2;
        base = (base[half] < x) ? base + half : base;
        n -= half;
    }
    return (*base < x) + base - a;
}

// Assuming (offset_beg != offset_end)
template<typename T, typename B>
B GallopingSearch(T* array, B offset_beg, B offset_end, T val) {
    if (array[offset_end - 1] < val) {
        return offset_end;
    }
    // galloping
    if (array[offset_beg] >= val) {
        return offset_beg;
    }
    if (array[offset_beg + 1] >= val) {
        return offset_beg + 1;
    }
    if (array[offset_beg + 2] >= val) {
        return offset_beg + 2;
    }

    B jump_idx = 4u;
    while (true) {
        B peek_idx = offset_beg + jump_idx;
        if (peek_idx >= offset_end) {
            return BranchFreeBinarySearch(array, (jump_idx >> 1u) + offset_beg + 1, offset_end, val);
        }
        if (array[peek_idx] < val) {
            jump_idx <<= 1u;
        }
        else {
            return array[peek_idx] == val ? peek_idx :
                BranchFreeBinarySearch(array, (jump_idx >> 1u) + offset_beg + 1, peek_idx + 1, val);
        }
    }
}



