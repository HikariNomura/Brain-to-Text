#pragma once
// Minimal .npy (NumPy array format) reader.
// Supports the common numeric dtypes and both C- and Fortran-order arrays.
// All data is converted to double on load for simplicity.
#include <string>
#include <vector>
#include <cstdint>

struct NpyArray {
    std::vector<size_t> shape;
    std::vector<double> data; // always stored in C (row-major) order

    size_t ndim() const { return shape.size(); }
    size_t size() const { return data.size(); }

    // Removes all axes of length 1 (equivalent to numpy.squeeze).
    void squeeze();

    // Strides for the current (C-order) shape, in elements.
    std::vector<size_t> strides() const;
};

// Throws std::runtime_error on any parsing/format problem.
NpyArray load_npy(const std::string& path);

// Saves an NpyArray to a file in NumPy .npy (v1.0) format.
void save_npy(const std::string& path, const NpyArray& arr);
