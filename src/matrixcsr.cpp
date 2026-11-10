#include "matrixcsr.h"
#include <stdexcept>
#include <string>
#include <algorithm>
#include <vector>
#include <numeric>

MatrixCSR::MatrixCSR(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), row_pointers_(rows + 1, 0)
{
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Matrix dimensions should be greater than zero");
    }
}

Vector MatrixCSR::operator*(const Vector& x) const 
{
    assert(cols_ == x.size() && "incompatible dimensions in SpMV");

    Vector result(rows_, Buffer<float>::UninitializedTag{});

    for (size_t i = 0; i < rows_; ++i) 
    {
        float sum = 0.0f;
        for (size_t j = row_pointers_[i]; j < static_cast<size_t>(row_pointers_[i + 1]); ++j) 
        {
            sum += values_[j] * x[col_indices_[j]];
        }
        result[i] = sum;
    }
    return result; 
}

void MatrixCSR::count_node_conexions(size_t row, size_t num_nnz_cols) 
{
    assert(row < rows_ && "row index out of range");
    row_pointers_[row + 1] += static_cast<int32_t>(num_nnz_cols);
}

void MatrixCSR::build_structure() 
{
    for (size_t i = 0; i < rows_; ++i) 
    {
        row_pointers_[i + 1] += row_pointers_[i];
    }

    nnz_ = static_cast<size_t>(row_pointers_[rows_]);

    values_ = Vector(nnz_);
    col_indices_ = Buffer<int32_t>(nnz_, Buffer<int32_t>::UninitializedTag{});

    row_offsets_ = Buffer<int32_t>(rows_, Buffer<int32_t>::UninitializedTag{});
    for (size_t i = 0; i < rows_; ++i) 
    {
        row_offsets_[i] = row_pointers_[i];
    }
}

void MatrixCSR::insert_topology(size_t row, size_t col)
{
    assert(row < rows_ && col < cols_ && "index out of range");

    size_t index = static_cast<size_t>(row_offsets_[row]);
    assert(index < static_cast<size_t>(row_pointers_[row + 1]) && "index out of range(row_pointers_)");

    col_indices_[index] = static_cast<int32_t>(col);
    row_offsets_[row]++;
}

void MatrixCSR::finalize() 
{
    int32_t* cols_ptr = col_indices_.data();
    for (size_t i = 0; i < rows_; ++i) 
    {
        int32_t start = row_pointers_[i];
        int32_t end   = row_pointers_[i + 1];
        if (end - start > 1) {
            std::sort(cols_ptr + start, cols_ptr + end);
        }
    }
}

void MatrixCSR::add_value(size_t row, size_t col, float val) 
{
    size_t start = static_cast<size_t>(row_pointers_[row]);
    size_t end   = static_cast<size_t>(row_pointers_[row + 1]);
    int32_t target_col = static_cast<int32_t>(col);

    for (size_t j = start; j < end; ++j) 
    {
        if (col_indices_[j] == target_col) 
        {
            values_[j] += val;
            return; 
        }
    }

    throw std::runtime_error(
        "MatrixCSR::add_value - coordinate does not exist (" 
        + std::to_string(row) + ", " + std::to_string(col) + ")"
    );
}