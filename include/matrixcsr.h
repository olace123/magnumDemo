#pragma once

#include "vector.h"
#include "buffer.h"
#include <cstdint> // for int32_t 

class MatrixCSR 
{
    private:

    size_t rows_ = 0;
    size_t cols_ = 0;
    size_t nnz_  = 0; // number of non zeros

    Vector values_; 

    // topology arrays (we use int_32_t to facilitate CUDA integration)
    Buffer<int32_t> col_indices_;
    Buffer<int32_t> row_pointers_;
    // insertion control
    Buffer<int32_t> row_offsets_; // insertion control array. Is necessary for the insert_topology method 

    public:

    // --- Constructor ---
    MatrixCSR(size_t rows, size_t cols);

    //(SpMV) Sparse matrix-vector multiplication
    Vector operator*(const Vector& x) const;

    // ------- acces methods to the internal arrays (to send them to the GPU)---------------
    // we use constant references so the user can't alter the topology
    const Vector& get_values() const { return values_; }
    const Buffer<int32_t>& get_col_indices() const { return col_indices_; }
    const Buffer<int32_t>& get_row_pointers() const { return row_pointers_; }
        
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t nnz() const { return nnz_; }

    // --------------------- Assembly methods --------------------------

    // -------------FASE 1: symbolic count -------------- 
    // mesh generator or python calls this method for every row 
    void count_node_conexions(size_t row, size_t num_nnz_cols);

    //------------FASE 2: memory alloc ----------------
    // this method is to be called only once before inserting values 
    void build_structure();

    // ----------FASE 3.1: topology is inserted in the CSR matrix -----------
    void insert_topology(size_t row, size_t col);

    //----- FASE 4: sorting (this is neccesary ONLY if the insertion of the topology was parallelized)
    void  finalize();

    // ----- actual contributions to the coefficients of the stiffness matrix------

    void add_value(size_t row, size_t col, float val);

};