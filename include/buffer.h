#pragma once

#include <cstddef> // for size_t 
#include <stdexcept> // for the exceptions 
#include <utility> // for std::swap
#include <iostream>
#include <cassert>
#include <algorithm> // for std::fill

// to avoid underflow errors. case:  size_t n = a-b
constexpr size_t MAX_BUFFER_SIZE = 100'000'000;// max size determined at compilation-time

enum class MemoryLocation { // to use NVIDIA UNIFIED MEMORY
    CPU,
    CUDA_UNIFIED
};

template <typename T>
class Buffer 
{
    protected:

    T* array_ = nullptr; // using google standar to name atributes
    size_t size_ = 0;
    size_t capacity_ = 0;
    bool owns_memory_ = false;

    void allocate_memory()
    {
        array_ = new T[capacity_];
        owns_memory_ = true;
    }

    void deallocate_memory()
    {
        if (owns_memory_) {
            delete[] array_;
        }
        array_ = nullptr;
        size_ = 0;
        capacity_ = 0;
        owns_memory_ = false;
    }

    public:

    struct UninitializedTag{};

    // ------ constructors and destructor  ------------

    Buffer(size_t size, UninitializedTag): size_(size), capacity_(size)
    {
        if (size == 0) throw std::invalid_argument("Size must be greater than 0");
        if (size >= MAX_BUFFER_SIZE) throw std::length_error("MAX_BUFFER_SIZE exceeded, check for underflow errors or modify the variable in Buffer.h");
        allocate_memory();
    }

    // trap for signed ints to avoid underflow errors.
    Buffer(int size, UninitializedTag) = delete; 

    Buffer() = default;

    Buffer(size_t size): Buffer(size, UninitializedTag{})
    {
        std::fill(array_, array_+capacity_, T{});
    } 

    Buffer(size_t size, T value): Buffer(size, UninitializedTag{})
    {
        std::fill(array_, array_+capacity_, value);
    }

    // TRAPS FOR SIGNED INTS to avoid underflow errors
    Buffer(int size)=delete;
    Buffer(int size, T value)=delete;
    // ----

    Buffer(const Buffer& other): size_(other.size_), capacity_(other.capacity_)
    {
        allocate_memory();
        for (size_t i = 0; i < size_; i++)
        {
            array_[i] = other.array_[i];
        }
    }

    Buffer(Buffer&& other) noexcept: array_(other.array_), size_(other.size_), capacity_(other.capacity_), owns_memory_(other.owns_memory_)
    {
        other.array_ = nullptr; 
        other.size_ = 0;
        other.capacity_ = 0;
        other.owns_memory_= false;
    }

    Buffer(T* external_data, size_t size) : array_(external_data), size_(size), capacity_(size), owns_memory_(false) 
    {
        // view constructor 
    }

    ~Buffer()
    {
        deallocate_memory();
    }

    //--------- overloads for operator = -------------

    Buffer& operator=(const Buffer& other)
    {
        if (this == &other) return *this;
        Buffer temp(other);
        std::swap(array_, temp.array_);
        std::swap(size_, temp.size_);
        std::swap(capacity_, temp.capacity_);
        std::swap(owns_memory_, temp.owns_memory_);
    
        return *this;
    }

    Buffer& operator=(Buffer&& other) noexcept
    {
        if (this == &other) return *this;

        deallocate_memory();
        std::swap(array_, other.array_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
        std::swap(owns_memory_, other.owns_memory_);
    
        return *this;
    }

    // --------------- access methods ---------------------------------

    T* data() noexcept { return array_; }
    const T* data() const noexcept { return array_; }

    T& at(size_t index)
    {
        if (index >= size_) throw std::out_of_range("Index out of range");
        return array_[index];
    }

    const T& at(size_t index) const
    {
        if (index >= size_) throw std::out_of_range("Index out of range");
        return array_[index];
    }

    T& operator[](size_t index)
    {
        assert(index < size_ && "Index out of bounds");
        return array_[index];
    }

    const T& operator[](size_t index) const 
    {
        assert(index < size_ && "Index out of bounds");
        return array_[index];
    }

    T& head() 
    {
        if (size_ == 0) throw std::out_of_range("Buffer is empty");
        return array_[0]; 
    }

    const T& head() const 
    { 
        if (size_ == 0) throw std::out_of_range("Buffer is empty");
        return array_[0]; 
    }

    T& tail() 
    {
        if (size_ == 0) throw std::out_of_range("Buffer is empty");
        return array_[size_ - 1]; 
    }

    const T& tail() const 
    { 
        if (size_ == 0) throw std::out_of_range("Buffer is empty");
        return array_[size_ - 1]; 
    }

    size_t size()const
    {
        return size_;
    }

    size_t capacity()const
    {
        return capacity_;
    }

    void print()const
    {
        if (size_ == 0) {
            std::cout<<"()\n";
            return;
        }
        std::cout<<'(';
        for (size_t i = 0; i < size_-1; i++)
        {
            std::cout<<array_[i]<<',';
        }
        std::cout<<array_[size_-1]<<")\n";
    }
};