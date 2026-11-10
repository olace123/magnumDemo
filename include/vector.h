#pragma once
#include "buffer.h"

class Vector: private Buffer<float>
{
    public:

    //---- access methods------
    using Buffer<float>::data;
    using Buffer<float>::at;
    using Buffer<float>::operator[];
    using Buffer<float>::head;
    using Buffer<float>::tail;
    using Buffer<float>::size;
    using Buffer<float>::capacity;
    using Buffer<float>::print;

    // ------ constructors and destructor  ------------

    Vector(size_t size, Buffer<float>::UninitializedTag): Buffer<float>(size, Buffer<float>::UninitializedTag{}){}

    Vector() = default;

    Vector(size_t size): Buffer<float>(size){}

    Vector(size_t size, float value): Buffer<float>(size, value){} 

    Vector(float* external_data, size_t size) : Buffer<float>(external_data, size) {}

    // ------------ overloads for {+=, -=} operators --------------------------

    void operator+=(const Vector& other);

    void operator-=(const Vector& other);

    // ----------- overloads for +operator ------------------------------------------

    Vector operator+(const Vector& other) const;

    friend Vector operator+(Vector&& temp, const Vector& other);

    Vector operator+(Vector&& other) const;

    friend Vector operator+(Vector&& otherL, Vector&& otherR);

    // ----------- overloads for -operator -----------------------------------------

    Vector operator-(const Vector& other) const;
    
    friend Vector operator-(Vector&& temp, const Vector& other);

    Vector operator-(Vector&& other) const;

    friend Vector operator-(Vector&& otherL, Vector&& otherR);

    // -------- dot product -------------------------------------

    double dot(const Vector& other)const;

    friend double dot(const Vector& otherL, const Vector& otherR);

    // -------- cross product -------------------------------------
    Vector cross(const Vector& other)const;

    friend Vector cross(const Vector& otherL, const Vector& otherR);

    // ------------ overloads for operator{*, *=} (escalar multiplication) -----------------
    void operator*=(float x);
    Vector operator*(float x) const;
    friend Vector operator*(Vector&& other, float x);
    friend Vector operator*(float x, const Vector& other);
    friend Vector operator*(float x, Vector&& other);
};