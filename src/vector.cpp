#include "vector.h"
#include <stdexcept> 
#include <cassert>

// ------------------ overloads for {+=, -=} operators --------------------------

void Vector::operator+=(const Vector& other)
{
    // when using assert in Debug mode (-g), if vectors are not same size, the program aborts and lets you know in what line the error occured.
    // in Release mode (-O3 -D NDEBUG), the compiler deletes this line. zero impact on performance.
    assert(size_ == other.size_ && "Vectors must be of the same size for addition");
    for (size_t i = 0; i < size_; i++)
    {
        array_[i] += other.array_[i];
    }

}

void Vector::operator-=(const Vector& other)
{
    
    assert(size_ == other.size_ && "Vectors must be of the same size for substraction");
    for (size_t i = 0; i < size_; i++)
    {
        array_[i] -= other.array_[i];
    }

}

// -------- overloads for operator+ -------------------

Vector Vector:: operator+(const Vector& other) const // c++ sum associativity works form left to right, so this overload will be used first in a chained operation
{
    Vector result(*this);
    result+= other;

    return result; /* we might think, why not std::move(sum_vect) ? if we do this, 
    the compiler will use the movement constructor to build (transfer) the result outside the method to an rvalue and then
    the= movement overload to transfer it to its final destiny and since sum_vect 
    is going to die at the method's exit we resonably think this is the best form. WE ARE WRONG!
    sice 2011 the modern compilers apply an optimization technice called NRVO (NAMED RETURN VALUE OPTIMIZATION).
    if somebody in the main does 'V_r = v1 +v2' the compiler notices that sum_vect is going to die soon anyway, 
    so instead of build in it in an exclusive memoryadress for sum_vect, it builds it directly in its final destiny: &V_r! 
    There is a rule for NRVO to be applied. The object must be attempted to be created inside the function/method - I say 
    attempted because it will actually be build in its final destiny-. this means that this type of 'return' does not apply to objects that come from outside. 

    also if the user does 'V_r = v1+v2+v3' then (v1+v2) will be built outside the method, in an rvalue, ready to added to v3. NRVO again !
    */
}


Vector operator+(Vector&& temp, const Vector& other) // value return, so it will use a constructur to build it
/* we could return a reference to an rvalue (Vector&&), but if the user does 'auto_result = V1 + V2 + V3;'
the program could crash, since auto may deduce 'auto == Vector&&' instead of 'Vector'
*/
{
    temp+= other;
    return std::move(temp); // this will force to use the movement constructor to build the rvalue copy

}

Vector Vector:: operator+(Vector&& other) const // case: v_t = v1 + (v2 + v3)
{
    other += *this;
    return std::move(other);
    
}

Vector operator+(Vector&& otherL, Vector&& otherR) // case: v_t = (v1 + v2) + (v3 + v4)
{
    otherL+=otherR;
    otherR.deallocate_memory();
    return std::move(otherL);

}

// -----------overloads for operator- -----------------------------------

Vector Vector:: operator-(const Vector& other) const // c++ substraction associativity works form left to right
{
    Vector aux_vect(*this);
    aux_vect-=other;

    return aux_vect; 
    
}

Vector operator-(Vector&& temp, const Vector& other) 
{

    temp-=other;

    return std::move(temp); // this will force to use the movement constructor to build the rvalue copy

}

Vector Vector:: operator-(Vector&& other) const 
{
    assert(size_ == other.size_ && "Vectors must be of the same size for substraction");
    for (size_t i = 0; i < size_; i++)
    {
        other.array_[i] = array_[i]-other.array_[i];
    }

    return std::move(other);
    
}

Vector operator-(Vector&& otherL, Vector&& otherR) 
{
    otherL-=otherR;
    otherR.deallocate_memory();

    return std::move(otherL);

}

// ------------ dot product -----------------------------------------

double Vector::dot(const Vector& other)const
{
    if (size_ != other.size_) throw std::invalid_argument("Vectors must be of the same size for dot product");

    double sum = 0;
    for (size_t i = 0; i < size_; i++)
    {
        sum+= array_[i]*other.array_[i];
    }
    
    return sum;
}

double dot(const Vector& otherL, const Vector& otherR)
{   
    return otherL.dot(otherR);
}

// ---------- cross product -----------------------------

Vector Vector::cross(const Vector& other)const
{
    if ((size_ != other.size_) || (size_ !=3)) throw std::invalid_argument("Vectors must be of size_ = 3 for crossProduct");
    Vector result(size_, Buffer<float>::UninitializedTag{});
    result.array_[0] = array_[1] * other.array_[2] - array_[2] * other.array_[1];
    result.array_[1] = array_[2] * other.array_[0] - array_[0] * other.array_[2];
    result.array_[2] = array_[0] * other.array_[1] - array_[1] * other.array_[0];

    return result;
}

Vector cross(const Vector& otherL, const Vector& otherR)
{
    
    return otherL.cross(otherR);
}

// ------------ overloads for operators{*, *=} (escalar multiplication) -----------------

void Vector::operator*=(float x)
{
    for (size_t i = 0; i < size_; i++)
    {
        array_[i]*= x;
    }
    
}

Vector Vector::operator*(float x) const
{
    Vector other(*this);
    other*=x;
    return other;
}

Vector operator*(Vector&& other, float x)
{
    other*=x;
    return std::move(other);
}

Vector operator*(float x, const Vector& other)
{
    return other*x;
}

Vector operator*(float x, Vector&& other)
{
    return std::move(other)*x;
}