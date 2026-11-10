#include "solvers.h"
#include <cmath>
#include <iostream>
#include <utility> 

Vector Solver::solve_cg_cpu(const MatrixCSR& A, const Vector& b, float tol, int max_iter) 
{
    // 1. Adivinanza inicial: x = 0
    Vector x(b.size(), 0.0f); 

    // 2. Residuo inicial: r = b - A*x (como x es 0, r = b)
    Vector r = b; 
    
    // 3. Dirección de búsqueda inicial
    Vector p = r; 
    
    double rs_old = r.dot(r); 

    for (int i = 0; i < max_iter; ++i) 
    {
        // SpMV optimizado
        Vector Ap = A * p; 

        double p_dot_Ap = p.dot(Ap);
        if (p_dot_Ap == 0.0) break; // Seguridad matemática para evitar división por cero
        
        float alpha = static_cast<float>(rs_old / p_dot_Ap);

        x += p * alpha; 
        r -= Ap * alpha; 

        double rs_new = r.dot(r);
        
        // Verificación de convergencia
        if (std::sqrt(rs_new) < tol) 
        {
            std::cout << "CG Convergencia alcanzada en la iteración: " << i << "\n";
            break;
        }

        float beta = static_cast<float>(rs_new / rs_old);

        // Actualizamos la dirección de búsqueda
        Vector p_new = r + (p * beta);
        p = std::move(p_new); 

        rs_old = rs_new;
    }

    return x; 
}