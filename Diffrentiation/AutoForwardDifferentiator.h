#include <iostream>
#include <cmath>
#include <stdexcept>
#define PI 3.141592653589793
// Dual number class for automatic differentiation
template<typename T>
class Dual {
public:
    T real; // Value
    T dual; // Derivative

    // Constructors
    Dual(T r = 0, T d = 0) : real(r), dual(d) {}

    // Arithmetic operators
    Dual operator+(const Dual& other) const {
        return Dual(real + other.real, dual + other.dual);
    }

    Dual operator-(const Dual& other) const {
        return Dual(real - other.real, dual - other.dual);
    }

    Dual operator*(const Dual& other) const {
        return Dual(real * other.real, real * other.dual + dual * other.real);
    }

    Dual operator/(const Dual& other) const {
        if (other.real == 0) {
            throw std::runtime_error("Division by zero in Dual");
        }
        T denom = other.real * other.real;
        return Dual(real / other.real, (dual * other.real - real * other.dual) / denom);
    }

    // Unary minus
    Dual operator-() const {
        return Dual(-real, -dual);
    }

    // Compound assignment operators
    Dual& operator+=(const Dual& other) {
        real += other.real;
        dual += other.dual;
        return *this;
    }

    Dual& operator-=(const Dual& other) {
        real -= other.real;
        dual -= other.dual;
        return *this;
    }

    Dual& operator*=(const Dual& other) {
        dual = real * other.dual + dual * other.real;
        real *= other.real;
        return *this;
    }

    Dual& operator/=(const Dual& other) {
        if (other.real == 0) {
            throw std::runtime_error("Division by zero in Dual");
        }
        dual = (dual * other.real - real * other.dual) / (other.real * other.real);
        real /= other.real;
        return *this;
    }

    // Compound assignment operators
    Dual& operator+=(const T val) {
        real += val;
        return *this;
    }

    Dual& operator-=(const T val) {
        real -= val;
        return *this;
    }

    Dual& operator*=(const T val) {
        dual *= val;
        real *= val;
        return *this;
    }

    Dual& operator/=(const T val) {
        if (val == 0) {
            throw std::runtime_error("Division by zero in Dual");
        }
        dual /= val;
        real /= val;
        return *this;
    }

    // Arithmetic operators for constants
    Dual operator+(const T val) const {
        return Dual(real + val, dual);
    }

    Dual operator-(const T val) const {
        return Dual(real - val, dual);
    }

    Dual operator*(const T val) const {
        return Dual(real * val, dual * val);
    }

    Dual operator/(const T val) const {
        if (val == 0) {
            throw std::runtime_error("Division by zero in Dual");
        }
        return Dual(real / val, dual / val);
    }
};

// Stream output
template<typename T>
std::ostream& operator<<(std::ostream& os, const Dual<T>& d) {
    os << "(" << d.real << ", " << d.dual << ")";
    return os;
}

// Overload common math functions for Dual numbers
template<typename T>
Dual<T> sin(const Dual<T>& x) {
    return Dual<T>(std::sin(x.real), std::cos(x.real) * x.dual);
}

template<typename T>
Dual<T> cos(const Dual<T>& x) {
    return Dual<T>(std::cos(x.real), -std::sin(x.real) * x.dual);
}

template<typename T>
Dual<T> exp(const Dual<T>& x) {
    T e = std::exp(x.real);
    return Dual<T>(e, e * x.dual);
}

template <typename T>
Dual<T> log(const Dual<T>& x) {
    return Dual<T>(std::log(x.real), x.dual / x.real);
}

template<typename T>
Dual<T> sigmoid(const Dual<T>& x) {
    // 1 / (1 + e^(-x));
    Dual<T> e_x = exp(-x);
    Dual<T> one = Dual<double>(1.0);
    return one / (one + e_x);
}

// Compute derivative using automatic differentiation
template<typename T>
T compute_derivative(T x) {
    Dual<T> x_dual(x, 1.0); // Set dual part to 1 for derivative
    Dual<T> result = f(x_dual);
    return result.dual;
}

// Activation function: f(x) = 1 + 1/(1 + e^(-x))
template<typename T>
Dual<T> f(const Dual<T>& x, const Dual<T>& y = Dual<double>(0, 0.0)) {
    Dual<T> s1 = sin(x) * exp(x + 1) * 2;
    Dual<T> s2 = sin(y) * 2 * sin(x);
    return s1 + s2;
}
