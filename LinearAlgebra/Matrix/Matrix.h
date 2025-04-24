// Linear Algebra/Matrix/Matrix.h
#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <stdexcept>
#include <cstring> // For std::memset

template <typename T>
class Matrix {
private:
    size_t rows_;
    size_t cols_;
    T* data_; // Raw array for matrix elements (row-major)

public:
    // Default constructor
    Matrix() : rows_(0), cols_(0), data_(nullptr) {}

    // Constructor with dimensions
    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), data_(nullptr) {
        if (rows == 0 || cols == 0) {
            throw std::invalid_argument("Matrix dimensions must be positive");
        }
        data_ = static_cast<T*>(malloc(rows * cols * sizeof(T)));
        if (!data_) {
            throw std::bad_alloc();
        }
        // Initialize elements to 0 (safe for numeric types)
        std::memset(data_, 0, rows * cols * sizeof(T));
    }

    // Copy constructor
    Matrix(const Matrix& other) : rows_(other.rows_), cols_(other.cols_), data_(nullptr) {
        if (other.data_) {
            data_ = static_cast<T*>(malloc(rows_ * cols_ * sizeof(T)));
            if (!data_) {
                throw std::bad_alloc();
            }
            std::memcpy(data_, other.data_, rows_ * cols_ * sizeof(T));
        }
    }

    // Move constructor
    Matrix(Matrix&& other) noexcept : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {
        other.rows_ = 0;
        other.cols_ = 0;
        other.data_ = nullptr;
    }

    // Destructor
    ~Matrix() {
        free(data_);
    }

    // Copy assignment
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            T* new_data = nullptr;
            if (other.data_) {
                new_data = static_cast<T*>(malloc(other.rows_ * other.cols_ * sizeof(T)));
                if (!new_data) {
                    throw std::bad_alloc();
                }
                std::memcpy(new_data, other.data_, other.rows_ * other.cols_ * sizeof(T));
            }
            free(data_);
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = new_data;
        }
        return *this;
    }

    // Move assignment
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            free(data_);
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = other.data_;
            other.rows_ = 0;
            other.cols_ = 0;
            other.data_ = nullptr;
        }
        return *this;
    }

    // Setter: Non-const element access (row-major)
    T& operator()(size_t i, size_t j) {
        if (i >= rows_ || j >= cols_) {
            throw std::out_of_range("Matrix index out of bounds");
        }
        return data_[i * cols_ + j]; // Row-major indexing
    }

    // Getter: Const element access
    const T& operator()(size_t i, size_t j) const {
        if (i >= rows_ || j >= cols_) {
            throw std::out_of_range("Matrix index out of bounds");
        }
        return data_[i * cols_ + j];
    }

    // Get dimensions
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    // Print matrix
    void print() const {
        for (size_t i = 0; i < rows_; ++i) {
            std::cout << "| ";
            for (size_t j = 0; j < cols_; ++j) {
                std::cout << (*this)(i, j) << " ";
            }
            std::cout << "|\n";
        }
        std::cout << "\n";
    }

    // Matrix Transpose
    Matrix<T> operator ~() const {
        Matrix<T> result(cols_, rows_);
        for (size_t i = 0; i < cols_; ++i) {
            for (size_t j = 0; j < rows_; ++j) {
                result(i, j) = (*this)(j, i);
            }
        }
        return result;
    }

    // Static function: Create zero matrix
    static Matrix<T> zeros(size_t rows, size_t cols) {
        if (rows == 0 || cols == 0) {
            throw std::invalid_argument("Matrix dimensions must be positive");
        }
        return Matrix<T>(rows, cols); // Constructor initializes to 0
    }

    // Static function: Create identity matrix
    static Matrix<T> identity(size_t n) {
        Matrix<T> result(n, n);
        for (size_t i = 0; i < n; ++i) {
            result(i, i) = static_cast<T>(1);
        }
        return result;
    }

    // Static function: Create diagonal matrix
    static Matrix<T> diagonal(const T* values, size_t n) {
        Matrix<T> result(n, n);
        for (size_t i = 0; i < n; ++i) {
            result(i, i) = values[i];
        }
        return result;
    }

    // Matrix addition
    Matrix operator+(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions mismatch for addition");
        }
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(i, j) = (*this)(i, j) + other(i, j);
            }
        }
        return result;
    }

    // Matrix multiplication
    Matrix operator*(const Matrix& other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument("Matrix dimensions mismatch for multiplication");
        }
        Matrix result(rows_, other.cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < other.cols_; ++j) {
                T sum = 0;
                for (size_t k = 0; k < cols_; ++k) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    // Scalar multiplication
    Matrix operator*(T scalar) const {
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(i, j) = (*this)(i, j) * scalar;
            }
        }
        return result;
    }

    // Equality operator
    bool operator==(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            return false;
        }
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                if ((*this)(i, j) != other(i, j)) {
                    return false;
                }
            }
        }
        return true;
    }

    // Stream output
    friend std::ostream& operator<<(std::ostream& os, const Matrix& mat) {
        for (size_t i = 0; i < mat.rows_; ++i) {
            os << "| ";
            for (size_t j = 0; j < mat.cols_; ++j) {
                os << mat(i, j) << " ";
            }
            os << "|\n";
        }
        return os;
    }
};

#endif // MATRIX_H