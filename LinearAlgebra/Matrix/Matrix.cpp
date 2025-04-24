#include<vector>
template <typename T> // making a template class for Matrix 
class Matrix{

    private:
    size_t rows;
    size_t cols;
    std::vector<std::vector<T> > data;

    public:
    Matrix(size_t rows_, size_t cols_){
        rows = rows_;
        cols = cols_;
        data = std::vector<std::vector<T> >(rows, std::vector<T>(cols, 0)); 
    }

    static Matrix zeros(int R, int C){
        return Matrix(R, C);
    }

    T operator ()(int i, int j){
        if(i < 0 || j < 0 || i >= rows || j >= cols){
            throw std::invalid_argument("Matrix index out of bound.");
        }

        return data(i, j);
    }

    Matrix operator *(Matrix m1){

    }

};
