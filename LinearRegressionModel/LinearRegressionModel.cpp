// LinearRegressionModel.cpp
#include "../LinearAlgebra/Matrix/Matrix.h"
#include "../CostFunction/CostFunction.cpp"
// #include "../CostFunction/CustomCostFunction.cpp"
#include "../RandomGenerator/RandomGenerator.h"
#include <memory>
#include <iostream>

class CustomCostFunction : public CostFunction {
    public:
        CustomCostFunction(size_t _nVariables) : CostFunction(_nVariables) {
            nInputParams = (_nVariables - 2)/2;
        }
    
        Dual<double> DualRepresentation() override {
            // Cost function: (y - (w0*x0 + w1*x1 + ... + bias))^2
            // vars(0, 0) = y, vars(0, 1) = w0, vars(0, 2) = x0, ..., vars(0, 2n+1) = bias
            Dual<double> prediction = vars(2*nInputParams + 1, 0); // bias
            for (size_t i = 0; i < nInputParams; ++i) {
                prediction = prediction + vars(2*i + 1, 0) * vars(2*i + 2, 0); // wi * xi
            }
            Dual<double> error = vars(0, 0) - prediction; // y - prediction
            return error * error; // (y - prediction)^2
        }
    
    private:
        size_t nInputParams; // Number of input parameters
};

class LinearRegressionModel {
private:
    size_t nInputParams; // Number of input parameters
    double learningRate;
    Matrix<double> weights;
    double bias;
    std::unique_ptr<CostFunction> C; // Use unique_ptr for safe memory management

public:
    LinearRegressionModel(size_t _nInputParams, double _learningRate = 0.4)
        : nInputParams(_nInputParams), learningRate(_learningRate) {
        // Initialize weights with random Gaussian values
        weights = Matrix<double>::randomGaussian(nInputParams, 1);
        RandomGenerator r(0.0, 1.0);
        bias = r.rand_gaussian();
        // Initialize CostFunction
        C = std::make_unique<CustomCostFunction>(2 * nInputParams + 2);
    }

    double predict(const Matrix<double>& inputs) {
        Matrix<double> ans = ~weights * inputs;
        return ans(0, 0) + bias;
    }

    void setCostFunction(std::unique_ptr<CostFunction> f) {
        if (f->getTotalVariables() != 2 * nInputParams + 2) {
            throw std::invalid_argument("The Cost Function must have proper variable assignment.");
        }
        C = std::move(f);
    }

    void learn(const Matrix<double>& inputs, double target) {
        // Set up cost function variables
        // vars: [y, w0, x0, w1, x1, ..., bias]
        C->setDualVariable(0, Dual<double>(target)); // y

        for (size_t i = 0; i < nInputParams; ++i) {
            // Set wi with dual part 1 to compute ∂C/∂wi
            C->setDualVariable(2 * i + 1, Dual<double>(weights(i, 0)));
            C->setDualVariable(2 * i + 2, Dual<double>(inputs(i, 0))); // xi
        }
        // Set bias with dual part 1 to compute ∂C/∂bias
        C->setDualVariable(2 * nInputParams + 1, Dual<double>(bias));

        // Update weights and bias using gradient descent
        for (size_t i = 0; i < nInputParams; ++i) {
            double dC_dwi = C->getDifferentiation(2 * i + 1); // ∂C/∂wi
            weights(i, 0) -= learningRate * dC_dwi;
        }
        double dC_db = C->getDifferentiation(2 * nInputParams + 1); // ∂C/∂bias
        bias -= learningRate * dC_db;
    }

    void displayModel() const {
        std::cout << "Displaying Model Parameters" << std::endl;
        std::cout << "WEIGHTS: " << std::endl;
        std::cout << ~weights << std::endl;
        std::cout << "BIAS: " << bias << std::endl;
    }
};



int main() {
    LinearRegressionModel model(1, 0.1);

    RandomGenerator r(0.0, 1.0);

    // Training y = 4x using 1000 training inputs
    int T = 1000000;
    for (int i = 0; i < T; ++i) {
        double x = r.rand_gaussian();  
        double some_error = r.rand_gaussian();
        double y = 4 * x + 12 + some_error;

        Matrix<double> input(1, 1);
        input(0, 0) = x;
        model.learn(input, y);

        if (i % 10000 == 0) {
            std::cout << "Iteration " << i << std::endl;
            model.displayModel();
        }
    }


    return 0;
}