#include "../LinearAlgebra/Matrix/Matrix.h"
#include "../CostFunction/CostFunction.h"
#include "../CostFunction/BCE_CostFunction.h"
#include "../RandomGenerator/RandomGenerator.h"
#include <memory>
#include <iostream>
#include <cmath>

class BinaryLogisticRegressionModel {
private:
    size_t nInputParams; // Number of input parameters
    double learningRate;
    Matrix<double> weights; // Matrix for weights (nInputParams x 1)
    double bias;
    std::unique_ptr<CostFunction> C; // Use unique_ptr for safe memory management

public:
    BinaryLogisticRegressionModel(size_t _nInputParams, double _learningRate = 0.01)
        : nInputParams(_nInputParams), learningRate(_learningRate) {
        if (_nInputParams == 0) {
            throw std::invalid_argument("Number of input parameters must be positive");
        }
        weights = Matrix<double>::randomGaussian(nInputParams, 1);
        RandomGenerator r(0.0, 1.0);
        bias = r.rand_gaussian();
        C = std::make_unique<BCE_CostFunction>(2 * nInputParams + 2);
    }

    double predict(const Matrix<double>& inputs) const {
        if (inputs.rows() != nInputParams || inputs.cols() != 1) {
            throw std::invalid_argument("Input must be nInputParams x 1");
        }
        Matrix<double> ans = ~weights * inputs;
        double z = ans(0, 0) + bias;
        // Clamp z to prevent overflow in exp
        z = std::max(std::min(z, 100.0), -100.0);
        return 1.0 / (1.0 + std::exp(-z));
    }

    int classify(const Matrix<double>& inputs, double threshold = 0.5) const {
        return predict(inputs) >= threshold ? 1 : 0;
    }

    void setCostFunction(std::unique_ptr<CostFunction> f) {
        if (f->getTotalVariables() != 2 * nInputParams + 2) {
            throw std::invalid_argument("The Cost Function must have proper variable assignment.");
        }
        C = std::move(f);
    }

    void learn(const Matrix<double>& inputs, double target) {
        if (inputs.rows() != nInputParams || inputs.cols() != 1) {
            throw std::invalid_argument("Input must be nInputParams x 1");
        }
        // Set baseline variables: [y, w0, x0, w1, x1, ..., bias]
        std::vector<double> gradients(nInputParams + 1);
        C->setDualVariable(0, Dual<double>(target, 0)); // y
        for (size_t j = 0; j < nInputParams; ++j) {
            C->setDualVariable(2 * j + 1, Dual<double>(weights(j, 0), 0)); // wj
            C->setDualVariable(2 * j + 2, Dual<double>(inputs(j, 0), 0)); // xj
        }
        C->setDualVariable(2 * nInputParams + 1, Dual<double>(bias, 0)); // bias

        // Compute gradient for each weight wi
        for (size_t i = 0; i < nInputParams; ++i) {
            C->setDualVariable(2 * i + 1, Dual<double>(weights(i, 0), 1));
            gradients[i] = C->getDifferentiation(2 * i + 1);
            C->setDualVariable(2 * i + 1, Dual<double>(weights(i, 0), 0));
        }

        // Compute gradient for bias
        C->setDualVariable(2 * nInputParams + 1, Dual<double>(bias, 1));
        gradients[nInputParams] = C->getDifferentiation(2 * nInputParams + 1);
        C->setDualVariable(2 * nInputParams + 1, Dual<double>(bias, 0));

        // Update weights and bias, checking for NaN
        for (size_t i = 0; i < nInputParams; ++i) {
            if (!std::isnan(gradients[i])) {
                weights(i, 0) -= learningRate * gradients[i];
            }
        }
        if (!std::isnan(gradients[nInputParams])) {
            bias -= learningRate * gradients[nInputParams];
        }
    }

    void displayModel() const {
        std::cout << "Displaying Model Parameters" << std::endl;
        std::cout << "WEIGHTS: " << std::endl;
        std::cout << ~weights << std::endl;
        std::cout << "BIAS: " << bias << std::endl;
    }
};

int main() {
    BinaryLogisticRegressionModel model(1, 0.01);
    RandomGenerator r(0.0, 2.0); // Increased stddev to balance y=1
    int T = 100000;
    for (int i = 0; i < T; ++i) {
        double x = r.rand(1, 3);
        double y = (x > 2.0) ? 1.0 : 0.0; // Binary label
        Matrix<double> input(1, 1);
        input(0, 0) = x;
        model.learn(input, y);
        if (i % 10000 == 0) {
            std::cout << "Iteration " << i << std::endl;
            model.displayModel();
        }
    }
    // Test prediction
    Matrix<double> test_input(1, 1);
    test_input(0, 0) = 3.0;
    std::cout << "Prediction for x=3: " << model.predict(test_input)
              << ", Class: " << model.classify(test_input) << std::endl;
    return 0;
}
