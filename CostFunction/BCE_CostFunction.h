#ifndef MSE_COST_FUNCTION_H
#define MSE_COST_FUNCTION_H

#include "CostFunction.h"

class BCE_CostFunction : public CostFunction {
    public:
        BCE_CostFunction(size_t _nVariables) : CostFunction(_nVariables) {
            nInputParams = (_nVariables - 2)/2;
        }
    
        Dual<double> DualRepresentation() override {
            // Cost function: -[y * log(σ(z)) + (1 - y) * log(1 - σ(z))]
            // vars: [y, w0, x0, w1, x1, ..., bias]
            Dual<double> z = vars(2 * nInputParams + 1, 0); // bias
            for (size_t i = 0; i < nInputParams; ++i) {
                z = z + vars(2 * i + 1, 0) * vars(2 * i + 2, 0); // wi * xi
            }
            // Sigmoid: σ(z) = 1 / (1 + exp(-z))
            Dual<double> sigmoid_z = Dual<double>(1.0) / (Dual<double>(1.0) + exp(-z));
            // Add small epsilon to avoid log(0)
            Dual<double> epsilon(1e-10);
            Dual<double> y = vars(0, 0);
            return -(y * log(sigmoid_z + epsilon) + (Dual<double>(1.0) - y) * log(Dual<double>(1.0) - sigmoid_z + epsilon));
        }
    
    private:
        size_t nInputParams; // Number of input parameters
};

#endif // MSE_COST_FUNCTION_H