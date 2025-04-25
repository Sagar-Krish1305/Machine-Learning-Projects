#include "CostFunction.cpp"
class MSE_CostFunction : public CostFunction {
    public:
        MSE_CostFunction(size_t _nVariables) : CostFunction(_nVariables) {
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