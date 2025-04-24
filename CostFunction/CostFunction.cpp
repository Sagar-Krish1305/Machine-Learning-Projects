#include "../Diffrentiation/AutoForwardDifferentiator.h"
#include "../LinearAlgebra/Matrix/Matrix.h"
#include <vector>
#include <iostream>

class CostFunction {
protected:
    Matrix<Dual<double>> vars;
    Dual<double> currValue;

public:
    CostFunction(size_t _nVariables) {
        currValue = Dual<double>(0, 0);
        vars = Matrix<Dual<double>>(_nVariables, 1);
        for(int i = 0 ; i < vars.rows() ; i++){
            vars(i, 0).dual = 0; // dual must be initialized to zero
        }
    }

    // Make DualRepresentation pure virtual
    virtual Dual<double> DualRepresentation() = 0;

    void setDualVariable(int index, Dual<double> d) {
        vars(index, 0) = d;
        // Update currValue whenever a variable is set
        currValue = DualRepresentation();
    }

    double getValue() {
        return currValue.real;
    }

    double getDifferentiation(size_t index){
        vars(index, 0).dual = 1;
        setDualVariable(index, vars(index, 0));

        double d = currValue.dual;

        vars(index, 0).dual = 0;
        setDualVariable(index, vars(index, 0));
        return d;
    }

    virtual ~CostFunction() = default; // Add virtual destructor for polymorphism
};

class CustomCostFunction : public CostFunction {
public:
    CustomCostFunction(size_t _nVariables) : CostFunction(_nVariables) {}

    Dual<double> DualRepresentation() override {
        // (y - y0)^2
        Dual<double> y = vars(0, 0);
        Dual<double> y0 = (vars(1, 0) * vars(2, 0)) + vars(3, 0);
        return (y0 - y) * (y0 - y);
    }
};

int main() {
    CustomCostFunction c(4);
    // Set dual part of x to 1 to compute derivative w.r.t. x
    Dual<double> y = Dual<double>(0, 0); // Derivative w.r.t. x
    Dual<double> b0 = Dual<double>(5, 0);
    Dual<double> x = Dual<double>(5, 0);
    Dual<double> b1 = Dual<double>(5, 0);

    c.setDualVariable(0, y);
    c.setDualVariable(1, b0);
    c.setDualVariable(2, x);
    c.setDualVariable(3, b1);

    std::cout << c.getValue() << " " << c.getDifferentiation(0) << std::endl;
    // Expected output: 75 20 (value = 75, derivative w.r.t. x = 20)

    return 0;
}