#include "../Diffrentiation/AutoForwardDifferentiator.h"
#include "../LinearAlgebra/Matrix/Matrix.h"
#include <vector>
#include <iostream>

class CostFunction {
protected:
    // Storing Variables and their Values
    Matrix<Dual<double>> vars;
    size_t nVariables;

public:
    CostFunction(size_t _nVariables) : nVariables(_nVariables){
        vars = Matrix<Dual<double>>(_nVariables, 1);
        for(int i = 0 ; i < vars.rows() ; i++){
            vars(i, 0).dual = 0; // dual must be initialized to zero
        }
    }

    // Make DualRepresentation pure virtual
    virtual Dual<double> DualRepresentation() = 0;

    void setDualVariable(int index, Dual<double> d) {
        vars(index, 0) = d;
    }

    size_t getTotalVariables(){
        return nVariables;
    }

    double getCurrValue() {
        return DualRepresentation().real;
    }

    double getDifferentiation(size_t index) {
        // Backup current real value
        double originalReal = vars(index, 0).real;
    
        // Create a Dual variable with dual = 1
        setDualVariable(index, Dual<double>(originalReal, 1));
    
        double derivative = DualRepresentation().dual;
    
        // Reset the dual part to 0
        setDualVariable(index, Dual<double>(originalReal, 0));
        return derivative;
    }

    virtual ~CostFunction() = default; // Add virtual destructor for polymorphism
};
