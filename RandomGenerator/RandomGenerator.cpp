#include <random>

// Random number generator class
class RandomGenerator {
    private:
        std::mt19937 gen; // Mersenne Twister engine
        std::uniform_real_distribution<double> uniform_dist; // Uniform distribution
        std::normal_distribution<double> gaussian_dist; // Gaussian distribution
    
    public:
        // Constructor with random seed
        RandomGenerator(double gaussian_mean = 0.0, double gaussian_stddev = 1.0) {
            std::random_device rd;
            gen.seed(rd());
            gaussian_dist = std::normal_distribution<double>(gaussian_mean, gaussian_stddev);
        }
    
        // Constructor with specified seed
        RandomGenerator(unsigned int seed, double gaussian_mean = 0.0, double gaussian_stddev = 1.0) {
            gen.seed(seed);
            gaussian_dist = std::normal_distribution<double>(gaussian_mean, gaussian_stddev);
        }
    
        // Generate uniform random value in [a, b]
        double rand(double a, double b) {
            uniform_dist = std::uniform_real_distribution<double>(a, b);
            return uniform_dist(gen);
        }
    
        // Generate Gaussian random value (unseeded)
        double rand_gaussian() {
            return gaussian_dist(gen);
        }
    
        // Generate Gaussian random value with specified seed
        double rand_gaussian(unsigned int seed) {
            gen.seed(seed);
            return gaussian_dist(gen);
        }
    
        // Reset Gaussian distribution parameters
        void set_gaussian_params(double mean, double stddev) {
            gaussian_dist = std::normal_distribution<double>(mean, stddev);
        }
    };