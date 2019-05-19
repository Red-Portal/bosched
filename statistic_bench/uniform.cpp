
#include "util.hpp"

int main(int argc, char** argv)
{
    auto seed = std::random_device();
    auto rng  = std::mt19937_64(seed());
    auto str  = std::string(argv[1]);

    namespace po = boost::program_options; 
    using namespace std::literals::string_literals;

    po::options_description desc("Options"); 
    desc.add_options() 
        ("N", po::value<int>(), "iterations") 
        ("sigma", po::value<double>()->default_value(1.0), "variance");

    po::variables_map vm;
    try 
    { 
        po::store(po::command_line_parser(argc, argv)
                  .options(desc).run(), vm);
        po::notify(vm); 
    } 
    catch(po::error& e) 
    { 
        std::cerr << "ERROR: " << e.what() << std::endl << std::endl; 
        std::cerr << desc << std::endl; 
        return 1;
    } 

    int N        = vm["N"].as<int>();
    double sigma = vm["sigma"].as<double>();
    std::cout << "distribution,mean,+-,dist_mean,dist_stddev"  << std::endl;

    for(size_t it = 0; it < 32; ++it) {
#pragma omp parallel for schedule(runtime)
        for(int i = 0; i < 1024; ++i) {
            do_not_optimize(i);
        }
    }

    auto dist =
        [sigma](std::mt19937_64& rng) -> double
        {
            double value;
            double const mu = 10;
            do
            {
                value = stats::rnorm(mu, sigma, rng);   
            } while(value < 0.0);
            return value;
        };
    auto work_gen = workload(N, dist, rng);
    benchmark(work_gen, "uniform", 10, sigma);
}
