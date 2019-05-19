
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
        ("iter", po::value<int>()->default_value(32), "iterations") 
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
    int iter     = vm["iter"].as<int>();
    double sigma = vm["sigma"].as<double>();
    std::cout << "distribution,mean,+-,dist_mean,dist_stddev"  << std::endl;

    {
        auto warmup =
            [](std::mt19937_64& rng)
            {
                double value;
                double const mu    = 1;
                double const sigma = 1;
                do { value = stats::rnorm(mu, sigma, rng); } while(value < 0.0);
                return value;
            };
        auto gen = workload(1024, warmup, rng);
        benchmark(gen, "warmup", 10, 1);
    }

    {
        auto gaussian =
            [sigma](std::mt19937_64& rng)
            {
                double value;
                double const mu    = 10;
                do
                    {
                        value = stats::rnorm(mu, sigma, rng);   
                    } while(value < 0.0);
                return value;
            };
        auto gen = workload_biased(N, gaussian, rng,
                                   [N](size_t i) {
                                       return static_cast<double>(i) / N + 1;
                                   });
        benchmark(gen, "bias1", iter, 10, sigma);
    }
}
