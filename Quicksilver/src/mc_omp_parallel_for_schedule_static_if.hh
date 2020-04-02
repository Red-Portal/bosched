#if defined(HAVE_OPENMP)
    #pragma omp parallel for schedule (runtime) MC_OMP_PARALLEL_FOR_IF_CONDITION
#endif

