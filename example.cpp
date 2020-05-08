#include <string>
#include <mpi.h>
#include <random>

#include <yalbb/simulator.hpp>
#include <yalbb/probe.hpp>
#include <yalbb/ljpotential.hpp>

#include "include/initial_conditions.hpp"
#include "include/zoltan_fn.hpp"
#include "include/spatial_elements.hpp"

template<int N>
void generate_random_particles(MESH_DATA<elements::Element<N>>& mesh, sim_param_t params){
    std::cout << "Generating data ..." << std::endl;
    std::shared_ptr<initial_condition::lj::RejectionCondition<N>> condition;
    const int MAX_TRIAL = 1000000;
    condition = std::make_shared<initial_condition::lj::RejectionCondition<N>>(
            &(mesh.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
            params.simsize, params.simsize, params.simsize, &params
    );
    initial_condition::lj::UniformRandomElementsGenerator<N>(params.seed, MAX_TRIAL)
            .generate_elements(mesh.els, params.npart, condition);
    std::cout << mesh.els.size() << " Done !" << std::endl;
}

int main(int argc, char** argv) {

    constexpr int N = 2;

    int rank, nproc;
    float ver;
    MESH_DATA<elements::Element<N>> mesh_data;

    std::cout << std::fixed << std::setprecision(6);

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    MPI_Comm APP_COMM;
    MPI_Comm_dup(MPI_COMM_WORLD, &APP_COMM);

    auto option = get_params(argc, argv);
    if (!option.has_value()) {
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    auto params = option.value();

    params.rc = 2.5f * params.sig_lj;
    params.simsize = std::ceil(params.simsize / params.rc) * params.rc;

    MPI_Bcast(&params.seed, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        print_params(params);
    }

    if(Zoltan_Initialize(argc, argv, &ver) != ZOLTAN_OK) {
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    auto zlb = zoltan_create_wrapper(APP_COMM);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////START PARITCLE INITIALIZATION///////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (rank == 0) {
        generate_random_particles<N>(mesh_data, params);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////FINISHED PARITCLE INITIALIZATION///////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Domain-box intersection function *required*
    auto boxIntersectFunc   = [](auto* zlb, double x1, double y1, double z1, double x2, double y2, double z2, int* PEs, int* num_found){
        Zoltan_LB_Box_Assign(zlb, x1, y1, z1, x2, y2, z2, PEs, num_found);
    };
    // Point-in-domain callback *required*
    auto pointAssignFunc    = [](auto* zlb, const auto* e, int* PE) {
        auto pos_in_double = get_as_double_array<N>(e->position);
        Zoltan_LB_Point_Assign(zlb, &pos_in_double.front(), PE);
    };
    // Partitioning + migration function *required*
    auto doLoadBalancingFunc= [](auto* zlb, MESH_DATA<elements::Element<N>>* mesh_data){
        Zoltan_Do_LB<N>(mesh_data, zlb);
    };
    // Data getter function (position and velocity) *required*
    auto getPositionPtrFunc = [](auto* e) -> std::array<Real, N>* { return &e->position; };
    auto getVelocityPtrFunc = [](auto* e) -> std::array<Real, N>* { return &e->velocity; };

    // Short range force function computation
    auto getForceFunc = [eps=params.eps_lj, sig=params.sig_lj, rc=params.rc, getPositionPtrFunc](const auto* receiver, const auto* source){
        return lj_compute_force<N>(receiver, source, eps, sig*sig, rc, getPositionPtrFunc);
    };

    FunctionWrapper fWrapper(getPositionPtrFunc, getVelocityPtrFunc, getForceFunc, boxIntersectFunc, pointAssignFunc, doLoadBalancingFunc);
    auto datatype = elements::register_datatype<N>();
    Probe probe(nproc);

    PolicyExecutor menon_criterion_policy(&probe,[npframe = params.npframe](Probe probe) {
      return (probe.get_current_iteration() % npframe == 0) && (probe.get_cumulative_imbalance_time() >= probe.compute_avg_lb_time());
    });

    auto [t, cum, dec, thist] = simulate<N>(zlb, &mesh_data, &menon_criterion_policy, fWrapper, &params, &probe, datatype, APP_COMM, "menon_");


    MPI_Finalize();
    return 0;

}
