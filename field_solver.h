#ifndef _FIELD_SOLVER_H_
#define _FIELD_SOLVER_H_

#define BOOST_BIND_NO_PLACEHOLDERS

#include <iostream>
#include <boost/multi_array.hpp>
#include <vector>
#include "spatial_mesh.h"
#include "inner_region.h"
// MPI & OpenMP
//#include <mpi.h>
#include <omp.h>

#include <cusp/hyb_matrix.h>
#include <cusp/monitor.h>
#include <cusp/gallery/poisson.h>
#include <cusp/krylov/cr.h>

//#include <cusp/gallery/poisson.h>
//#include <cusp/coo_matrix.h>
//#include <cusp/print.h>

//#include <thrust/host_vector.h>

class Field_solver {
  public:
    Field_solver( Spatial_mesh &spat_mesh, Inner_regions_manager &inner_regions );
    void eval_potential( Spatial_mesh &spat_mesh, Inner_regions_manager &inner_regions );
    void eval_fields_from_potential( Spatial_mesh &spat_mesh );
    virtual ~Field_solver();
  private:
    int nx, ny, nz;
    double dx, dy, dz;
  private:
    int max_Jacobi_iterations;
    double rel_tolerance;
    double abs_tolerance;
    boost::multi_array<double, 3> phi_current;
    boost::multi_array<double, 3> phi_next;

    cusp::hyb_matrix<int, double, cusp::host_memory> A; // matrix of linear system
    cusp::array1d<double,cusp::host_memory> x; // unknown vector
    cusp::array1d<double,cusp::host_memory> b; // right-hand part of Poisson equation
    cusp::identity_operator<double, cusp::host_memory> M; // preconditioner;
    //cusp::monitor<double> monitor; // terminator of CG evaluation

    void allocate_current_next_phi();
    // Solve potential
    void solve_poisson_eqn_Jacobi( Spatial_mesh &spat_mesh,
				   Inner_regions_manager &inner_regions );
    void init_current_phi_from_spat_mesh_phi( Spatial_mesh &spat_mesh );
    void single_Jacobi_iteration( Spatial_mesh &spat_mesh,
				  Inner_regions_manager &inner_regions );
    void set_phi_next_at_boundaries();
    void set_phi_next_at_inner_regions( Inner_regions_manager &inner_regions );
    bool iterative_Jacobi_solutions_converged();
    void compute_phi_next_at_inner_points( Spatial_mesh &spat_mesh );
    void set_phi_next_as_phi_current();
    void transfer_solution_to_spat_mesh( Spatial_mesh &spat_mesh );
    // Eval fields from potential
    double boundary_difference( double phi1, double phi2, double dx );
    double central_difference( double phi1, double phi2, double dx );
};

#endif /* _FIELD_SOLVER_H_ */
