#ifndef _FIELD_SOLVER_H_
#define _FIELD_SOLVER_H_

#define BOOST_BIND_NO_PLACEHOLDERS

#include <iostream>
#include <boost/multi_array.hpp>
#include <vector>
#include "spatial_mesh.h"
#include "inner_region.h"

#include <cusp/hyb_matrix.h>
#include <cusp/monitor.h>
#include <cusp/gallery/poisson.h>
#include <cusp/krylov/cg.h>
#include <cusp/blas/blas.h>


class Field_solver {
  public:
    Field_solver( Spatial_mesh &spat_mesh, Inner_regions_manager &inner_regions );
    void eval_potential( Spatial_mesh &spat_mesh, Inner_regions_manager &inner_regions );
    void eval_fields_from_potential( Spatial_mesh &spat_mesh);
    void prepare_linear_system( Spatial_mesh &spat_mesh, Inner_regions_manager &inner_regions );
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

    cusp::csr_matrix<int, double, cusp::device_memory> A_d; // matrix of linear system
    cusp::array1d<double,cusp::device_memory> x; // unknown vector
    cusp::array1d<double,cusp::device_memory> b; // right-hand part of Poisson equation
    cusp::array1d<double,cusp::device_memory> b_const; // constant part of b (boundary condition)
    cusp::array1d<double,cusp::device_memory> b_var; // variable part of b (charge density)
    cusp::identity_operator<double, cusp::device_memory> M; // preconditioner;
    std::vector<int> ind_phi; // storage of indeces of non-boundary and non-inner nodes


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
