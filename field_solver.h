#ifndef _FIELD_SOLVER_H_
#define _FIELD_SOLVER_H_

#define BOOST_BIND_NO_PLACEHOLDERS

#include <iostream>
#include <boost/multi_array.hpp>
#include <vector>
#include "spatial_mesh.h"
#include "inner_region.h"

#include <cusp/hyb_matrix.h>
// coverege criteria monitior
#include <cusp/monitor.h>
// delete?
#include <cusp/gallery/poisson.h>
// basic CUSP linear algebra library
#include <cusp/blas/blas.h>
// CUSP methods libraries
#include <cusp/krylov/cg.h>
#include <cusp/krylov/bicg.h>
#include <cusp/krylov/cr.h>
#include <cusp/relaxation/jacobi.h>


class Field_solver {
  public:
    Field_solver( Spatial_mesh &spat_mesh, 
                  Inner_regions_manager &inner_regions );
    Field_solver( Spatial_mesh &spat_mesh, 
                  Inner_regions_manager &inner_regions, 
                  Config &conf );
    Field_solver( Spatial_mesh &spat_mesh, 
                  Inner_regions_manager &inner_regions, 
                  hid_t h5_field_solver_group );
    void eval_potential( Spatial_mesh &spat_mesh, Inner_regions_manager &inner_regions );
    void eval_fields_from_potential( Spatial_mesh &spat_mesh);
    void prepare_linear_system( Spatial_mesh &spat_mesh, Inner_regions_manager &inner_regions );
    void write_to_file( hid_t hdf5_file_id );
    virtual ~Field_solver();
  private:
    int nx, ny, nz;
    double dx, dy, dz;
  private:
    std::string solving_method;
    int max_iterations;
    double rel_tolerance;
    double abs_tolerance;
    boost::multi_array<double, 3> phi_current;
    boost::multi_array<double, 3> phi_next;

    std::vector<int> ind_phi; // storage of indeces of non-boundary and non-inner nodes
    cusp::csr_matrix<int, double, cusp::device_memory> A_d; // matrix of linear system
    cusp::array1d<double,cusp::device_memory> x; // unknown vector
    cusp::array1d<double,cusp::device_memory> b; // right-hand part of Poisson equation
    cusp::array1d<double,cusp::device_memory> b_const; // constant part of b (boundary condition)
    cusp::array1d<double,cusp::device_memory> b_var; // variable part of b (charge density)
    cusp::identity_operator<double, cusp::device_memory> M; // preconditioner;
    // Jacobi features
    cusp::relaxation::jacobi<double, cusp::device_memory> S; // Jacobi smoother
    cusp::array1d<double, cusp::device_memory> r; //resudal vector

    void allocate_current_next_phi();
    // Solve potential
    void solve_poisson_eqn_Jacobi( Spatial_mesh &spat_mesh,
				   Inner_regions_manager &inner_regions );
    void solve_linear_system();
    void solve_CG();
    void solve_BCG();
    void solve_CR();
    void solve_Jacobi();
    void init_current_phi_from_spat_mesh_phi( Spatial_mesh &spat_mesh );
    /*void single_Jacobi_iteration( Spatial_mesh &spat_mesh,
				  Inner_regions_manager &inner_regions );*/
    void set_phi_next_at_boundaries();
    void set_phi_next_at_inner_regions( Inner_regions_manager &inner_regions );
    /*bool iterative_Jacobi_solutions_converged();*/
    void compute_phi_next_at_inner_points( Spatial_mesh &spat_mesh );
    void set_phi_next_as_phi_current();
    void transfer_solution_to_spat_mesh( Spatial_mesh &spat_mesh );
    // Eval fields from potential
    double boundary_difference( double phi1, double phi2, double dx );
    double central_difference( double phi1, double phi2, double dx );
    void hdf5_status_check( herr_t status );
};



#endif /* _FIELD_SOLVER_H_ */
