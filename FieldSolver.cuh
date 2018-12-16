#ifndef _FIELD_SOLVER_H_
#define _FIELD_SOLVER_H_

#include <iostream>
#include <vector>
#include "SpatialMeshCu.cuh"
#include "inner_region.h"
#include "cuda.h"
#include "cuda_runtime.h"

class FieldSolver {
public:
	FieldSolver(SpatialMeshCu &spat_mesh, Inner_regions_manager &inner_regions);
	void eval_potential(Inner_regions_manager &inner_regions);
	void eval_fields_from_potential();
	virtual ~FieldSolver();
private:
	SpatialMeshCu& mesh;

private:
	int max_Jacobi_iterations;
	double rel_tolerance;
	double abs_tolerance;
	double *dev_phi_next;
	//boost::multi_array<double, 3> phi_current;
	//boost::multi_array<double, 3> phi_next;
	void allocate_next_phi();
	void init_constants();
	// Solve potential
	void solve_poisson_eqn_Jacobi(Inner_regions_manager &inner_regions);
	void single_Jacobi_iteration(Inner_regions_manager &inner_regions);
	void set_phi_next_at_boundaries();
	void compute_phi_next_at_inner_points();
	void set_phi_next_at_inner_regions(Inner_regions_manager &inner_regions);
	bool iterative_Jacobi_solutions_converged();
	void set_phi_next_as_phi_current();
	void transfer_solution_to_spat_mesh();
	// Eval fields from potential
	double boundary_difference(double phi1, double phi2, double dx);
	double central_difference(double phi1, double phi2, double dx);
};

#endif  _FIELD_SOLVER_H_
