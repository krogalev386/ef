#ifndef _DOMAIN_H_
#define _DOMAIN_H_

#define BOOST_BIND_NO_PLACEHOLDERS

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <hdf5.h>
#include "config.h"
#include "time_grid.h"
#include "spatial_mesh.h"
#include "inner_region.h"
#include "particle_to_mesh_map.h"
#include "field_solver.h"
#include "External_field.h"
#include "particle_interaction_model.h"
#include "particle_source.h"
#include "particle.h"
#include "vec3d.h"
#include "physical_constants.h"
#include <mpi.h>

//#define M_PI 3.14159265358979323846264338327

class Domain {
  private:
    //Domain() {};
  public:
    Time_grid time_grid;
    Spatial_mesh spat_mesh;
    Inner_regions_manager inner_regions;
    Particle_to_mesh_map particle_to_mesh_map;
    Field_solver field_solver;    
    Particle_sources_manager particle_sources;
    External_fields_manager external_fields;
    Particle_interaction_model particle_interaction_model;
  public:
    Domain( Config &conf );
    Domain( hid_t h5file_id );
    void start_pic_simulation();
    void continue_pic_simulation();
    void run_pic();
    void eval_and_write_fields_without_particles();
    void write_step_to_save();
    void write();
    void set_output_filename_prefix_and_suffix( std::string prefix, std::string suffix );
    void prepare_linear_system();
    virtual ~Domain();
  private:
    // Pic algorithm
    void prepare_boris_integration();
    void advance_one_time_step();
    void eval_charge_density();
    void eval_potential_and_fields();
    void push_particles();
    void apply_domain_constrains();
    void remove_particles_inside_inner_regions();
    void update_time_grid();
    // Push particles
    void boris_integration();
    void shift_new_particles_velocities_half_time_step_back();
    void update_momentum( double dt );
    Vec3d compute_electric_field_at_particle_position(
	Particle &particle, unsigned int particle_idx, unsigned int source_idx );
    Vec3d binary_field_at_particle_position(
	Particle &particle, unsigned int particle_idx, unsigned int source_idx );
    Vec3d compute_magnetic_field_at_particle_position( Particle &particle );
    void boris_update_particle_momentum_no_mgn_field(
	Particle &p, double dt, Vec3d total_el_field );
    void boris_update_particle_momentum( Particle &p, double dt,
					 Vec3d total_el_field, Vec3d total_mgn_field );
    void update_position( double dt );    
    // Boundaries and generation
    void apply_domain_boundary_conditions();
    bool out_of_bound( const Particle &p );
    void generate_new_particles();    
    // Various functions
    void print_particles();
    bool negative( hid_t hdf5_id );
    void hdf5_status_check( herr_t status );
    // Write to file
    std::string output_filename_prefix;
    std::string output_filename_suffix;
};

#endif /* _DOMAIN_H_ */
