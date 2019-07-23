#include "particle_source.h"

void check_and_warn_if_not( const bool &should_be, const std::string &message );
void check_and_exit_if_not( const bool &should_be, const std::string &message );

Particle_source::Particle_source( 
    Config &conf, 
    Particle_source_config_part &src_conf )
{
    //check_correctness_of_related_config_fields( conf, src_conf );
    set_parameters_from_config( src_conf );
}

Particle_source::Particle_source( hid_t h5_particle_source_group_id )
{
    // Read from h5
    read_hdf5_source_parameters( h5_particle_source_group_id );
    read_hdf5_particles( h5_particle_source_group_id );
    // Random number generator
    // Instead of saving/loading it's state to file just
    // reinit with different seed.
    //
    // https://stackoverflow.com/questions/19665818/generate-random-numbers-using-c11-random-library
    //
    std::random_device rd;
    rnd_gen = std::mt19937( rd() );
}

void Particle_source::check_correctness_of_related_config_fields( 
    Config &conf, 
    Particle_source_config_part &src_conf )
{
    initial_number_of_particles_gt_zero( conf, src_conf );
    particles_to_generate_each_step_ge_zero( conf, src_conf );
    temperature_gt_zero( conf, src_conf );
    mass_gt_zero( conf, src_conf );
}

void Particle_source::set_parameters_from_config( Particle_source_config_part &src_conf )
{
    name = src_conf.name;
    initial_number_of_particles = src_conf.initial_number_of_particles;
    particles_to_generate_each_step = 
	src_conf.particles_to_generate_each_step;
    mean_momentum = vec3d_init( src_conf.mean_momentum_x, 
				src_conf.mean_momentum_y,
				src_conf.mean_momentum_z );
    temperature = src_conf.temperature;
    charge = src_conf.charge;
    mass = src_conf.mass;
    /*-------------------------------------------*/
    time_particle_injection_start = src_conf.time_particle_injection_start;
    time_particle_injection_stop = src_conf.time_particle_injection_stop;
    /*-------------------------------------------*/
    std::random_device rd;
    rnd_gen = std::mt19937( rd() );
    // Initial id
    max_id = 0;
}
	
void Particle_source::read_hdf5_source_parameters( hid_t h5_particle_source_group_id )
{	
    herr_t status;

    size_t grp_name_size = 0;
    char *grp_name = NULL;
    grp_name_size = H5Iget_name( h5_particle_source_group_id, grp_name, grp_name_size );
    grp_name_size = grp_name_size + 1;
    grp_name = new char[ grp_name_size ];
    grp_name_size = H5Iget_name( h5_particle_source_group_id, grp_name, grp_name_size );
    std::string longname = std::string( grp_name );
    name = longname.substr( longname.find_last_of("/") + 1 );
    delete[] grp_name;

    double mean_mom_x, mean_mom_y, mean_mom_z;
    
    status = H5LTget_attribute_double( h5_particle_source_group_id, "./",
				       "temperature", &temperature );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_particle_source_group_id, "./",
				       "mean_momentum_x", &mean_mom_x );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_particle_source_group_id, "./",
				       "mean_momentum_y", &mean_mom_y );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_particle_source_group_id, "./",
				       "mean_momentum_z", &mean_mom_z );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_particle_source_group_id, "./",
				       "charge", &charge );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_particle_source_group_id, "./",
				       "mass", &mass );
    hdf5_status_check( status );
    status = H5LTget_attribute_int( h5_particle_source_group_id, "./",
				    "initial_number_of_particles",
				    &initial_number_of_particles );
    hdf5_status_check( status );
    status = H5LTget_attribute_int( h5_particle_source_group_id, "./",
				    "particles_to_generate_each_step",
				    &particles_to_generate_each_step );
    hdf5_status_check( status );
    status = H5LTget_attribute_uint( h5_particle_source_group_id, "./",
				     "max_id", &max_id );
    hdf5_status_check( status );
    /*---------------------------------------------------------------*/
    status = H5LTget_attribute_double( h5_particle_source_group_id, "./",
				     "time_particle_injection_start", 
                                     &time_particle_injection_start );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_particle_source_group_id, "./",
				     "time_particle_injection_stop", 
                                     &time_particle_injection_stop );
    hdf5_status_check( status );
    /*---------------------------------------------------------------*/
    mean_momentum = vec3d_init( mean_mom_x, mean_mom_y, mean_mom_z );
}


void Particle_source::read_hdf5_particles( hid_t h5_particle_source_group_id )
{    
    herr_t status;
    hid_t filespace, memspace, dset;
    hid_t plist_id;
    const int ndims = 1;
    hsize_t dims[ndims];
    
    dset = H5Dopen( h5_particle_source_group_id, "./particle_id", H5P_DEFAULT );
    hdf5_status_check( dset );
    filespace = H5Dget_space( dset ); hdf5_status_check( filespace );
    int actual_ndims = H5Sget_simple_extent_ndims( filespace );
    check_and_exit_if_not( actual_ndims == 1,
			   "N of dimensions in Particle dataset != 1" );
    H5Sget_simple_extent_dims( filespace, dims, NULL );
    status = H5Sclose( filespace ); hdf5_status_check( status );
    status = H5Dclose( dset ); hdf5_status_check( status );

    unsigned int total_num_of_particles = dims[0];
    
    int *id_buf = new int[ total_num_of_particles ];
    double *x_buf = new double[ total_num_of_particles ];
    double *y_buf = new double[ total_num_of_particles ];
    double *z_buf = new double[ total_num_of_particles ];
    double *px_buf = new double[ total_num_of_particles ];
    double *py_buf = new double[ total_num_of_particles ];
    double *pz_buf = new double[ total_num_of_particles ];
    
    memspace = filespace = H5S_ALL;
    plist_id = H5P_DEFAULT;	
    
    dset = H5Dopen( h5_particle_source_group_id, "./particle_id", plist_id );
    hdf5_status_check( dset );
    status = H5Dread( dset, H5T_NATIVE_INT,
		      memspace, filespace, plist_id, id_buf );
    hdf5_status_check( status );
    status = H5Dclose( dset ); hdf5_status_check( status );

    dset = H5Dopen( h5_particle_source_group_id, "./position_x", plist_id );
    hdf5_status_check( dset );
    status = H5Dread( dset, H5T_NATIVE_DOUBLE,
		      memspace, filespace, plist_id, x_buf );
    hdf5_status_check( status );
    status = H5Dclose( dset ); hdf5_status_check( status );

    dset = H5Dopen( h5_particle_source_group_id, "./position_y", plist_id );
    hdf5_status_check( dset );
    status = H5Dread( dset, H5T_NATIVE_DOUBLE,
		      memspace, filespace, plist_id, y_buf );
    hdf5_status_check( status );
    status = H5Dclose( dset ); hdf5_status_check( status );

    dset = H5Dopen( h5_particle_source_group_id, "./position_z", plist_id );
    hdf5_status_check( dset );
    status = H5Dread( dset, H5T_NATIVE_DOUBLE,
		      memspace, filespace, plist_id, z_buf );
    hdf5_status_check( status );
    status = H5Dclose( dset ); hdf5_status_check( status );

    
    dset = H5Dopen( h5_particle_source_group_id, "./momentum_x", plist_id );
    hdf5_status_check( dset );
    status = H5Dread( dset, H5T_NATIVE_DOUBLE,
		      memspace, filespace, plist_id, px_buf );
    hdf5_status_check( status );
    status = H5Dclose( dset ); hdf5_status_check( status );

    dset = H5Dopen( h5_particle_source_group_id, "./momentum_y", plist_id );
    hdf5_status_check( dset );
    status = H5Dread( dset, H5T_NATIVE_DOUBLE,
		      memspace, filespace, plist_id, py_buf );
    hdf5_status_check( status );
    status = H5Dclose( dset ); hdf5_status_check( status );

    dset = H5Dopen( h5_particle_source_group_id, "./momentum_z", plist_id );
    hdf5_status_check( dset );
    status = H5Dread( dset, H5T_NATIVE_DOUBLE,
		      memspace, filespace, plist_id, pz_buf );
    hdf5_status_check( status );
    status = H5Dclose( dset ); hdf5_status_check( status );
        
    // status = H5Sclose( filespace ); hdf5_status_check( status );
    // status = H5Sclose( memspace ); hdf5_status_check( status );
    // status = H5Pclose( plist_id ); hdf5_status_check( status );

    particles.reserve( total_num_of_particles );
    for( unsigned int i = 0; i < total_num_of_particles; i++ ){
	Vec3d pos = vec3d_init( x_buf[i], y_buf[i], z_buf[i] );
	Vec3d mom = vec3d_init( px_buf[i], py_buf[i], pz_buf[i] );
	particles.emplace_back( id_buf[i], charge, mass, pos, mom );
	particles[i].momentum_is_half_time_step_shifted = true;
    }     

    delete[] id_buf;
    delete[] x_buf;
    delete[] y_buf;
    delete[] z_buf;
    delete[] px_buf;
    delete[] py_buf;
    delete[] pz_buf;
}

void Particle_source::generate_initial_particles()
{
    //particles.reserve( initial_number_of_particles );
    generate_num_of_particles( initial_number_of_particles );
}

void Particle_source::generate_each_step()
{
    //particles.reserve( particles.size() + particles_to_generate_each_step );
    generate_num_of_particles( particles_to_generate_each_step );
}
    
void Particle_source::generate_num_of_particles( int num_of_particles )
{
    Vec3d pos, mom;
    std::vector<int> vec_of_ids;

    populate_vec_of_ids( vec_of_ids, num_of_particles ); 
    for ( int i = 0; i < num_of_particles; i++ ) {
	pos = uniform_position_in_source( rnd_gen );
	mom = maxwell_momentum_distr( mean_momentum, temperature, mass, rnd_gen );
	particles.emplace_back( vec_of_ids[i], charge, mass, pos, mom );
    }
}


void Particle_source::populate_vec_of_ids(
    std::vector<int> &vec_of_ids, int num_of_particles )
{
    vec_of_ids.reserve( num_of_particles );
    
    for( int i = 0; i < num_of_particles; i++ ){
	vec_of_ids.push_back( max_id++ );
    }
}

double Particle_source::random_in_range( 
    const double low, const double up, std::mt19937 &rnd_gen )
{
    std::uniform_real_distribution<double> uniform_distr( low, up );
    return uniform_distr( rnd_gen );
}

Vec3d Particle_source::maxwell_momentum_distr(
    const Vec3d mean_momentum, const double temperature, const double mass, 
    std::mt19937 &rnd_gen )
{    
    double maxwell_gauss_std_mean_x = vec3d_x( mean_momentum );
    double maxwell_gauss_std_mean_y = vec3d_y( mean_momentum );
    double maxwell_gauss_std_mean_z = vec3d_z( mean_momentum );
    /*-------------------------------------------------------*/
    const double k = 1.38065e-16;
    double maxwell_gauss_std_dev = sqrt( k * temperature / mass );
    /*-------------------------------------------------------*/
    std::normal_distribution<double> 
	normal_distr_x( maxwell_gauss_std_mean_x, maxwell_gauss_std_dev );
    std::normal_distribution<double> 
	normal_distr_y( maxwell_gauss_std_mean_y, maxwell_gauss_std_dev );
    std::normal_distribution<double> 
	normal_distr_z( maxwell_gauss_std_mean_z, maxwell_gauss_std_dev );

    Vec3d mom;
    mom = vec3d_init( normal_distr_x( rnd_gen ),
		      normal_distr_y( rnd_gen ),
		      normal_distr_z( rnd_gen ) );		     
    mom = vec3d_times_scalar( mom, 1.0 ); // recheck
    return mom;
}

void Particle_source::update_particles_position( double dt )
{
    for ( auto &p : particles )
	p.update_position( dt );
}


void Particle_source::print_particles()
{
    std::cout << "Source name: " << name << std::endl;
    for ( auto& p : particles  ) {	
	p.print_short();
    }
    return;
}

void Particle_source::write_to_file( hid_t group_id )
{
    std::cout << "Source name = " << name << ", "
	      << "number of particles = " << particles.size()
	      << std::endl;
    hid_t current_source_group_id;
    herr_t status;
    std::string table_of_particles_name = name;

    current_source_group_id = H5Gcreate( group_id,
					 ( "./" + table_of_particles_name ).c_str(),
					 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hdf5_status_check( current_source_group_id );

    write_hdf5_particles( current_source_group_id );
    write_hdf5_source_parameters( current_source_group_id );

    status = H5Gclose( current_source_group_id );
    hdf5_status_check( status );
    
    return;
}

void Particle_source::write_hdf5_particles( hid_t current_source_group_id )
{    
    herr_t status;
    hid_t filespace, memspace, dset;
    hid_t plist_id;
    int rank = 1;
    hsize_t dims[rank];
    dims[0] = particles.size();
    
    filespace = H5Screate_simple( rank, dims, NULL );
    memspace = H5S_ALL;
    plist_id = H5P_DEFAULT;
    
    // todo: is it possible to get rid of this copying?
    int *id_buf = new int[ particles.size() ];
    double *x_buf = new double[ particles.size() ];
    double *y_buf = new double[ particles.size() ];
    double *z_buf = new double[ particles.size() ];
    double *px_buf = new double[ particles.size() ];
    double *py_buf = new double[ particles.size() ];
    double *pz_buf = new double[ particles.size() ];
    
    for( unsigned int i = 0; i < particles.size(); i++ ){
	id_buf[i] = particles[i].id;
	x_buf[i] = vec3d_x( particles[i].position );
	y_buf[i] = vec3d_y( particles[i].position );
	z_buf[i] = vec3d_z( particles[i].position );
	px_buf[i] = vec3d_x( particles[i].momentum );
	py_buf[i] = vec3d_y( particles[i].momentum );
	pz_buf[i] = vec3d_z( particles[i].momentum );
    }     
    
    dset = H5Dcreate( current_source_group_id, "./particle_id",
		      H5T_STD_I32BE, filespace,
		      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    hdf5_status_check( dset );
    status = H5Dwrite( dset, H5T_NATIVE_INT,
		       memspace, filespace, plist_id, id_buf );
    hdf5_status_check( status );
    status = H5Dclose( dset ); hdf5_status_check( status );

    dset = H5Dcreate( current_source_group_id, "./position_x",
		      H5T_IEEE_F64BE, filespace,
		      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    hdf5_status_check( dset );
    status = H5Dwrite( dset, H5T_NATIVE_DOUBLE,
		       memspace, filespace, plist_id, x_buf );
    hdf5_status_check( status );
    status = H5Dclose( dset ); hdf5_status_check( status );

    dset = H5Dcreate( current_source_group_id, "./position_y",
		      H5T_IEEE_F64BE, filespace,
		      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    hdf5_status_check( dset );
    status = H5Dwrite( dset, H5T_NATIVE_DOUBLE,
		       memspace, filespace, plist_id, y_buf );
    hdf5_status_check( status );
    status = H5Dclose( dset ); hdf5_status_check( status );

    dset = H5Dcreate( current_source_group_id, "./position_z",
		      H5T_IEEE_F64BE, filespace,
		      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    hdf5_status_check( dset );
    status = H5Dwrite( dset, H5T_NATIVE_DOUBLE,
		       memspace, filespace, plist_id, z_buf );
    hdf5_status_check( status );
    status = H5Dclose( dset ); hdf5_status_check( status );

    
    dset = H5Dcreate( current_source_group_id, "./momentum_x",
		      H5T_IEEE_F64BE, filespace,
		      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    hdf5_status_check( dset );
    status = H5Dwrite( dset, H5T_NATIVE_DOUBLE,
		       memspace, filespace, plist_id, px_buf );
    hdf5_status_check( status );
    status = H5Dclose( dset ); hdf5_status_check( status );

    dset = H5Dcreate( current_source_group_id, "./momentum_y",
		      H5T_IEEE_F64BE, filespace,
		      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    hdf5_status_check( dset );
    status = H5Dwrite( dset, H5T_NATIVE_DOUBLE,
		       memspace, filespace, plist_id, py_buf );
    hdf5_status_check( status );
    status = H5Dclose( dset ); hdf5_status_check( status );

    dset = H5Dcreate( current_source_group_id, "./momentum_z",
		      H5T_IEEE_F64BE, filespace,
		      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    hdf5_status_check( dset );
    status = H5Dwrite( dset, H5T_NATIVE_DOUBLE,
		       memspace, filespace, plist_id, pz_buf );
    hdf5_status_check( status );
    status = H5Dclose( dset ); hdf5_status_check( status );
        
    status = H5Sclose( filespace ); hdf5_status_check( status );
    // status = H5Sclose( memspace ); hdf5_status_check( status );
    // status = H5Pclose( plist_id ); hdf5_status_check( status );

    delete[] id_buf;
    delete[] x_buf;
    delete[] y_buf;
    delete[] z_buf;
    delete[] px_buf;
    delete[] py_buf;
    delete[] pz_buf;
}


void Particle_source::write_hdf5_source_parameters( hid_t current_source_group_id )
{
    herr_t status;
    int single_element = 1;
    std::string current_group = "./";
    double mean_mom_x = vec3d_x( mean_momentum );
    double mean_mom_y = vec3d_y( mean_momentum );
    double mean_mom_z = vec3d_z( mean_momentum );
    
    status = H5LTset_attribute_string( current_source_group_id,
				       current_group.c_str(),
				       "geometry_type", geometry_type.c_str() );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "temperature", &temperature, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "mean_momentum_x", &mean_mom_x, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "mean_momentum_y", &mean_mom_y, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "mean_momentum_z", &mean_mom_z, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "charge", &charge, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "mass", &mass, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_int( current_source_group_id,
				    current_group.c_str(),
				    "initial_number_of_particles",
				    &initial_number_of_particles,
				    single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_int( current_source_group_id,
				    current_group.c_str(),
				    "particles_to_generate_each_step",
				    &particles_to_generate_each_step,
				    single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_uint( current_source_group_id,
				     current_group.c_str(),
				     "max_id", &max_id, single_element );
    hdf5_status_check( status );
}


void Particle_source::initial_number_of_particles_gt_zero( 
    Config &conf, 
    Particle_source_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.initial_number_of_particles > 0,
	"initial_number_of_particles <= 0" );
}

void Particle_source::particles_to_generate_each_step_ge_zero( 
    Config &conf, 
    Particle_source_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.particles_to_generate_each_step >= 0,
	"particles_to_generate_each_step < 0" );
}


void Particle_source::temperature_gt_zero( 
    Config &conf, 
    Particle_source_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.temperature >= 0,
	"temperature < 0" );
}

void Particle_source::mass_gt_zero( 
    Config &conf, 
    Particle_source_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.mass >= 0,
	"mass < 0" );
}

void Particle_source::hdf5_status_check( herr_t status )
{
    if( status < 0 ){
	std::cout << "Something went wrong while reading or writing Particle_source "
		  << name << "."
		  << "Aborting." << std::endl;
	exit( EXIT_FAILURE );
    }
}



// Box source


Particle_source_box::Particle_source_box( 
    Config &conf, 
    Particle_source_box_config_part &src_conf ) :
    Particle_source( conf, src_conf )
{
    geometry_type = "box";
    check_correctness_of_related_config_fields( conf, src_conf );
    set_parameters_from_config( src_conf );
    generate_initial_particles();
}


Particle_source_box::Particle_source_box( hid_t h5_particle_source_box_group_id ) :
    Particle_source( h5_particle_source_box_group_id )
{
    geometry_type = "box";
    read_hdf5_source_parameters( h5_particle_source_box_group_id );
}

void Particle_source_box::check_correctness_of_related_config_fields( 
    Config &conf, 
    Particle_source_box_config_part &src_conf )
{
    x_right_ge_zero( conf, src_conf );
    x_right_le_particle_source_x_left( conf, src_conf );
    x_left_le_grid_x_size( conf, src_conf );    
    y_bottom_ge_zero( conf, src_conf );
    y_bottom_le_particle_source_y_top( conf, src_conf );
    y_top_le_grid_y_size( conf, src_conf );
    z_near_ge_zero( conf, src_conf );
    z_near_le_particle_source_z_far( conf, src_conf );
    z_far_le_grid_z_size( conf, src_conf );
}


void Particle_source_box::set_parameters_from_config(
    Particle_source_box_config_part &src_conf )
{
    xleft = src_conf.box_x_left;
    xright = src_conf.box_x_right;
    ytop = src_conf.box_y_top;
    ybottom = src_conf.box_y_bottom;
    znear = src_conf.box_z_near;
    zfar = src_conf.box_z_far;
}


void Particle_source_box::read_hdf5_source_parameters( hid_t h5_particle_source_box_group_id )
{
    herr_t status;
    
    status = H5LTget_attribute_double( h5_particle_source_box_group_id, "./",
				       "box_x_left", &xleft );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_particle_source_box_group_id, "./",
				       "box_x_right", &xright );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_particle_source_box_group_id, "./",
				       "box_y_top", &ytop );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_particle_source_box_group_id, "./",
				       "box_y_bottom", &ybottom );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_particle_source_box_group_id, "./",
				       "box_z_far", &zfar );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_particle_source_box_group_id, "./",
				       "box_z_near", &znear );
    hdf5_status_check( status );
}

void Particle_source_box::x_right_ge_zero( 
    Config &conf, 
    Particle_source_box_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.box_x_right >= 0,
	"box_x_right < 0" );
}

void Particle_source_box::x_right_le_particle_source_x_left( 
    Config &conf, 
    Particle_source_box_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.box_x_right <= src_conf.box_x_left,
	"box_x_right > box_x_left" );
}

void Particle_source_box::x_left_le_grid_x_size( 
    Config &conf, 
    Particle_source_box_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.box_x_left <= conf.mesh_config_part.grid_x_size,
	"box_x_left > grid_x_size" );
}

void Particle_source_box::y_bottom_ge_zero( 
    Config &conf, 
    Particle_source_box_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.box_y_bottom >= 0,
	"box_y_bottom < 0" );
}

void Particle_source_box::y_bottom_le_particle_source_y_top( 
    Config &conf, 
    Particle_source_box_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.box_y_bottom <= src_conf.box_y_top,
	"box_y_bottom > box_y_top" );
}

void Particle_source_box::y_top_le_grid_y_size( 
    Config &conf, 
    Particle_source_box_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.box_y_top <= conf.mesh_config_part.grid_y_size,
	"box_y_top > grid_y_size" );
}

void Particle_source_box::z_near_ge_zero( 
    Config &conf, 
    Particle_source_box_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.box_z_near >= 0,
	"box_z_near < 0" );
}

void Particle_source_box::z_near_le_particle_source_z_far( 
    Config &conf, 
    Particle_source_box_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.box_z_near <= src_conf.box_z_far,
	"box_z_near > box_z_far" );
}

void Particle_source_box::z_far_le_grid_z_size( 
    Config &conf, 
    Particle_source_box_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.box_z_far <= conf.mesh_config_part.grid_z_size,
	"box_z_far > grid_z_size" );
}



void Particle_source_box::write_hdf5_source_parameters( hid_t current_source_group_id )
{
    Particle_source::write_hdf5_source_parameters( current_source_group_id );
    
    herr_t status;
    int single_element = 1;
    std::string current_group = "./";    

    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "box_x_left", &xleft, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "box_x_right", &xright, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "box_y_top", &ytop, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "box_y_bottom", &ybottom, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "box_z_far", &zfar, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "box_z_near", &znear, single_element );
    hdf5_status_check( status );
}


Vec3d Particle_source_box::uniform_position_in_source( std::mt19937 &rnd_gen )
{
    return uniform_position_in_cube( xright, ytop, zfar,
				     xleft, ybottom, znear,
				     rnd_gen );
}

Vec3d Particle_source_box::uniform_position_in_cube( 
    const double xright,  const double ytop, const double zfar,
    const double xleft, const double ybottom, const double znear,
    std::mt19937 &rnd_gen )
{
    return vec3d_init( random_in_range( xright, xleft, rnd_gen ), 
		       random_in_range( ybottom, ytop, rnd_gen ),
		       random_in_range( znear, zfar, rnd_gen ) );
}





// Cylinder source


Particle_source_cylinder::Particle_source_cylinder( 
    Config &conf, 
    Particle_source_cylinder_config_part &src_conf ) :
    Particle_source( conf, src_conf )
{
    geometry_type = "cylinder";
    //check_correctness_of_related_config_fields( conf, src_conf );
    set_parameters_from_config( src_conf );
    generate_initial_particles();
}

Particle_source_cylinder::Particle_source_cylinder(
    hid_t h5_particle_source_cylinder_group_id ) :
    Particle_source( h5_particle_source_cylinder_group_id )
{
    geometry_type = "cylinder";
    read_hdf5_source_parameters( h5_particle_source_cylinder_group_id );
}



void Particle_source_cylinder::check_correctness_of_related_config_fields( 
    Config &conf, 
    Particle_source_cylinder_config_part &src_conf )
{
    // todo:
    radius_gt_zero( conf, src_conf );
    axis_start_x_min_rad_ge_zero( conf, src_conf );
    axis_start_x_plus_rad_le_grid_x_size( conf, src_conf );
    axis_start_y_min_rad_ge_zero( conf, src_conf );
    axis_start_y_plus_rad_le_grid_y_size( conf, src_conf );
    axis_start_z_min_rad_ge_zero( conf, src_conf );
    axis_start_z_plus_rad_le_grid_z_size( conf, src_conf );
    axis_end_x_min_rad_ge_zero( conf, src_conf );
    axis_end_x_plus_rad_le_grid_x_size( conf, src_conf );
    axis_end_y_min_rad_ge_zero( conf, src_conf );
    axis_end_y_plus_rad_le_grid_y_size( conf, src_conf );
    axis_end_z_min_rad_ge_zero( conf, src_conf );
    axis_end_z_plus_rad_le_grid_z_size( conf, src_conf );
}


void Particle_source_cylinder::set_parameters_from_config(
    Particle_source_cylinder_config_part &src_conf )
{
    axis_start_x = src_conf.cylinder_axis_start_x;
    axis_start_y = src_conf.cylinder_axis_start_y;
    axis_start_z = src_conf.cylinder_axis_start_z;
    axis_end_x = src_conf.cylinder_axis_end_x;
    axis_end_y = src_conf.cylinder_axis_end_y;
    axis_end_z = src_conf.cylinder_axis_end_z;
    radius = src_conf.cylinder_radius;
}


void Particle_source_cylinder::read_hdf5_source_parameters(
    hid_t h5_particle_source_cylinder_group_id )
{
    herr_t status;
    
    status = H5LTget_attribute_double( h5_particle_source_cylinder_group_id, "./",
				       "cylinder_axis_start_x", &axis_start_x );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_particle_source_cylinder_group_id, "./",
				       "cylinder_axis_start_y", &axis_start_y );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_particle_source_cylinder_group_id, "./",
				       "cylinder_axis_start_z", &axis_start_z );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_particle_source_cylinder_group_id, "./",
				       "cylinder_axis_end_x", &axis_end_x );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_particle_source_cylinder_group_id, "./",
				       "cylinder_axis_end_y", &axis_end_y );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_particle_source_cylinder_group_id, "./",
				       "cylinder_axis_end_z", &axis_end_z );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_particle_source_cylinder_group_id, "./",
				       "cylinder_radius", &radius );
    hdf5_status_check( status );    
}

void Particle_source_cylinder::radius_gt_zero( 
    Config &conf, 
    Particle_source_cylinder_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.cylinder_radius >= 0,
	"radius < 0" );
}

void Particle_source_cylinder::axis_start_x_min_rad_ge_zero( 
    Config &conf, 
    Particle_source_cylinder_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.cylinder_axis_start_x - src_conf.cylinder_radius >= 0,
	"cylinder_axis_start_x - cylinder_radius < 0" );
}

void Particle_source_cylinder::axis_start_x_plus_rad_le_grid_x_size( 
    Config &conf, 
    Particle_source_cylinder_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.cylinder_axis_start_x + src_conf.cylinder_radius <= conf.mesh_config_part.grid_x_size,
	"cylinder_axis_start_x + cylinder_radius > grid_x_size" );
}

void Particle_source_cylinder::axis_start_y_min_rad_ge_zero( 
    Config &conf, 
    Particle_source_cylinder_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.cylinder_axis_start_y - src_conf.cylinder_radius >= 0,
	"cylinder_axis_start_y - cylinder_radius < 0" );
}

void Particle_source_cylinder::axis_start_y_plus_rad_le_grid_y_size( 
    Config &conf, 
    Particle_source_cylinder_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.cylinder_axis_start_y + src_conf.cylinder_radius <= conf.mesh_config_part.grid_y_size,
	"cylinder_axis_start_y + cylinder_radius > grid_y_size" );
}

void Particle_source_cylinder::axis_start_z_min_rad_ge_zero( 
    Config &conf, 
    Particle_source_cylinder_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.cylinder_axis_start_z - src_conf.cylinder_radius >= 0,
	"cylinder_axis_start_z - cylinder_radius < 0" );
}

void Particle_source_cylinder::axis_start_z_plus_rad_le_grid_z_size( 
    Config &conf, 
    Particle_source_cylinder_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.cylinder_axis_start_z + src_conf.cylinder_radius <= conf.mesh_config_part.grid_z_size,
	"cylinder_axis_start_z + cylinder_radius > grid_z_size" );
}

void Particle_source_cylinder::axis_end_x_min_rad_ge_zero( 
    Config &conf, 
    Particle_source_cylinder_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.cylinder_axis_end_x - src_conf.cylinder_radius >= 0,
	"cylinder_axis_end_x - cylinder_radius < 0" );
}

void Particle_source_cylinder::axis_end_x_plus_rad_le_grid_x_size( 
    Config &conf, 
    Particle_source_cylinder_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.cylinder_axis_end_x + src_conf.cylinder_radius <= conf.mesh_config_part.grid_x_size,
	"cylinder_axis_end_x + cylinder_radius > grid_x_size" );
}

void Particle_source_cylinder::axis_end_y_min_rad_ge_zero( 
    Config &conf, 
    Particle_source_cylinder_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.cylinder_axis_end_y - src_conf.cylinder_radius >= 0,
	"cylinder_axis_end_y - cylinder_radius < 0" );
}

void Particle_source_cylinder::axis_end_y_plus_rad_le_grid_y_size( 
    Config &conf, 
    Particle_source_cylinder_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.cylinder_axis_end_y + src_conf.cylinder_radius <= conf.mesh_config_part.grid_y_size,
	"cylinder_axis_end_y + cylinder_radius > grid_y_size" );
}

void Particle_source_cylinder::axis_end_z_min_rad_ge_zero( 
    Config &conf, 
    Particle_source_cylinder_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.cylinder_axis_end_z - src_conf.cylinder_radius >= 0,
	"cylinder_axis_end_z - cylinder_radius < 0" );
}

void Particle_source_cylinder::axis_end_z_plus_rad_le_grid_z_size( 
    Config &conf, 
    Particle_source_cylinder_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.cylinder_axis_end_z + src_conf.cylinder_radius <= conf.mesh_config_part.grid_z_size,
	"cylinder_axis_end_z + cylinder_radius > grid_z_size" );
}


void Particle_source_cylinder::write_hdf5_source_parameters( hid_t current_source_group_id )
{
    Particle_source::write_hdf5_source_parameters( current_source_group_id );
    
    herr_t status;
    int single_element = 1;
    std::string current_group = "./";    

    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "cylinder_axis_start_x", &axis_start_x,
				       single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "cylinder_axis_start_y", &axis_start_y,
				       single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "cylinder_axis_start_z", &axis_start_z,
				       single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "cylinder_axis_end_x", &axis_end_x,
				       single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "cylinder_axis_end_y", &axis_end_y,
				       single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "cylinder_axis_end_z", &axis_end_z,
				       single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "cylinder_radius", &radius, single_element );
    hdf5_status_check( status );

}


Vec3d Particle_source_cylinder::uniform_position_in_source( std::mt19937 &rnd_gen )
{
    return uniform_position_in_cylinder( rnd_gen );
}

Vec3d Particle_source_cylinder::uniform_position_in_cylinder( std::mt19937 &rnd_gen )
{
    // random point in cylinder along z
    Vec3d cyl_axis = vec3d_init( ( axis_end_x - axis_start_x ),
				 ( axis_end_y - axis_start_y ),
				 ( axis_end_z - axis_start_z ) );
    double cyl_axis_length = vec3d_length( cyl_axis );
    double x, y, z;
    double r, phi;
    r = sqrt( random_in_range( 0.0, 1.0, rnd_gen ) ) * radius;
    phi = random_in_range( 0.0, 2.0 * M_PI, rnd_gen );
    z = random_in_range( 0.0, cyl_axis_length, rnd_gen );
    //
    x = r * cos( phi );
    y = r * sin( phi );
    z = z;
    Vec3d random_pnt_in_cyl_along_z = vec3d_init( x, y, z );
    // rotate:
    // see https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula
    // todo: Too complicated. Try rejection sampling.
    Vec3d random_pnt_in_rotated_cyl;
    Vec3d unit_cyl_axis = vec3d_normalized( cyl_axis );
    Vec3d unit_along_z = vec3d_init( 0, 0, 1.0 );
    Vec3d rotation_axis = vec3d_cross_product( unit_along_z, unit_cyl_axis );
    double rotation_axis_length = vec3d_length( rotation_axis );
    if ( rotation_axis_length == 0 ) {
	if ( copysign( 1.0, vec3d_z( unit_cyl_axis ) ) >= 0 ){
	    random_pnt_in_rotated_cyl = random_pnt_in_cyl_along_z;
	} else {
	    random_pnt_in_rotated_cyl = vec3d_negate( random_pnt_in_cyl_along_z );
	}
    } else {
	Vec3d unit_rotation_axis = vec3d_normalized( rotation_axis );
	double rot_cos = vec3d_dot_product( unit_cyl_axis, unit_along_z );
	double rot_sin = rotation_axis_length;
	
	random_pnt_in_rotated_cyl =
	    vec3d_add(
		vec3d_times_scalar( random_pnt_in_cyl_along_z, rot_cos ),
		vec3d_add(
		    vec3d_times_scalar(
			vec3d_cross_product( unit_rotation_axis,
					     random_pnt_in_cyl_along_z ),
			rot_sin ),
		    vec3d_times_scalar(
			unit_rotation_axis,
			( 1 - rot_cos ) * vec3d_dot_product(
			    unit_rotation_axis,
			    random_pnt_in_cyl_along_z ) ) ) );
    }
    // shift:
    Vec3d shifted = vec3d_add( random_pnt_in_rotated_cyl,
			       vec3d_init( axis_start_x, axis_start_y, axis_start_z ) );
    return shifted;
}




// Tube_Along_Z source


Particle_source_tube_along_z::Particle_source_tube_along_z( 
    Config &conf, 
    Particle_source_tube_along_z_config_part &src_conf ) :
    Particle_source( conf, src_conf )
{
    geometry_type = "tube_along_z";
    check_correctness_of_related_config_fields( conf, src_conf );
    set_parameters_from_config( src_conf );
    generate_initial_particles();
}


Particle_source_tube_along_z::Particle_source_tube_along_z(
    hid_t h5_particle_source_tube_along_z_group_id ) :
    Particle_source( h5_particle_source_tube_along_z_group_id )
{
    geometry_type = "tube_along_z";
    read_hdf5_source_parameters( h5_particle_source_tube_along_z_group_id );
}


void Particle_source_tube_along_z::check_correctness_of_related_config_fields( 
    Config &conf, 
    Particle_source_tube_along_z_config_part &src_conf )
{
    // todo:
    inner_and_outer_radius_gt_zero( conf, src_conf );
    inner_radius_less_outer_radius( conf, src_conf );
    axis_x_min_outer_rad_ge_zero( conf, src_conf );
    axis_x_plus_outer_rad_le_grid_x_size( conf, src_conf );
    axis_y_min_outer_rad_ge_zero( conf, src_conf );
    axis_y_plus_outer_rad_le_grid_y_size( conf, src_conf );
    axis_start_z_ge_zero( conf, src_conf );
    axis_start_z_le_axis_end_z( conf, src_conf );
    axis_end_z_le_grid_z_size( conf, src_conf );
}


void Particle_source_tube_along_z::set_parameters_from_config(
    Particle_source_tube_along_z_config_part &src_conf )
{
    axis_x = src_conf.tube_along_z_axis_x;
    axis_y = src_conf.tube_along_z_axis_y;
    axis_start_z = src_conf.tube_along_z_axis_start_z;
    axis_end_z = src_conf.tube_along_z_axis_end_z;
    inner_radius = src_conf.tube_along_z_inner_radius;
    outer_radius = src_conf.tube_along_z_outer_radius;
}


void Particle_source_tube_along_z::read_hdf5_source_parameters(
    hid_t h5_particle_source_tube_along_z_group_id )
{
    herr_t status;
    
    status = H5LTget_attribute_double( h5_particle_source_tube_along_z_group_id, "./",
				       "tube_along_z_axis_x", &axis_x );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_particle_source_tube_along_z_group_id, "./",
				       "tube_along_z_axis_y", &axis_y );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_particle_source_tube_along_z_group_id, "./",
				       "tube_along_z_axis_start_z", &axis_start_z );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_particle_source_tube_along_z_group_id, "./",
				       "tube_along_z_axis_end_z", &axis_end_z );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_particle_source_tube_along_z_group_id, "./",
				       "tube_along_z_inner_radius", &inner_radius );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_particle_source_tube_along_z_group_id, "./",
				       "tube_along_z_outer_radius", &outer_radius );
    hdf5_status_check( status );
}

void Particle_source_tube_along_z::inner_and_outer_radius_gt_zero( 
    Config &conf, 
    Particle_source_tube_along_z_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.tube_along_z_inner_radius >= 0,
	"inner_radius < 0" );
    check_and_exit_if_not( 
	src_conf.tube_along_z_outer_radius >= 0,
	"outer_radius < 0" );
}

void Particle_source_tube_along_z::inner_radius_less_outer_radius( 
    Config &conf, 
    Particle_source_tube_along_z_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.tube_along_z_inner_radius <= src_conf.tube_along_z_outer_radius,
	"inner_radius > outer_radius" );
}


void Particle_source_tube_along_z::axis_x_min_outer_rad_ge_zero( 
    Config &conf, 
    Particle_source_tube_along_z_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.tube_along_z_axis_x - src_conf.tube_along_z_outer_radius >= 0,
	"tube_along_z_axis_x - tube_along_z_outer_radius < 0" );
}

void Particle_source_tube_along_z::axis_x_plus_outer_rad_le_grid_x_size( 
    Config &conf, 
    Particle_source_tube_along_z_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.tube_along_z_axis_x + src_conf.tube_along_z_outer_radius <=
	conf.mesh_config_part.grid_x_size,
	"tube_along_z_axis_x + tube_along_z_outer_radius > grid_x_size" );
}

void Particle_source_tube_along_z::axis_y_min_outer_rad_ge_zero( 
    Config &conf, 
    Particle_source_tube_along_z_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.tube_along_z_axis_y - src_conf.tube_along_z_outer_radius >= 0,
	"tube_along_z_axis_y - tube_along_z_outer_radius < 0" );
}

void Particle_source_tube_along_z::axis_y_plus_outer_rad_le_grid_y_size( 
    Config &conf, 
    Particle_source_tube_along_z_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.tube_along_z_axis_y + src_conf.tube_along_z_outer_radius <=
	conf.mesh_config_part.grid_y_size,
	"tube_along_z_axis_y + tube_along_z_outer_radius > grid_y_size" );
}

void Particle_source_tube_along_z::axis_start_z_ge_zero( 
    Config &conf, 
    Particle_source_tube_along_z_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.tube_along_z_axis_start_z >= 0,
	"tube_along_z_axis_start_z < 0" );
}

void Particle_source_tube_along_z::axis_start_z_le_axis_end_z( 
    Config &conf, 
    Particle_source_tube_along_z_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.tube_along_z_axis_start_z <= src_conf.tube_along_z_axis_end_z,
	"tube_along_z_axis_start_z > tube_along_z_axis_end_z" );
}

void Particle_source_tube_along_z::axis_end_z_le_grid_z_size( 
    Config &conf, 
    Particle_source_tube_along_z_config_part &src_conf )
{
    check_and_exit_if_not( 
	src_conf.tube_along_z_axis_end_z <=
	conf.mesh_config_part.grid_z_size,
	"tube_along_z_axis_end_z > grid_z_size" );
}


void Particle_source_tube_along_z::write_hdf5_source_parameters(
    hid_t current_source_group_id )
{
    Particle_source::write_hdf5_source_parameters( current_source_group_id );
    
    herr_t status;
    int single_element = 1;
    std::string current_group = "./";    

    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "tube_along_z_axis_x", &axis_x,
				       single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "tube_along_z_axis_y", &axis_y,
				       single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "tube_along_z_axis_start_z", &axis_start_z,
				       single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "tube_along_z_axis_end_z", &axis_end_z,
				       single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "tube_along_z_outer_radius", &outer_radius,
				       single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_source_group_id,
				       current_group.c_str(),
    				       "tube_along_z_inner_radius", &inner_radius,
				       single_element );
    hdf5_status_check( status );
}


Vec3d Particle_source_tube_along_z::uniform_position_in_source( std::mt19937 &rnd_gen )
{
    return uniform_position_in_tube_along_z( rnd_gen );
}

Vec3d Particle_source_tube_along_z::uniform_position_in_tube_along_z(
    std::mt19937 &rnd_gen )
{
    double x, y, z;
    double r, phi;
    r = sqrt(random_in_range( inner_radius / outer_radius, 1.0, rnd_gen)) * outer_radius;
    phi = random_in_range( 0.0, 2.0 * M_PI, rnd_gen );
    z = random_in_range( axis_start_z, axis_end_z, rnd_gen );
    //
    x = r * cos( phi );
    y = r * sin( phi );
    z = z;     
    return vec3d_init( x + axis_x, y + axis_y, z );
}



void check_and_warn_if_not( const bool &should_be, const std::string &message )
{
    if( !should_be ){
	std::cout << "Warning: " + message << std::endl;
    }
    return;
}

void check_and_exit_if_not( const bool &should_be, const std::string &message )
{
    if( !should_be ){
	std::cout << "Warning: " + message << std::endl;
	std::cout << "Aborting." << std::endl;
	exit( EXIT_FAILURE );
    }
    return;
}
