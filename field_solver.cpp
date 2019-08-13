#include "field_solver.h"

Field_solver::Field_solver( Spatial_mesh &spat_mesh,
                            Inner_regions_manager &inner_regions )
{
    nx = spat_mesh.x_n_nodes;
    ny = spat_mesh.y_n_nodes;
    nz = spat_mesh.z_n_nodes;
    dx = spat_mesh.x_cell_size;
    dy = spat_mesh.y_cell_size;
    dz = spat_mesh.z_cell_size;

    allocate_current_next_phi();
}
/*-------------------------------------------------------------*/
Field_solver::Field_solver( Spatial_mesh &spat_mesh,
                            Inner_regions_manager &inner_regions, 
                            Config &conf )
{
    nx = spat_mesh.x_n_nodes;
    ny = spat_mesh.y_n_nodes;
    nz = spat_mesh.z_n_nodes;
    dx = spat_mesh.x_cell_size;
    dy = spat_mesh.y_cell_size;
    dz = spat_mesh.z_cell_size;

    allocate_current_next_phi();

    solving_method = conf.field_solver_config_part.solving_method;
    abs_tolerance = conf.field_solver_config_part.abs_tolerance;
    rel_tolerance = conf.field_solver_config_part.rel_tolerance;
    max_iterations = conf.field_solver_config_part.max_iterations;
}

Field_solver::Field_solver( Spatial_mesh &spat_mesh,
                            Inner_regions_manager &inner_regions, 
                            hid_t h5_field_solver_group )
{
    nx = spat_mesh.x_n_nodes;
    ny = spat_mesh.y_n_nodes;
    nz = spat_mesh.z_n_nodes;
    dx = spat_mesh.x_cell_size;
    dy = spat_mesh.y_cell_size;
    dz = spat_mesh.z_cell_size;

    allocate_current_next_phi();

    herr_t status;

    char *char_method = new char[128];
    status = H5LTget_attribute_string( h5_field_solver_group, "./",
                                       "solving_method", char_method );
    solving_method = std::string( char_method );
    delete [] char_method;
    hdf5_status_check( status );    

    status = H5LTget_attribute_double( h5_field_solver_group, "./",
				    "abs_tolerance", &abs_tolerance ); 
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_field_solver_group, "./",
				    "rel_tolerance", &rel_tolerance ); 
    hdf5_status_check( status );
    status = H5LTget_attribute_int( h5_field_solver_group, "./",
				    "max_iterations", &max_iterations ); 
    hdf5_status_check( status );
}

void Field_solver::write_to_file( hid_t h5_field_solver_group )
{
    hid_t group_id;
    herr_t status;
    int single_element = 1;
    std::string hdf5_groupname = "/FieldSolver";
    group_id = H5Gcreate( h5_field_solver_group, hdf5_groupname.c_str(),
			  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); hdf5_status_check( group_id );

    status = H5LTset_attribute_double( h5_field_solver_group, hdf5_groupname.c_str(),
				       "abs_tolerance", &abs_tolerance, single_element ); 
    hdf5_status_check( status );
    status = H5LTset_attribute_double( h5_field_solver_group, hdf5_groupname.c_str(),
				       "rel_tolerance", &rel_tolerance, single_element ); 
    hdf5_status_check( status );
    status = H5LTset_attribute_int( h5_field_solver_group, hdf5_groupname.c_str(),
				    "max_iterations", &max_iterations, single_element ); 
    hdf5_status_check( status );
    status = H5LTset_attribute_string( h5_field_solver_group,
                                       hdf5_groupname.c_str(),
                                       "solving_method", solving_method.c_str() );
    hdf5_status_check( status );
    status = H5Gclose(group_id); 
    hdf5_status_check( status );
}

void Field_solver::hdf5_status_check( herr_t status )
{
    if( status < 0 ){
        std::cout << "Something went wrong while writing Field_solver "
                  << solving_method << "."
                  << "Aborting." << std::endl;
        exit( EXIT_FAILURE );
    }
}

/*-------------------------------------------------------------*/

void Field_solver::allocate_current_next_phi()
{
    phi_current.resize( boost::extents[nx][ny][nz] );
    phi_next.resize( boost::extents[nx][ny][nz] );
}

void Field_solver::eval_potential( Spatial_mesh &spat_mesh,
                                   Inner_regions_manager &inner_regions )
{
    solve_poisson_eqn_Jacobi( spat_mesh, inner_regions );
}
/*---------------------------------------------------------------*/
void Field_solver::prepare_linear_system( Spatial_mesh &spat_mesh,
                                          Inner_regions_manager &inner_regions )
{
    std::cout << "Preparation of linear system..." << std::endl;
    // creation of vector of inner-region and boundary nodes
    // addition of inner nodes
    ind_phi = std::vector<int>(nx*ny*nz, 0);
    for ( auto &region : inner_regions.regions)
        for ( auto &inner_node : region.inner_nodes)
            ind_phi[inner_node.x + inner_node.y*nx + inner_node.z*nx*ny] = -1;
    // addition of boundary nodes
    for ( int j = 0; j < ny; j++ )
        for ( int l = 0; l < nz; ++l ){
            ind_phi[0 + j*nx + l*nx*ny] = -1;
            ind_phi[(nx-1) + j*nx + l*nx*ny] = -1;
        }
    //
    for ( int i = 0; i < nx; i++ ) 
        for ( int l = 0; l < nz; ++l ){ 
            ind_phi[i + 0*nx + l*nx*ny] = -1;
            ind_phi[i + (ny-1)*nx + l*nx*ny] = -1;
        }
    //
    for ( int i = 0; i < nx; i++ ) 
        for ( int j = 0; j < ny; ++j ){ 
            ind_phi[i + j*nx + 0*nx*ny] = -1;
            ind_phi[i + j*nx + (nz-1)*nx*ny] = -1;
        }

    // creation of index vector ( ind_phi[l] == -1 means that related node is inner/boundary )
    int vec_size = 0;
    for (int i = 0; i < nx*ny*nz; ++i){
        if (ind_phi[i] != -1)
            ind_phi[i] = vec_size++;
    }

    x = cusp::array1d<double,cusp::device_memory> (vec_size, 0);
    b = cusp::array1d<double,cusp::device_memory> (vec_size, 0);    
    b_const = cusp::array1d<double,cusp::device_memory> (vec_size, 0);    
    b_var = cusp::array1d<double,cusp::device_memory> (vec_size, 0);    

    M = cusp::identity_operator<double, cusp::device_memory>(vec_size, vec_size);

    // matrix preparation
    cusp::coo_matrix<int, double, cusp::host_memory> A(vec_size, vec_size, 7*vec_size);
    int nd_ind; //index of current node
    int matr_elem = 0; //number of current sparce matrix element
    // indeces for access to neighbourgh nodes
    int nd_ind_i_p;
    int nd_ind_i_m;
    int nd_ind_j_p;
    int nd_ind_j_m;
    int nd_ind_l_p;
    int nd_ind_l_m;

    double dxdx = dx*dx;
    double dydy = dy*dy;
    double dzdz = dz*dz;

    init_current_phi_from_spat_mesh_phi( spat_mesh ); //phi_current is filling by zeros here (fixed)

    for (int l = 0; l < nz; ++l)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i){
                nd_ind = i + j*nx + l*nx*ny;
                if (ind_phi[nd_ind] != -1){

                    nd_ind_i_p = nd_ind + 1;
                    nd_ind_i_m = nd_ind - 1;
                    nd_ind_j_p = nd_ind + 1*nx;
                    nd_ind_j_m = nd_ind - 1*nx;
                    nd_ind_l_p = nd_ind + 1*ny*nx;
                    nd_ind_l_m = nd_ind - 1*ny*nx;

                    A.row_indices[matr_elem] = ind_phi[nd_ind];
                    A.column_indices[matr_elem] = ind_phi[nd_ind];   //A(ind_phi[nd_ind], ind_phi[nd_ind]) = -2/dxdx - 2/dydy - 2/dzdz;
                    A.values[matr_elem] = -2/dxdx - 2/dydy - 2/dzdz;
                    matr_elem++;
                    if (ind_phi[nd_ind_i_m] != -1){
                        A.row_indices[matr_elem] = ind_phi[nd_ind];
                        A.column_indices[matr_elem] = ind_phi[nd_ind_i_m];   //A(ind_phi[nd_ind], ind_phi[nd_ind_i_m]) = 1/dxdx;
                        A.values[matr_elem] = 1/dxdx;
                        matr_elem++;
                    } else {
                        b_const[ind_phi[nd_ind]] -= phi_current[i-1][j][l]/dxdx;//phi[nd_ind_i_m]/dxdx;
                    }

                    if (ind_phi[nd_ind_i_p] != -1){
                        A.row_indices[matr_elem] = ind_phi[nd_ind];
                        A.column_indices[matr_elem] = ind_phi[nd_ind_i_p];   //A(ind_phi[nd_ind], ind_phi[nd_ind_i_p]) = 1/dxdx;
                        A.values[matr_elem] = 1/dxdx;
                        matr_elem++;
                    } else {
                        b_const[ind_phi[nd_ind]] -= phi_current[i+1][j][l]/dxdx;//phi[nd_ind_i_p]/dxdx;
                    }

                    if (ind_phi[nd_ind_j_m] != -1){
                        A.row_indices[matr_elem] = ind_phi[nd_ind];
                        A.column_indices[matr_elem] = ind_phi[nd_ind_j_m];   //A(ind_phi[nd_ind], ind_phi[nd_ind_j_m]) = 1/dydy;
                        A.values[matr_elem] = 1/dydy;
                        matr_elem++;
                    } else {
                        b_const[ind_phi[nd_ind]] -= phi_current[i][j-1][l]/dydy;
                    }

                    if (ind_phi[nd_ind_j_p] != -1){
                        A.row_indices[matr_elem] = ind_phi[nd_ind];
                        A.column_indices[matr_elem] = ind_phi[nd_ind_j_p];   //A(ind_phi[nd_ind], ind_phi[nd_ind_j_p]) = 1/dydy;
                        A.values[matr_elem] = 1/dydy;
                        matr_elem++;
                    } else {
                        b_const[ind_phi[nd_ind]] -= phi_current[i][j+1][l]/dydy;
                    }

                    if (ind_phi[nd_ind_l_m] != -1){
                        A.row_indices[matr_elem] = ind_phi[nd_ind];
                        A.column_indices[matr_elem] = ind_phi[nd_ind_l_m];   //A(ind_phi[nd_ind], ind_phi[nd_ind_l_m]) = 1/dzdz;
                        A.values[matr_elem] = 1/dzdz;
                        matr_elem++;
                    } else {
                        b_const[ind_phi[nd_ind]] -= phi_current[i][j][l-1]/dzdz;
                    }

                    if (ind_phi[nd_ind_l_p] != -1){
                        A.row_indices[matr_elem] = ind_phi[nd_ind];
                        A.column_indices[matr_elem] = ind_phi[nd_ind_l_p];   //A(ind_phi[nd_ind], ind_phi[nd_ind_l_p]) = 1/dzdz;
                        A.values[matr_elem] = 1/dzdz;
                        matr_elem++;
                    } else {
                        b_const[ind_phi[nd_ind]] -= phi_current[i][j][l+1]/dzdz;
                    }
                }
            }
    A.resize(vec_size, vec_size, matr_elem);
    A_d = A;
    if ( solving_method.compare("Jacobi") == 0){
       std::cout << "Smoother preparation..." << std::endl;
       S = cusp::relaxation::jacobi<double, cusp::device_memory>(A_d);
       std::cout << "Smoother preparation done" << std::endl;
       r = cusp::array1d<double, cusp::device_memory>(A.num_rows);
    }
}

void Field_solver::solve_linear_system()
{
    std::cout << solving_method << std::endl;
    if (solving_method.compare("CG") == 0)
        solve_CG();
    else if ( solving_method.compare("CR") == 0)
        solve_CR();
    else if (solving_method.compare("BCG") == 0)
        solve_BCG();
    else if ( solving_method.compare("Jacobi") == 0)
        solve_Jacobi();
}

void Field_solver::solve_CG()
{
    cusp::monitor<double> monitor(b, max_iterations, rel_tolerance, abs_tolerance, true);  
    cusp::krylov::cg(A_d, x, b, monitor, M);
}

void Field_solver::solve_CR()
{
    cusp::monitor<double> monitor(b, max_iterations, rel_tolerance, abs_tolerance, true);  
    cusp::krylov::cr(A_d, x, b, monitor, M);
}

void Field_solver::solve_BCG()
{
    int i;
}


void Field_solver::solve_Jacobi()
{
    std::cout << "cusp::multiply(A_d,x,r)..." << std::endl;
    cusp::multiply(A_d,x,r);
    std::cout << "cusp::blas::axpy(b,r,-1)" << std::endl;
    cusp::blas::axpy(b,r,-1);
    std::cout << "cusp::monitor<double> monitor" << std::endl;
    cusp::monitor<double> monitor(b, max_iterations, rel_tolerance, abs_tolerance, true);  
    while( !monitor.finished(r) ){
        S(A_d, b, x);
        cusp::multiply(A_d,x,r);
        cusp::blas::axpy(b,r,-1);
        ++monitor;
    }
          
}
/*---------------------------------------------------------------*/

void Field_solver::solve_poisson_eqn_Jacobi( Spatial_mesh &spat_mesh,
                                             Inner_regions_manager &inner_regions )
{
    double time = omp_get_wtime();
    init_current_phi_from_spat_mesh_phi( spat_mesh ); //phi_current is filling by zeros here (fixed)
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int l = 0; l < nz; l++)
                if (ind_phi[i + j*nx+ l*nx*ny] != -1){
                    ind_phi[i + j*nx+ l*nx*ny];
                    x[ind_phi[i + j*nx + l*nx*ny]] = phi_current[i][j][l];
                    b_var[ind_phi[i + j*nx + l*nx*ny]] = -4*M_PI*spat_mesh.charge_density[i][j][l];
                }
    cusp::blas::axpby(b_const, b_var, b, 1, 1);
    //cusp::monitor<double> monitor(b, 150, 1e-2, 0, true);  
    //cusp::krylov::cg(A_d, x, b, monitor, M);
    solve_linear_system(); // solve A_d*x = b

    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int l = 0; l < nz; l++)
                phi_current[i][j][l] = x[ind_phi[i + j*nx + l*nx*ny]];

    transfer_solution_to_spat_mesh( spat_mesh );
    std::cout << "Field solver time: " << omp_get_wtime() - time << std::endl;
    return;
}


void Field_solver::init_current_phi_from_spat_mesh_phi( Spatial_mesh &spat_mesh )
{
    phi_current.assign( spat_mesh.potential.data(),
                        spat_mesh.potential.data() + spat_mesh.potential.num_elements() );
    return;
}


void Field_solver::transfer_solution_to_spat_mesh( Spatial_mesh &spat_mesh )
{
    spat_mesh.potential.assign( phi_current.data(),
                                phi_current.data() + phi_current.num_elements() );
}


void Field_solver::eval_fields_from_potential( Spatial_mesh &spat_mesh )
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    int nz = spat_mesh.z_n_nodes;
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    double dz = spat_mesh.z_cell_size;
    boost::multi_array<double, 3> &phi = spat_mesh.potential;
    double ex, ey, ez;
    //
    for ( int i = 0; i < nx; i++ ) {
        for ( int j = 0; j < ny; j++ ) {
            #pragma omp parallel for
            for ( int k = 0; k < nz; k++ ) {
                if ( i == 0 ) {
                    ex = - boundary_difference( phi[i][j][k], phi[i+1][j][k], dx );
                } else if ( i == nx-1 ) {
                    ex = - boundary_difference( phi[i-1][j][k], phi[i][j][k], dx );
                } else {
                    ex = - central_difference( phi[i-1][j][k], phi[i+1][j][k], dx );
                }

                if ( j == 0 ) {
                    ey = - boundary_difference( phi[i][j][k], phi[i][j+1][k], dy );
                } else if ( j == ny-1 ) {
                    ey = - boundary_difference( phi[i][j-1][k], phi[i][j][k], dy );
                } else {
                    ey = - central_difference( phi[i][j-1][k], phi[i][j+1][k], dy );
                }

                if ( k == 0 ) {
                    ez = - boundary_difference( phi[i][j][k], phi[i][j][k+1], dz );
                } else if ( k == nz-1 ) {
                    ez = - boundary_difference( phi[i][j][k-1], phi[i][j][k], dz );
                } else {
                    ez = - central_difference( phi[i][j][k-1], phi[i][j][k+1], dz );
                }

                spat_mesh.electric_field[i][j][k] = vec3d_init( ex, ey, ez );
            }
        }
    }

    return;
}

double Field_solver::central_difference( double phi1, double phi2, double dx )
{
    return ( (phi2 - phi1) / ( 2.0 * dx ) );
}

double Field_solver::boundary_difference( double phi1, double phi2, double dx )
{
    return ( (phi2 - phi1) / dx );
}


Field_solver::~Field_solver()
{
    // delete phi arrays?
}
