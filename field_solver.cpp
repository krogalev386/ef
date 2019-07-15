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

void Field_solver::solve_poisson_eqn_Jacobi( Spatial_mesh &spat_mesh,
                                             Inner_regions_manager &inner_regions )
{
    max_Jacobi_iterations = 20;//2000;
    int iter;
    
    //CUSP test
    // works only for cuboid space
    cusp::gallery::poisson7pt(A,(nx-2),(ny-2),(nz-2)); //initialization of system matrix
    x = cusp::array1d<double, cusp::host_memory>((nx-2)*(ny-2)*(nz-2),0);
    b = cusp::array1d<double, cusp::host_memory>((nx-2)*(ny-2)*(nz-2),1);
    //monitor = cusp::monitor<double>(b, 100, 1e-3, 0, false);
    cusp::monitor<double> monitor(b, 1000, 1e-4, 0, false);
    M = cusp::identity_operator<double, cusp::host_memory>(A.num_rows, A.num_cols);


    init_current_phi_from_spat_mesh_phi( spat_mesh ); //phi_current is filling by zeros here (fixed)


    set_phi_next_at_boundaries();

    for (int i = 0; i < (nx-2); i++)
        for (int j = 0; j < (ny-2); j++)
            for (int l = 0; l < (nz-2); l++){
                x[i + j*(nx-2) + l*(nx-2)*(ny-2)] = phi_current[i+1][j+1][l+1];
                b[i + j*(nx-2) + l*(nx-2)*(ny-2)] = 4*M_PI*dx*dx*spat_mesh.charge_density[i+1][j+1][l+1];
            }

    //std::cout << std::endl;

    cusp::krylov::cr(A, x, b, monitor, M);

    for (int i = 0; i < (nx-2); i++)
        for (int j = 0; j < (ny-2); j++)
            for (int l = 0; l < (nz-2); l++){
                phi_current[i+1][j+1][l+1] = x[i + j*(nx-2) + l*(nx-2)*(ny-2)];
            }
    
    /*for( iter = 0; iter < max_Jacobi_iterations; ++iter ){
        single_Jacobi_iteration( spat_mesh, inner_regions );
        if ( iterative_Jacobi_solutions_converged() ) {
            break;
        }
        set_phi_next_as_phi_current();
    }
    if ( iter == max_Jacobi_iterations ){
        printf("WARING: potential evaluation did't converge after max iterations!\n");
    }*/
    set_phi_next_as_phi_current(); // here phi_next becomes zeros, phi_current don't (fixed)

    /*for (int i = 0; i < (nx-2); i++)
        for (int j = 0; j < (ny-2); j++)
            for (int l = 0; l < (nz-2); l++){
                if(phi_current[i+1][j+1][l+1]!=0)
                    std::cout << phi_current[i+1][j+1][l+1];
            }*/

    transfer_solution_to_spat_mesh( spat_mesh );

    return;
}


void Field_solver::init_current_phi_from_spat_mesh_phi( Spatial_mesh &spat_mesh )
{
    phi_current.assign( spat_mesh.potential.data(),
                        spat_mesh.potential.data() + spat_mesh.potential.num_elements() );
    return;
}

void Field_solver::single_Jacobi_iteration( Spatial_mesh &spat_mesh,
					    Inner_regions_manager &inner_regions )
{
    set_phi_next_at_boundaries();
    compute_phi_next_at_inner_points( spat_mesh );
    set_phi_next_at_inner_regions( inner_regions );
}

void Field_solver::set_phi_next_at_boundaries()
{
    for ( int j = 0; j < ny; j++ ) {
        for ( int k = 0; k < nz; k++ ) {
            phi_next[0][j][k] = phi_current[0][j][k];
            phi_next[nx-1][j][k] = phi_current[nx-1][j][k];
        }
    }
    //
    for ( int i = 0; i < nx; i++ ) {
        for ( int k = 0; k < nz; k++ ) {
            phi_next[i][0][k] = phi_current[i][0][k];
            phi_next[i][ny-1][k] = phi_current[i][ny-1][k];
        }
    }
    //
    for ( int i = 0; i < nx; i++ ) {
        for ( int j = 0; j < ny; j++ ) {
            phi_next[i][j][0] = phi_current[i][j][0];
            phi_next[i][j][nz-1] = phi_current[i][j][nz-1];
        }
    }
}


void Field_solver::compute_phi_next_at_inner_points( Spatial_mesh &spat_mesh )
{
    double dxdxdydy = dx * dx * dy * dy;
    double dxdxdzdz = dx * dx * dz * dz;
    double dydydzdz = dy * dy * dz * dz;
    double dxdxdydydzdz = dx * dx * dy * dy * dz * dz;
    double denom = 2 * ( dxdxdydy + dxdxdzdz + dydydzdz );
    
    /*// Parallelization parameters
    int i, j, k;
    //double start_time = omp_get_wtime();
    #pragma omp parallel for private(j, k)
    for ( i = 1; i < nx - 1; i++ ) {
        for ( j = 1; j < ny - 1; j++ ) {
            for ( k = 1; k < nz - 1; k++ ) {
                phi_next[i][j][k] =
                    ( phi_current[i-1][j][k] + phi_current[i+1][j][k] ) * dydydzdz;
                phi_next[i][j][k] = phi_next[i][j][k] +
                    ( phi_current[i][j-1][k] + phi_current[i][j+1][k] ) * dxdxdzdz;
                phi_next[i][j][k] = phi_next[i][j][k] +
                    ( phi_current[i][j][k-1] + phi_current[i][j][k+1] ) * dxdxdydy;
                // Delta phi = - 4 * pi * rho
                phi_next[i][j][k] = phi_next[i][j][k] +
                    4.0 * M_PI * spat_mesh.charge_density[i][j][k] * dxdxdydydzdz;
                phi_next[i][j][k] = phi_next[i][j][k] / denom;
            }
        }
    }*/
    //std::cout << "Time: " << omp_get_wtime() - start_time << std::endl;
}

/*void Field_solver::compute_phi_next_at_inner_points( Spatial_mesh &spat_mesh )
{
    double dxdxdydy = dx * dx * dy * dy;
    double dxdxdzdz = dx * dx * dz * dz;
    double dydydzdz = dy * dy * dz * dz;
    double dxdxdydydzdz = dx * dx * dy * dy * dz * dz;
    double denom = 2 * ( dxdxdydy + dxdxdzdz + dydydzdz );
    
    // Parallelization parameters
    int omp_thread, omp_n_of_threads;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_n_of_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_process_rank);
    //#pragma omp parallel for shared(phi_next, nx, ny, nz) private(i_first, i_last, i, j, k)
    int block_size = ceil(nx/opm_n_of_threads);
    int i_first, i_last;
    if (mpi_n_of_proc != 1){
        if (mpi_process_rank == 0) { 
            i_first = 1;
            i_last = block_size - 1;
        } 
        else if (mpi_process_rank == (mpi_n_of_proc - 1)) {
            i_first = mpi_process_rank*block_size;
            i_last = nx - 2;
        } 
        else {
            i_first = mpi_process_rank*block_size;
            i_last = i_first + block_size - 1;
        }
    } else {
        i_first = 1;
        i_last = nx - 2;
    }
    
        
    for ( int i = i_first; i <= i_last; i++ ) {
        for ( int j = 1; j < ny - 1; j++ ) {
            for ( int k = 1; k < nz - 1; k++ ) {
                phi_next[i][j][k] =
                    ( phi_current[i-1][j][k] + phi_current[i+1][j][k] ) * dydydzdz;
                phi_next[i][j][k] = phi_next[i][j][k] +
                    ( phi_current[i][j-1][k] + phi_current[i][j+1][k] ) * dxdxdzdz;
                phi_next[i][j][k] = phi_next[i][j][k] +
                    ( phi_current[i][j][k-1] + phi_current[i][j][k+1] ) * dxdxdydy;
                // Delta phi = - 4 * pi * rho
                phi_next[i][j][k] = phi_next[i][j][k] +
                    4.0 * M_PI * spat_mesh.charge_density[i][j][k] * dxdxdydydzdz;
                phi_next[i][j][k] = phi_next[i][j][k] / denom;
            }
            //MPI_Bcast(&phi_next[i][j][0], nz, MPI_DOUBLE, mpi_process_rank, MPI_COMM_WORLD);
        }
    }
}*/

void Field_solver::set_phi_next_at_inner_regions( Inner_regions_manager &inner_regions )
{
    for( auto &reg : inner_regions.regions ){
        for( auto &node : reg.inner_nodes ){
            // todo: mark nodes at edge during construction
            // if (!node.at_domain_edge( nx, ny, nz )) {
            phi_next[node.x][node.y][node.z] = reg.potential;
            // }
        }
    }
}


bool Field_solver::iterative_Jacobi_solutions_converged()
{
    // todo: bind tol to config parameters
    //abs_tolerance = std::max( dx * dx, std::max( dy * dy, dz * dz ) ) / 5;
    abs_tolerance = 1.0e-5;//1.0e-5;
    rel_tolerance = 1.0e-12;//1.0e-12;
    double diff;
    double rel_diff;
    //double tol;
    //
    for ( int i = 0; i < nx; i++ ) {
        for ( int j = 0; j < ny; j++ ) {
            for ( int k = 0; k < nz; k++ ) {
                diff = fabs( phi_next[i][j][k] - phi_current[i][j][k] );
                rel_diff = diff / fabs( phi_current[i][j][k] );
                if ( diff > abs_tolerance || rel_diff > rel_tolerance ){
                    return false;
                }
            }
        }
    }
    return true;
}


void Field_solver::set_phi_next_as_phi_current()
{
    // Looks like straightforward assignment
    //   phi_next = phi_current
    // would result in copy.
    // Hopefully, it could be avoided with std::swap
    std::swap( phi_current, phi_next );
}

void Field_solver::transfer_solution_to_spat_mesh( Spatial_mesh &spat_mesh )
{
    spat_mesh.potential.assign( phi_next.data(),
                                phi_next.data() + phi_next.num_elements() );
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
