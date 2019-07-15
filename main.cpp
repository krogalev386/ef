#define BOOST_BIND_NO_PLACEHOLDERS
#include <iostream>
#include <regex>
#include <string>
#include <hdf5.h>
#include "config.h"
#include "domain.h"
#include "parse_cmd_line.h"
#include <omp.h>

void construct_domain( std::string config_or_h5_file,
		       Domain **dom,
		       bool *continue_from_h5 );
void extract_filename_prefix_and_suffix_from_h5filename( std::string h5_file,
							 std::string *prefix,
							 std::string *suffix );


int main( int argc, char *argv[] )
{
    std::string config_or_h5_file;
    parse_cmd_line( argc, argv, config_or_h5_file );

    bool continue_from_h5 = false;
    Domain *dom = NULL;
    construct_domain( config_or_h5_file, &dom, &continue_from_h5 );

    double time = omp_get_wtime();

    if( continue_from_h5 ){
	dom->continue_pic_simulation();
    } else {
	dom->start_pic_simulation();
    }

    std::cout << omp_get_wtime() - time << std::endl;
    
    // finalize_whatever_left
    delete dom;
    return 0;
}


void construct_domain( std::string config_or_h5_file,
		       Domain **dom,
		       bool *continue_from_h5 )
{
    std::string extension =
	config_or_h5_file.substr( config_or_h5_file.find_last_of(".") + 1 );

    if ( extension == "h5" ){
	hid_t h5file_id = H5Fopen( config_or_h5_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
	if( h5file_id < 0 ){
	    std::cout << "Can't open file: " << config_or_h5_file << std::endl;
	    exit( EXIT_FAILURE );
	}
	
	std::string filename_prefix, filename_suffix;
	extract_filename_prefix_and_suffix_from_h5filename( config_or_h5_file,
							    &filename_prefix,
							    &filename_suffix );
	*dom = new Domain( h5file_id );
	(*dom)->set_output_filename_prefix_and_suffix( filename_prefix, filename_suffix );
	*continue_from_h5 = true;
    } else {    
	Config conf( config_or_h5_file );
	conf.print();	
	*dom = new Domain( conf );
    }
}


void extract_filename_prefix_and_suffix_from_h5filename( std::string h5_file,
							 std::string *prefix,
							 std::string *suffix )
{
    std::regex rgx("[0-9]{7}");
    std::smatch match;
    std::regex_search( h5_file, match, rgx );
    if( match.size() == 1 ){
	*prefix = h5_file.substr( 0, match.position(0) );
	*suffix = h5_file.substr( match.position(0) + 7 );	    
    } else {
	std::cout << "Can't identify filename prefix and suffix in " << h5_file << std::endl;
	std::cout << "Aborting." << std::endl;
	exit( EXIT_FAILURE );
    }    
}
