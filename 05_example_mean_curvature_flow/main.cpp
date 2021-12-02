#include <igl/read_triangle_mesh.h>
#include <igl/cotmatrix.h>
#include <igl/boundary_loop.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>

#include <igl/opengl/glfw/Viewer.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>

#include <iostream>
#include <vector>

#include <mg_data.h>
#include <mg_precompute.h>
#include <normalize_unit_area.h>
#include <min_quad_with_fixed_mg.h>

#include <chrono>

Eigen::MatrixXd V,U;
Eigen::MatrixXi F;
Eigen::SparseMatrix<double> L, BL;
igl::opengl::glfw::Viewer viewer;
std::vector<mg_data> mg;


int main(int argc, char *argv[])
{
	using namespace Eigen;
	using namespace std;

	/* Timer */
	std::chrono::high_resolution_clock::time_point t0, t1, t2, t3;
	std::chrono::duration<double> duration;

	// load mesh
	igl::read_triangle_mesh("../../meshes/beard_man.obj", V, F);
	normalize_unit_area(V,F);
	cout << "original mesh: |V| " << V.rows() << ", |F|: " << F.rows() << endl;


	// construct multigrid hierarchy
	t0 = std::chrono::high_resolution_clock::now();
	int min_coarsest_nV = 500;
	float coarsening_ratio = 0.25;
	int decimation_type = 1;
	mg_precompute(V,F,coarsening_ratio, min_coarsest_nV, decimation_type, mg);
	t1 = std::chrono::high_resolution_clock::now();
	duration = t1 - t0;
	std::cout << "[1] Build the hierarchy in " << duration.count() << "seconds \n";

	U = V;
	igl::cotmatrix(V,F,L);

	/* For BiLaplacian system */
	bool isBilaplacianSystem = true;

	const auto &key_down = [&isBilaplacianSystem, &t0, &t1, &duration](igl::opengl::glfw::Viewer &viewer,unsigned char key,int mod)->bool
	{
		switch(key)
		{
		case 'r':
		case 'R':
			U = V;
			break;
		case ' ':
		{
			// mean curvature flow [Kazhdan et al. 2012]
			// mg parameters
			double delta = 0.01;
			int maxIter = 20;
			double tolerance = 1e-16;
			double mg_tol = 5e-7;

			
			

			// save previous mesh
			MatrixXd Upre = U;

			// compute linear system
			SparseMatrix<double> M, MInv;
			igl::massmatrix(U, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
			if (isBilaplacianSystem) {
				igl::invert_diag(M, MInv);
				BL = L * MInv * L;
			}
			SparseMatrix<double> LHS;
			if (isBilaplacianSystem) {
				std::cout << "Working with a Bilaplacian system \n";
				LHS = M - delta * BL;
			}
			else {
				std::cout << "Working with a Laplacian system \n";
				LHS = M - delta * L;
			}
			MatrixXd RHS = M*U;

			// mg solve
			t0 = std::chrono::high_resolution_clock::now();
			min_quad_with_fixed_mg_data solverData;
			SimplicialLDLT<SparseMatrix<double>> coarseSolver;
			min_quad_with_fixed_mg_precompute(LHS, solverData, mg, coarseSolver);
			t1 = std::chrono::high_resolution_clock::now();
			duration = t1 - t0;	
			std::cout << "[2] Precompute in " << duration.count() << "seconds \n";
			vector<double> rHis;

			t0 = std::chrono::high_resolution_clock::now();
			min_quad_with_fixed_mg_solve(solverData, RHS, Upre, coarseSolver, mg_tol, mg, U, rHis);
			t1 = std::chrono::high_resolution_clock::now();
			duration = t1 - t0;
			std::cout << "[3] Solve in " << duration.count() << "seconds \n";

			// rescale output 
			normalize_unit_area(U,F);
			break;
		}
		default:
			return false;
		}
		// Send new positions, update normals, recenter
		viewer.data().set_vertices(U);
		viewer.data().compute_normals();
		viewer.core().align_camera_center(U,F);
		return true;
	};

	// Use original normals as pseudo-colors
	MatrixXd N;
	igl::per_vertex_normals(V,F,N);
	MatrixXd C = N.rowwise().normalized().array()*0.5+0.5;

	// Initialize smoothing with base mesh
	viewer.data().set_mesh(U, F);
	viewer.data().set_colors(C);
	viewer.callback_key_down = key_down;

	// set background color
	Vector4f backColor;
	backColor << 208/255., 237/255., 227/255., 1.;
	viewer.core().background_color = backColor;

	// not showing edges
	viewer.data().show_lines = false;

	cout<<"Press [space] to smooth."<<endl;;
	cout<<"Press [r] to reset."<<endl;;
	return viewer.launch();
}
