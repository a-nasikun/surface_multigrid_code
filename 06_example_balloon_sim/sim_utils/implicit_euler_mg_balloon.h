#include <Eigen/Core>
#include <vector>
#include <iostream>

#include <igl/min_quad_with_fixed.h>
#include <igl/writeDMAT.h>

#include "MaterialModel.h"
#include "MeshConnectivity.h"
#include "ElasticShell.h"

#include <mg_precompute_block.h>
#include <mg_data.h>
#include <min_quad_with_fixed_mg.h>

#include <profc.h>

template <class SFF>
double implicit_euler_mg_balloon(const MeshConnectivity & mesh,
    const Eigen::SparseMatrix<double> & M,
    Eigen::MatrixXd & curPos,
    Eigen::VectorXd & qdot,
    const Eigen::VectorXd & fExt,
    const Eigen::VectorXi & bi,
    const Eigen::VectorXd & curEdgeDOFs,
    const MaterialModel<SFF> & mat,
    const double & dt,
    const Eigen::VectorXd & thicknesses,
    const std::vector<Eigen::Matrix2d> & abars,
    const std::vector<Eigen::Matrix2d> & bbars,
    std::vector<mg_data> & mg,
    double & mg_tolerance)
{
    using namespace std;
    Eigen::VectorXd qdot0 = qdot;
    Eigen::MatrixXd curPos0 = curPos;

    Eigen::VectorXd dx; // update of qdot i.e. dqot

    Eigen::VectorXd bc(bi.rows()); // place holder
    bc.setZero();
    Eigen::VectorXd Beq;
    Eigen::SparseMatrix<double> Aeq;

    //compute newton step direction
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;

    for (int i = 0; i < 10; i++) {

        Eigen::VectorXd G;
        std::vector<Eigen::Triplet<double>> hessian;

        double energy = ElasticShell<SFF>::elasticEnergy(mesh, curPos, curEdgeDOFs, mat, thicknesses, abars, bbars, ElasticShell<SFF>::EnergyTerm::ET_STRETCHING, &G, &hessian);
        Eigen::SparseMatrix<double> K(qdot.rows(), qdot.rows());
        K.setFromTriplets(hessian.begin(), hessian.end());

        Eigen::SparseMatrix<double> tmp_H;
        Eigen::VectorXd tmp_g;

        // dynamic solve
        // Eigen::SparseMatrix<double> I(K.rows(), K.rows());
        // I.setIdentity();
        tmp_H = M + dt * dt * K;
        tmp_g = M * (qdot - qdot0) + dt * G + dt * fExt;
        tmp_g = -tmp_g;

        {
            // PROFC_NODE("MG time");
            // mqwf multigrid
            min_quad_with_fixed_mg_data solverData;
            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> coarseSolver;
            vector<double> rHis;
            Eigen::VectorXd dxInit(tmp_H.rows());
            dxInit.setZero();
            min_quad_with_fixed_mg_precompute(tmp_H,solverData, mg, coarseSolver);
            min_quad_with_fixed_mg_solve(solverData, tmp_g, dxInit, coarseSolver,mg_tolerance, mg, dx, rHis);
            std::cout << "euler converge: " << tmp_g.transpose() * dx << std::endl;
        }

        //perform backtracking on the newton search direction
        //Guarantee that step is a descent step and that it will make sufficient decreate
        double alpha = 1;
        double p = 0.5;
        double c = 1e-8;


        auto f = [&](const Eigen::VectorXd & tmp_qdot) -> double {
            double E_total = 0;
            double Ek = 0.5 * (tmp_qdot - qdot0).transpose() * M * (tmp_qdot - qdot0);
            Eigen::MatrixXd newPos = curPos0;
            for (int j = 0; j < newPos.rows(); j++)
            {
                newPos.row(j) += dt * tmp_qdot.segment<3>(3 * j);
                E_total += newPos.row(j) * fExt.segment<3>(3 * j);
            }
            double V = ElasticShell<SFF>::elasticEnergy(mesh, newPos, curEdgeDOFs, mat, thicknesses, abars, bbars, ElasticShell<SFF>::EnergyTerm::ET_STRETCHING, NULL, NULL);
            E_total = E_total + Ek + V;

            return E_total;
        };

        double f0 = f(qdot);

        double s = f0 + c * tmp_g.transpose() * dx; // sufficient decrease

        while (alpha > 1e-8) {
   
            if (f(qdot + alpha * dx) <= s) {
                qdot += alpha * dx;
                break;
            }
            alpha *= p;
        }
        std::cout << "alpha: " << alpha << std::endl;

        // update curPos
        curPos = curPos0;
        for (int j = 0; j < curPos.rows(); j++) {
            curPos.row(j) += dt * qdot.segment<3>(3 * j);
        }
    }

    return 0;

}