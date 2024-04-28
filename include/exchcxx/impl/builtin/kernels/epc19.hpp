#pragma once
#include <cmath>

#include <exchcxx/impl/builtin/fwd.hpp>
#include <exchcxx/impl/builtin/constants.hpp>
#include <exchcxx/impl/builtin/kernel_type.hpp>
#include <exchcxx/impl/builtin/util.hpp>

#include <exchcxx/impl/builtin/kernels/screening_interface.hpp>



namespace ExchCXX {

template <>
struct kernel_traits< BuiltinEPC19 > :
  public lda_screening_interface< BuiltinEPC19 > {

  static constexpr bool is_lda  = false;
  static constexpr bool is_gga  = true;
  static constexpr bool is_mgga = false;
  static constexpr bool needs_laplacian = false;
  static constexpr bool is_kedf = false;

  static constexpr double dens_tol  = 1e-24;
  static constexpr double zeta_tol  = 1e-15;
  static constexpr double sigma_tol  = 1.000000000000004e-32;
  static constexpr double tau_tol = is_kedf ? 0.0 : 1e-20;

  static constexpr double a = 1.9; 
  static constexpr double b = 1.3; 
  static constexpr double c = 8.1;
  static constexpr double d = 1600.0;
  static constexpr double q = 8.2;
  static constexpr double pMass = 1836.152676;

  static constexpr bool is_hyb  = false;
  static constexpr double exx_coeff = 0.0;

  BUILTIN_KERNEL_EVAL_RETURN
    eval_exc_unpolar_impl( double rho, double sigma, double& eps ) {

  }

  BUILTIN_KERNEL_EVAL_RETURN
    eval_exc_vxc_unpolar_impl( double rho, double sigma, double& eps, double& vrho, double& vsigma ) {

  }

  BUILTIN_KERNEL_EVAL_RETURN
    eval_exc_polar_impl( double rho_a, double rho_b, double sigma_aa, double sigma_ab, double sigma_bb, double& eps ) {

  }

  BUILTIN_KERNEL_EVAL_RETURN
    eval_exc_vxc_polar_impl( double rho_e, double rho_p, double sigma_ee, double sigma_ep, double sigma_pp, double& eps, double& vrho_e, double& vrho_p, double& vsigma_ee, double& vsigma_ep, double& vsigma_pp ) {

    // lambda function that computes X and its derivatives
    auto computeX = [&](double rho_e, double rho_p, std::vector<double> & outvec) {
      // size output vector correctly
      if(outvec.size() != 6)
        outvec.resize(6);

      // denominator
      double de = a - b * safe_math::sqrt(rho_e * rho_p) + c * rho_e * rho_p;
      outvec[0] = 1.0 / de;

      // first-derivatives
      outvec[1] = ( 0.5 * b * safe_math::pow(rho_e,-0.5) * safe_math::sqrt(rho_p) - c * rho_p ) / (de * de);
      outvec[2] = ( 0.5 * b * safe_math::pow(rho_p,-0.5) * safe_math::sqrt(rho_e) - c * rho_e ) / (de * de);

      // second-derivatives
      double nu1 = -a * b / 4 * safe_math::pow(rho_e, -1.5) * safe_math::sqrt(rho_p);
      double nu2 =  3 * b * b / 4 / rho_e * rho_p;
      double nu3 = -9 * b * c / 4.0 * safe_math::pow(rho_e, -0.5) * safe_math::pow(rho_p, 1.5);
      double nu4 =  2 * c * c * rho_p * rho_p;
      double nu = nu1 + nu2 + nu3 + nu4;
      outvec[3] = nu / (de * de * de);  // ee

      // compute ep
      nu1 =  a * b / 4 * safe_math::pow(rho_e * rho_p, -0.5);
      nu2 = -a * c + b * b / 4;
      nu3 = -3.0 * b * c  / 4 * safe_math::sqrt(rho_e * rho_p);
      nu4 =  c * c * rho_e * rho_p;
      nu = nu1 + nu2 + nu3 + nu4;
      outvec[4] = nu / (de * de * de); // ep

      // compute pp
      nu1 = -a * b / 4 * safe_math::pow(rho_p, -1.5) * safe_math::sqrt(rho_e);
      nu2 =  3 * b * b / 4 / rho_p * rho_e;
      nu3 = -9 * b * c / 4.0 * safe_math::pow(rho_p, -0.5) * safe_math::pow(rho_e, 1.5);
      nu4 =  2 * c * c * rho_e * rho_e;
      nu = nu1 + nu2 + nu3 + nu4;
      outvec[5] = nu / (de * de * de);  // pp

    };

    // lambda function that computes Y0 and its derivatives
    auto computeY0 = [&](double rho_e, double rho_p, std::vector<double> & outvec) {
      // size output vector correctly
      if(outvec.size() != 6)
        outvec.resize(6);

      outvec[0] = rho_e * rho_p;
      outvec[1] = rho_p;
      outvec[2] = rho_e;

      outvec[3] = 0.0;
      outvec[4] = 1.0;
      outvec[5] = 0.0;

    };

    // lambda function that computes Y1
    auto computeY1 = [&](double rho_e, double rho_p, std::vector<double> & outvec) {
      // size output vector correctly
      if(outvec.size() != 6)
        outvec.resize(6);       
      
      outvec[0] = pMass * pMass * safe_math::pow(rho_e, -1.0 / 3) * safe_math::pow(rho_p, 2.0 / 3) / ((1 + pMass) * (1 + pMass));
      outvec[1] = -1.0 / 3 * outvec[0] / rho_e;
      outvec[2] =  2.0 / 3 * outvec[0] / rho_p;

      // second-order derivatives
      outvec[3] =  4.0 / 9 * outvec[0] / (rho_e * rho_e); 
      outvec[4] = -2.0 / 9 * outvec[0] / (rho_e * rho_p);
      outvec[5] =  0.0;
    };

    // lambda function that computes Y2
    auto computeY2 = [&](double rho_e, double rho_p, std::vector<double> & outvec) {
      // size output vector correctly
      if(outvec.size() != 3)
        outvec.resize(3);       

      outvec[0] =  2.0 * pMass * safe_math::pow(rho_e * rho_p, -1.0 / 3) / ((1 + pMass) * (1 + pMass));
      outvec[1] = -1.0 / 3 * outvec[0] / rho_e;
      outvec[2] = -1.0 / 3 * outvec[0] / rho_p;
    };

    // lambda function that computes Y3
    auto computeY3 = [&](double rho_e, double rho_p, std::vector<double> & outvec) {
      // size output vector correctly
      if(outvec.size() != 6)
        outvec.resize(6);

      outvec[0] = safe_math::pow(rho_e, 2.0 / 3) * safe_math::pow(rho_p, -1.0 / 3) / ((1 + pMass) * (1 + pMass));
      outvec[1] =  2.0 / 3 * outvec[0] / rho_e;
      outvec[2] = -1.0 / 3 * outvec[0] / rho_p;

      // second-order derivatives
      outvec[3] =  0.0;
      outvec[4] = -2.0 / 9 * outvec[0] / (rho_e * rho_p);
      outvec[5] =  4.0 / 9 * outvec[0] / (rho_p * rho_p);
    };

    // lambda function that computes Z
    auto computeZ = [&](double rho_e, double rho_p, std::vector<double> & outvec) {
      // size output vector correctly
      if(outvec.size() != 6)
        outvec.resize(6);       

      double factor = -1.0 * q / safe_math::pow(rho_e * rho_p, 1.0 / 6);
      outvec[0] = safe_math::exp(factor);
      outvec[1] = outvec[0] * q / (6.0 * rho_e * safe_math::pow(rho_e * rho_p, 1.0 / 6));
      outvec[2] = outvec[0] * q / (6.0 * rho_p * safe_math::pow(rho_e * rho_p, 1.0 / 6));

      // second-order derivatives
      outvec[3] = outvec[0] * q * q / (36 * rho_e * rho_e * safe_math::pow(rho_e * rho_p, 1.0 / 3)) - 7 * outvec[0] * q / (36 * rho_e * rho_e * safe_math::pow(rho_e * rho_p, 1.0 / 6));
      outvec[4] = outvec[0] * q * q / (36 * rho_e * rho_p * safe_math::pow(rho_e * rho_p, 1.0 / 3)) - outvec[0] * q / (36 * rho_e * rho_p * safe_math::pow(rho_e * rho_p, 1.0 / 6));
      outvec[5] = outvec[0] * q * q / (36 * rho_p * rho_p * safe_math::pow(rho_e * rho_p, 1.0 / 3)) - 7 * outvec[0] * q / (36 * rho_p * rho_p * safe_math::pow(rho_e * rho_p, 1.0 / 6));
    };

    // lambda function that compites Ge
    auto computeGe = [&](std::vector<double> & X, std::vector<double> & Y1, std::vector<double> & Z, std::vector<double> & outvec) {
      // size output vector correctly
      if(outvec.size() != 6)
        outvec.resize(6);       

      outvec[0] = -d * X[0] * Y1[0] * Z[0];
      outvec[1] = -d * (X[1] * Y1[0] * Z[0] + X[0] * Y1[1] * Z[0] + X[0] * Y1[0] * Z[1]);
      outvec[2] = -d * (X[2] * Y1[0] * Z[0] + X[0] * Y1[2] * Z[0] + X[0] * Y1[0] * Z[2]);

      // second-order derivatives
      outvec[3] = -d * (Y1[0] * X[3] * Z[0] + X[0] * Y1[3] * Z[0] + X[0] * Y1[0] * Z[3]
                  + 2 * X[1] * Y1[1] * Z[0] + 2 * X[1] * Y1[0] * Z[1] + 2 * X[0] * Y1[1] * Z[1]);

      outvec[4] = -d * (Y1[0] * X[4] * Z[0] + X[0] * Y1[4] * Z[0] + X[0] * Y1[0] * Z[4]
                  + X[1] * Y1[2] * Z[0] + X[2] * Y1[1] * Z[0] 
                  + X[1] * Y1[0] * Z[2] + X[2] * Y1[0] * Z[1]
                  + X[0] * Y1[2] * Z[1] + X[0] * Y1[1] * Z[2]);

      outvec[5] = 0.0;
      
    };

    // lambda function that computes F0
    auto computeF0 = [&](std::vector<double> & X, std::vector<double> & Y0, std::vector<double> & Y2, std::vector<double> & Z, double csigma, std::vector<double> & outvec) {
      // size output vector correctly
      if(outvec.size() != 3)
        outvec.resize(3);       

      outvec[0] = X[0] * Y0[0] + d * X[0] * Y2[0] * csigma * Z[0];
      outvec[1] = Y0[0] * X[1] + X[0] * Y0[1] + d * csigma * (Y2[0] * X[1] * Z[0] 
                + X[0] * Y2[1] * Z[0] + X[0] * Y2[0] * Z[1]);
      outvec[2] = Y0[0] * X[2] + X[0] * Y0[2] + d * csigma * (Y2[0] * X[2] * Z[0] 
                + X[0] * Y2[2] * Z[0] + X[0] * Y2[0] * Z[2]);
    };

    // lambda function that compites Gp
    auto computeGp = [&](std::vector<double> & X, std::vector<double> & Y3, std::vector<double> & Z, std::vector<double> & outvec) {
      // size output vector correctly
      if(outvec.size() != 6)
        outvec.resize(6);       

      outvec[0] = -d * X[0] * Y3[0] * Z[0];
      outvec[1] = -d * (X[1] * Y3[0] * Z[0] + X[0] * Y3[1] * Z[0] + X[0] * Y3[0] * Z[1]);
      outvec[2] = -d * (X[2] * Y3[0] * Z[0] + X[0] * Y3[2] * Z[0] + X[0] * Y3[0] * Z[2]);

      // second-order derivatives
      outvec[5] = -d * (Y3[0] * X[5] * Z[0] + X[0] * Y3[5] * Z[0] + X[0] * Y3[0] * Z[5]
                  + 2 * X[2] * Y3[2] * Z[0] + 2 * X[2] * Y3[0] * Z[2] + 2 * X[0] * Y3[2] * Z[2]);

      outvec[4] = -d * (Y3[0] * X[4] * Z[0] + X[0] * Y3[4] * Z[0] + X[0] * Y3[0] * Z[4]
                  + X[1] * Y3[2] * Z[0] + X[2] * Y3[1] * Z[0] 
                  + X[1] * Y3[0] * Z[2] + X[2] * Y3[0] * Z[1]
                  + X[0] * Y3[2] * Z[1] + X[0] * Y3[1] * Z[2]);

      outvec[3] = 0.0;
      
    };


    if (rho_e < 1e-15 or rho_p < 1e-15) {
      eps = 0.;
      vrho_e = 0.;
      vrho_p = 0.;
      vsigma_ee = 0.0; 
      vsigma_ep = 0.0; 
      vsigma_pp = 0.0; 
    } else{
        // X and derivatives
        std::vector<double> X_vec, Y0_vec, Y1_vec, Y2_vec, Y3_vec, Z_vec;
        // non-gradient pieces
        computeX( rho_e, rho_p, X_vec);
        computeY0(rho_e, rho_p, Y0_vec);
        computeY1(rho_e, rho_p, Y1_vec);
        computeY2(rho_e, rho_p, Y2_vec);
        computeY3(rho_e, rho_p, Y3_vec);
        computeZ( rho_e, rho_p, Z_vec);
        // compute Ge and Gp
        std::vector<double> F0, Ge, Gp;
        computeGe(X_vec, Y1_vec, Z_vec, Ge);
        computeGp(X_vec, Y3_vec, Z_vec, Gp);
        // compute F0
        computeF0(X_vec, Y0_vec, Y2_vec, Z_vec, sigma_ep, F0);
        eps  = -1.0 * (F0[0] - Ge[1] * sigma_ee - Gp[2] * sigma_pp) / (rho_e*rho_p);
        vrho_e =  -1.0 * (F0[1] - Ge[3] * sigma_ee - Gp[4] * sigma_pp);
        vrho_p =  -1.0 * (F0[2] - Ge[4] * sigma_ee - Gp[5] * sigma_pp);
        vsigma_ee = Ge[1];
        vsigma_ep = -1.0 * X_vec[0] * ( d * Y2_vec[0] * Z_vec[0]);
        vsigma_pp = Gp[2];
    }
  }


};

struct BuiltinEPC19 : detail::BuiltinKernelImpl< BuiltinEPC19 > {

  BuiltinEPC19( Spin p ) :
    detail::BuiltinKernelImpl< BuiltinEPC19 >(p) { }
  
  virtual ~BuiltinEPC19() = default;

};



} // namespace ExchCXX