////////////////////////////////////////////////////////////////////////////////////////////////////
//
//               R A N B O X 
//               -----------
//
// This program searches for overdensities in multi-dimensional data
// by first transforming the feature space into its copula by the integral transform on all
// variables. Optionally, principal component
// analysis or a correlated variables removal can be applied to reduce dimensionality.
// The algorithm searches the data for a multi-D box of dimensionality smaller than the original
// one, to locate the box where a test statistic connected to the highest local density is maximized.
// The expected background in the box is computed by considering the box volume, or by
// considering the amount of data in a sideband constructed around the
// box. An iteration of subspaces where the box is constructed scans
// the space of possible overdensities. Two methods are used to find
// the most promising region of space: either with the ZPL maximization
// or with the maximization of the ratio of observed and predicted events.
//
// This version of the code scans in random subspaces where Nvar intervals define the box. 
// A twin version of the code instead performs an incremental scan by considering an interval in two of the
// features, locating the Nbest more promising regions, and then incrementing iteratively the number
// of dimensions where an interval smaller than [0,1] is requested, until Nvarmax intervals have
// been imposed (see code RanBoxIter.cpp).
//
// T. Dorigo, 2019-2022
//  
//
//       To prepare this code for running in your working area, please do the following:
//
//       1) Save this file to the disk area you want to run it on
//
//       2) Change address of input data file and input and output directories:
//                    dataname = "/lustre/cmswork/dorigo/RanBox/datafiles/HEPMASS/...";
//                    dataname = "/lustre/cmswork/dorigo/RanBox/datafiles/MiniBooNE/...";
//                    dataname = "/lustre/cmswork/dorigo/RanBox/datafiles/Fraud/...";
//                    dataname = "/lustre/cmswork/dorigo/RanBox/datafiles/EEG/...";
//                    dataname = "/lustre/cmswork/dorigo/RanBox/datafiles/LHCOlympics/...";
//          to the correct paths where you stored data. Also change:
//      	  	      string outputPath = "/lustre/cmswork/dorigo/RanBox/Output";
//		              string asciiPath  = "/lustre/cmswork/dorigo/RanBox/Ascii";
//          to your Output and Ascii directories.
//
//		 3) To compile, execute the following command 
// 		      > g++ -g `root-config --libs --cflags` RanBox.cpp -o RanBox
//		 4) To run, try e.g.
// 		      > ./RanBox -Dat 4 -Nsi 1000 -Nba 9000 -Nva 10 -Alg 5
//          which generates 10% gaussian signal events (in Gaussian_dims features) on 
//          a flat background, and searches with a maximum of 10 variables defining the box,
//          initializing the box with kernel density estimate.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TF1.h"
#include "TMath.h"
#include "TROOT.h"
#include "TRandom.h"
#include "TRandom3.h"
#include "Riostream.h"
#include "TPrincipal.h" // for PCA 
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>  // to be able to read in data with commas

using namespace std;

#include "pca.C" // file created by previous iteration on same file

void X2Pmia(Double_t *x, Double_t *p) {
  for (Int_t i = 0; i < gNVariables; i++) {
    p[i] = 0;
    for (Int_t j = 0; j < gNVariables; j++) {
      p[i] += (x[j] - gMeanValues[j]) 
        * gEigenVectors[j *  gNVariables + i] / gSigmaValues[j];
    }
  }
}

// Constants
// ---------
static const int maxNtr        = 10000; 
static bool force_gaussians    = false;      // In gaussian toy gen, force to pick gaussian dims for subspace 
static const int ND            = 50;         // total number of considered space dimensions
static const int maxNvar       = ND;         // max dimensionality of search box
static const int maxEvents     = 100000;
static const double sqrt2      = sqrt(2.);
static const int maxClosest    = 20;
static const int N2var         = 105; // maxNvar*(maxNvar-1)/2;
static const double alpha      = 0.05;

// Control variables
// -----------------
static int RegFactor           = 1;          // But it could be larger if using useSB true, because of flukes
static int maxJumps            = 10;         // max number of jumps in each direction to search for new minima
static double maxBoxVolume     = 0.5;        // max volume of search box
static double syst             = 0.;         // systematics on tau!
static int maxGDLoops          = 100;        // max number of GD loops
static bool verbose            = false;      // control printouts on screen
static bool plots              = true;       // whether to plot distributions
static bool RemoveHighCorr     = false;      // high-correlation vars are removed, NSEL are kept among NAD
static bool Multivariate       = true;       // whether signal shows correlations among features
static bool debug              = false;      // couts toggle
static bool compactify         = false;      // If true, empty space in domain of features is removed
static bool fixed_gaussians    = true;       // In gaussian toy generations, controls shape of signal
static bool narrow_gaussians   = true;       // In gaussian toy generations, controls what kind of signal is injected
static int Gaussian_dims       = 15;
static double maxRho           = 0.2;
static double maxHalfMuRange   = 0.35;
static int NseedTrials         = 1;          // Number of repetition of same subspace search, for tests of clustering
static double shrinking_factor = 0.9;        // decreasing factor for steps in direction not previously taken
static double widening_factor  = 1.0;        // widening factor for steps in same direction
static double InitialLambda    = 0.5;        // step size in search
static double sidewidth        = 0.5;
static double D2kernel         = 0.04;

// Other control variables
// -----------------------
static double bignumber      = pow(10,20.);
static double smallnumber    = pow(10,-20.);
static double epsilon        = 0.01;        // coarseness of scan of multi-D unit cube
static double InvEpsilon     = 100.;        // inverse of epsilon

// Factorial function
// ------------------
double Factorial (int n) {
  if (n<=1) return 1;
  double r = 1.;
  for (int i=1; i<=n; i++) {
    r *= i;
  }
  return r;
}

// Tail probability for observing >=n events when mu are expected from a Poisson
// -----------------------------------------------------------------------------
double Poisson (double n, double mu) {
  return TMath::Gamma(n,mu);
}

// Return Z score for a Poisson counting exp seeing n, expecting mu events
// -----------------------------------------------------------------------
double Zscore_Poisson (double n, double mu) {
  double p = TMath::Gamma(n,mu);
  if (p<1.E-310) return 37.7; // for larger significances it breaks down
  return sqrt(2)*TMath::ErfcInverse(2*p);
}

// Ratio maximization based on on/off using box volume and total stat in full space
// --------------------------------------------------------------------------------
double R (int Non, int Ntot, double volume) {
  if (volume==0.) return 0.;
  double tau = (1.-volume)/volume;
  int Noff   = Ntot-Non;
  return (double)Non/(Ntot*volume+1.);
}

// Ratio maximization based on on/off, version 2 with regularization
// -----------------------------------------------------------------
double R2 (int Non, double Noff) {
  double r = (double)Non/(RegFactor+Noff);
  return r;
}

// Profile likelihood Z score from Li and Ma for on/off problem - we will use this,
// as it is defined without problem for arbitrarily large inputs
// --------------------------------------------------------------------------------
double ZPL (int Non, int Ntot, double volume) {
  if (volume==0 || volume==1 || Non==0) return 0.;
  if (Non>0 && Ntot-Non==0) return 0.;
  double tau = (1.-volume)/volume;
  if (Non==(Ntot-Non)/tau) return 0.;
  int Noff = Ntot-Non;
  double z = sqrt(2)* pow (Non*log(Non*(1+tau)/Ntot)+Noff*log(Noff*(1+tau)/(Ntot*tau)),0.5);
  if (z!=z) return 0.;
  if ((double)Non<Noff/tau) return -z;
  return z;
}

// Profile likelihood Z score from Li and Ma for on/off problem 
// Version with direct tau input
// ------------------------------------------------------------
double ZPLtau (int Non, int Noff, double tau) {
  if (Non==0 || Noff==0 || Non==Noff/tau || tau==0) return 0;
  if (Non>0 && Noff==0) return 0.;
  int Ntot = Non+Noff;
  double z = sqrt2 * pow (Non*log(Non*(1+tau)/Ntot)+Noff*log(Noff*(1+tau)/(Ntot*tau)),0.5);
  if (z!=z) return 0.;
  if ((double)Non<Noff/tau) return -z;
  return z;
}

// Version of ZPL which incorporates a possible background systematic in the form
// of passing lowest Z considering tau variations by syst*100%
// ------------------------------------------------------------------------------
double ZPLsyst (int Non, int Ntot, double volume, double syst) {
  if (volume==0 || volume==1 || Non==0) return 0.;
  if (Non>0 && Ntot-Non==0) return 0.;
  double tau    = (1.-volume)/volume;
  if (Non==(Ntot-Non)/tau) return 0.;
  double taulow = (1.-syst)*(1.-volume)/volume;
  double tauhig = (1.+syst)*(1.-volume)/volume;
  int Noff = Ntot-Non;
  double z    = sqrt2 * pow (Non*log(Non*(1+tau)/Ntot)+Noff*log(Noff*(1+tau)/(Ntot*tau)),0.5);
  double zhig = sqrt2 * pow (Non*log(Non*(1+tauhig)/Ntot)+Noff*log(Noff*(1+tauhig)/(Ntot*tauhig)),0.5);
  double zlow = sqrt2 * pow (Non*log(Non*(1+taulow)/Ntot)+Noff*log(Noff*(1+taulow)/(Ntot*taulow)),0.5);
  if ((double)Non<Noff/tau)    z    = -z;
  if ((double)Non<Noff/tauhig) zhig = -zhig;
  if ((double)Non<Noff/taulow) zlow = -zlow;
  if (z>zhig) z = zhig;
  if (z>zlow) z = zlow;
  if (z!=z) return 0.;
  return z;
}

// Cholesky-Banachiewicz decomposition of covariance matrix, used to generate a multivariate
// Gaussian distribution in N dimensions.
// NB It returns false (without finishing the calculation of L) if A is not pos.def., thereby
// it can be directly used to check that A is indeed positive definite.
// ------------------------------------------------------------------------------------------
static double A[ND][ND];
static double L[ND][ND];
bool Cholesky(int N) {  
  // We take a covariance matrix A, of dimension NxN, and we find a lower triangular
  // matrix L such that LL^T = A. See https://en.wikipedia.org/wiki/Cholesky_decomposition
  // -------------------------------------------------------------------------------------
  double sum1;
  double sum2;
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      L[i][j] = 0;
    }
  }
  for (int i=0; i<N; i++) {
    for (int j=0; j<=i; j++) {
      sum1 = 0;
      sum2 = 0;
      if (j>0) {
        for (int k=0; k<j-1; k++) {
        sum1 += L[i][k] * L[j][k];
        }
        for (int k=0; k<j-1; k++) {
        sum2 += L[j][k] * L[j][k];
        }
      }
      if (i==j) {
    	L[j][j] = sqrt(A[j][j]-sum2);
      } else { // i>j
	    L[i][j] = (A[i][j]-sum1)/L[j][j];
      } // all others remain zero      
      if (L[i][j] != L[i][j]) return false;
    }
  }
  return true;
}


// This routine determines the sidebands of a box in a foul-proof way
// NB unlike before, we allow boundaries of sidebands to have any value
// --------------------------------------------------------------------
void determineSB (double Smi[maxNvar], double Sma[maxNvar], double Bmi[maxNvar], double Bma[maxNvar], int Nvar) {
  double minratio = bignumber;
  double AvailableRatio[maxNvar];
  // Find minimum ratio of available space to box space, among all directions
  for (int i=0; i<Nvar; i++) {
    Smi[i] = Bmi[i]*(1+sidewidth)-Bma[i]*sidewidth;
    if (Smi[i]<0.) Smi[i] = 0.;
    Sma[i] = Bma[i]*(1+sidewidth)-Bmi[i]*sidewidth;
    if (Sma[i]>1.) Sma[i] = 1.;
    AvailableRatio[i] = (1.-(Bma[i]-Bmi[i]))/(Bma[i]-Bmi[i]);
    if (AvailableRatio[i]<minratio) minratio = AvailableRatio[i];
  }
  // Order by available ratio
  int ind[maxNvar];
  for (int i=0; i<Nvar; i++) { ind[i]=i; };
  for (int times=0; times<Nvar; times++) {
    for (int i=Nvar-1; i>0; i--) {
      if (AvailableRatio[ind[i]]<AvailableRatio[ind[i-1]]) {
        // Swap indices
        int tmp  = ind[i];
        ind[i]   = ind[i-1];
        ind[i-1] = tmp;
      }
    }
  }
  // Now AvailableRatio[ind[Nvar-1]] is the largest, AvailableRatio[ind[0]] is the smallest
  double NeededRatioPerVar;
  double CurrentFactor = 1.;
  for (int i=0; i<Nvar; i++) {
    if (AvailableRatio[ind[i]]==0) continue; // can't use this dimension
    NeededRatioPerVar = pow(2./CurrentFactor,1./(Nvar-i))-1.;
    if (AvailableRatio[ind[i]]<NeededRatioPerVar) { // use all the space available for this var
      Smi[ind[i]] = 0.;
      Sma[ind[i]] = 1.;
      CurrentFactor = CurrentFactor*(1.+AvailableRatio[ind[i]]);
      if (i<Nvar-1) NeededRatioPerVar = pow(2./CurrentFactor,Nvar-i-1)-1.; // rescaled needed ratio for the others
    } else { // We can evenly share the volume in the remaining coordinates
      double distmin = Bmi[ind[i]];
      double deltax  = Bma[ind[i]]-Bmi[ind[i]];
      if (distmin>1.-Bma[ind[i]]) { // Upper boundary is closest
        distmin = 1.-Bma[ind[i]];
        if (2.*distmin/deltax>=NeededRatioPerVar) {
            Smi[ind[i]] = Bmi[ind[i]]-NeededRatioPerVar*deltax/2.; // epsilon*(int)(InvEpsilon*(Bmi[ind[i]]-NeededRatioPerVar*deltax/2.));
            Sma[ind[i]] = Bma[ind[i]]+NeededRatioPerVar*deltax/2.; // epsilon*(int)(InvEpsilon*(Bma[ind[i]]+NeededRatioPerVar*deltax/2.));
        } else {
            Sma[ind[i]] = 1.;
            Smi[ind[i]] = 1.-deltax*(1.+NeededRatioPerVar); // epsilon*(int)(InvEpsilon*(1.-deltax*(1.+NeededRatioPerVar)));
        }
        CurrentFactor = CurrentFactor*(1.+NeededRatioPerVar);
      } else { // lower boundary is closest 
        if (2.*distmin/deltax>=NeededRatioPerVar) {
            Smi[ind[i]] = Bmi[ind[i]]-NeededRatioPerVar*deltax/2.; // epsilon*(int)(InvEpsilon*(Bmi[ind[i]]-NeededRatioPerVar*deltax/2.));
            Sma[ind[i]] = Bma[ind[i]]+NeededRatioPerVar*deltax/2.; // epsilon*(int)(InvEpsilon*(Bma[ind[i]]+NeededRatioPerVar*deltax/2.));
        } else {
            Smi[ind[i]] = 0.;
            Sma[ind[i]] = deltax*(1.+NeededRatioPerVar); // epsilon*(int)(InvEpsilon*(deltax*(1.+NeededRatioPerVar)));
        }
        CurrentFactor = CurrentFactor*(1.+NeededRatioPerVar);
      }
    }
  }
  return;
}

// This routine checks whether n has binary decomposition with bit index on
// ------------------------------------------------------------------------
bool bitIsOn (int n, int index) {
  int imax = (int)(log(n)/log(2));
  if (imax<index) return false;
  for (int i=imax; i>=index; i--) {
    if (n-pow(2,i)>=0) {
      n = n-pow(2,i);
      if (imax==index) return true;
    }
  }
  return false;
}

// ----------------------------------------------------------------------------------------------
//
//                       ----------------
//                       R  A  N  B  O  X  
//                       ----------------
//
//                This macro scans NSEL-dimensional subspaces of a NAD dimensional feature
//                space, in search for a multidimensional interval which contains significantly
//                more data than predicted from uniform density assumptions. 
//                The data are preprocessed to fit in a hypercube whose marginals are all flat,
//                and optionally reduced by principal component analysis.
//                The box scanning each subspace is initialized at random (if Algorithm=0) or by
//                finding a dense region with a cluster search.
//                The test statistic that defines the most dense regions can be chosen between
//                a "Profile likelihood" Z-value and a density ratio.
//                It is possible to generate synthetic data flat in all features or with an injected
//                fraction of signal sampled from a multidimensional gaussian of customizable parameters.
//                In this version the code may also read in data from the "HEPMASS" dataset available
//                from the UCI repository, the miniBooNE dataset, or a credit card fraud dataset.
//
//                Original creation - Tommaso Dorigo, 2019-2022
// ------------------------------------------------------------------------------------------------------

int main (int argc, char *argv[]) { 

  // dataset: type of dataset used in search:
  // 0=HEPMASS; (see http://archive.ics.uci.edu/ml/datasets/hepmass)
  // 1=miniBooNE; (see https://archive.ics.uci.edu/ml/datasets/MiniBooNE+particle+identification)
  // 2=credit card fraud data; (see https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
  // 3=EEG data; (see https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State)
  // 4=synthetic dataset (mock - generated on demand); 
  // 5=LHC olympics 2020 qq dataset (see https://zenodo.org/record/6466204#.Ypdsl-ixVhF).
  // 6=LHC olympics 2020 qqq dataset (see https://zenodo.org/record/6466204#.Ypdsl-ixVhF).

  // Defaults
  // --------
  int dataset = 0;
  int Ntrials = 1000;
  int Nsignal = 200;
  int Nbackground = 4800;
  int Algorithm = 5;
  bool useZPL = false;
  bool useSB = true;
  bool PCA = false;
  int Nvar = 8;
  int Nremoved = 0;
  int speedup = 1;
  int NH0 = 1;
  
  for (int i=1; i<argc; i++) { // NNBB the first argv is the command launching the program itself! so i=1
    if (!strcmp(argv[i],"-h")) {
      cout << "List of arguments:" << endl;
      cout << "-Dat Dataset type" << endl;
      cout << "-Ntr Number of trials" << endl;
      cout << "-Nsi Number of signal events" << endl;
      cout << "-Nba Number of background events" << endl;
      cout << "-Alg Initialization algorithm (1-5)" << endl;
      cout << "-Zpl Use ZPL test stat (default false)" << endl;
      cout << "-Pca Whether to do PCA (def false)" << endl;
      cout << "-Nva Number of variables" << endl;
      cout << "-Nre Variables to remove" << endl;
      cout << "-Spe Speedup factor in search" << endl;
      cout << "-Reg regularization factor (only if not using -Zpl)" << endl;
      return 0;
    }
    else if (!strcmp(argv[i],"-Dat")) {dataset = atoi(argv[++i]);}
    else if (!strcmp(argv[i],"-Ntr")) {Ntrials = atoi(argv[++i]);}
    else if (!strcmp(argv[i],"-Nsi")) {Nsignal = atoi(argv[++i]);}
    else if (!strcmp(argv[i],"-Nba")) {Nbackground = atoi(argv[++i]);}
    else if (!strcmp(argv[i],"-Alg")) {Algorithm = atoi(argv[++i]);}
    else if (!strcmp(argv[i],"-Nva")) {Nvar = atoi(argv[++i]);}
    else if (!strcmp(argv[i],"-Nre")) {Nremoved = atoi(argv[++i]);}
    else if (!strcmp(argv[i],"-Spe")) {speedup = atoi(argv[++i]);}
    else if (!strcmp(argv[i],"-Reg")) {RegFactor = atoi(argv[++i]);}
    else if (!strcmp(argv[i],"-Zpl")) {useZPL = true; useSB = false; ++i;}
    else if (!strcmp(argv[i],"-Pca")) {PCA = true; ++i;}
    else { 
      cout << "Warning, choices not allowed" << endl;
      return 0;
    }
  }

  // dataset: type of dataset used in search:
  // 0=HEPMASS; 1=miniBooNE; 2=fraud data; 3=EEG data; 4=synthetic dataset (mock); 5=LHC olympics 2020 dataset.
  // Ntrials: number of random subspaces investigated
  // Algorithm: type of seeding clustering used in initialization of box
  // useZPL: whether to use ZPL as test statistic to maximize
  // useSB: whether to use sidebands in density prediction inside box
  // PCA: whether to apply PCA to data (removing lowest Nremoved components)
  // Nvar: box dimension
  // Nremoved: removed dimensions (with PCA or with discarding highly correlated variables)
  // speedup: reduction factor of used data in initial clustering
  // RegFactor: added number of exp background to denominator of R_reg test statistic
  // If one_off, we provide a single list of features for the only subspace to investigate.
  // This allows to keep RanBox and RanBoxIter code versions aligned.
  // --------------------------------------------------------------------------------------
  bool one_off = false;
  if (one_off) {
    Nvar = 12;
  }
  int Ivar_fix[12] = { 17, 21, 9, 5, 10, 13, 14, 18, 26, 6, 24, 11 };
  // int Ivar_fix[12] = { 0, 1, 3, 5, 7, 9, 10, 11, 14, 15, 16, 19 };
  
  // Depending on user choices, dimensions are set differently:
  // ----------------------------------------------------------
  int NAD;
  int NSEL;
  if (Nremoved>0) {
    if (PCA) {
      RemoveHighCorr = false;
    } else {
      RemoveHighCorr = true;
    }
  } 

  bool mock = false;
  if (dataset==0) { // HEPMASS data
    NAD = 27;
    NSEL = NAD-Nremoved;
  } else if (dataset==1) { // MiniBooNE data
    NAD = 50;
    NSEL = NAD-Nremoved;
    D2kernel = 0.1;
  } else if (dataset==2) { // credit card data
    NAD = 30;
    NSEL = NAD-Nremoved;
    D2kernel = 0.04;
  } else if (dataset==3) { // EEG eye detection data
    NAD = 14;
    NSEL = NAD-Nremoved;
    D2kernel = 0.04;
  } else if (dataset==4) { // mock data
    NAD = 20;
    NSEL = 20;
    mock = true;
  } else if (dataset==5) { // LHC olympics 2020
    NAD = 14;
    NSEL = NAD-Nremoved;
    D2kernel = 0.01; // check
    mock = false;
  } else if (dataset==6) { // LHC olympics 2020
    NAD = 14;
    NSEL = NAD-Nremoved;
    D2kernel = 0.01; // check
    mock = false;
  }


  // Change preset generator
  // NB other versions of TRandom are flawed
  // ---------------------------------------
  delete gRandom;
  double seed = 0.1;
  TRandom3 * myRNG = new TRandom3();
  gRandom = myRNG;
  gRandom->SetSeed(seed);
  
  // Check if input parameters and other user choices are consistent
  // ---------------------------------------------------------------
  if (pow(0.5,Nvar)*(Nsignal+Nbackground)<1) cout << "Warning - too few events for this Nvar" << endl;

  if (NAD<Nvar) {
    cout << "  Sorry, cannot set subspace dimension larger than active variables dim." << endl;
    return 0; // (to avoid infinite loop in choosing vars)
  }
  if (Nvar<2) {
    cout << "  Sorry, Nvar must be >=2" << endl;
    return 0;
  }
  if (maxHalfMuRange>=0.5) maxHalfMuRange = 0.499;

  if (Algorithm!=0) maxJumps = 0; // It is only useful for random seeding
  if ((Gaussian_dims>0 && Nsignal==0 && mock) ||  
      (NseedTrials>1 && Ntrials>1) || (!fixed_gaussians && maxHalfMuRange!=0.35)) {
    cout << "  Inconsistent choice of pars." << endl;
    return 0;
  }

  if (dataset==1 && NH0>1) {
    cout << "  Sorry, can only do 1 trial with miniBoone data for now" << endl;
    return 0;
  }

  if (force_gaussians && Gaussian_dims<Nvar && Gaussian_dims>0) {
    cout << "  Cannot have forced gaussians on with these settings!" << endl;
    return 0;
  }

  if (NH0>1) plots = false;
  if (speedup<1) speedup = 1; // factor to speed up algorithms of clustering, set >1 (e.g. 4 or 8) for quick checks

  double did = 999.999*gRandom->Uniform();             // Identifier for summary printouts
  int id     = (int)did;
  if (Nbackground+Nsignal<=100) return 0;                // Avoid too small datasets;
  double FlatFrac = Nbackground/(Nbackground+Nsignal); // FlatFrac: fraction of flat events in toy 
                                                       // (the rest are multivariate normal)

  // Defaults
  // --------
  string outputPath = "/lustre/cmswork/dorigo/RanBox/Output";
  string asciiPath  = "/lustre/cmswork/dorigo/RanBox/Ascii";
  
  std::stringstream sstr;
  std::stringstream sstr2;
  if (mock) {
    sstr << "/RB_mock_" << id << "_Ntr" << Ntrials 
	 << "_NS" << Nsignal << "_NB" << Nbackground << "_A" << Algorithm;
    if (Algorithm>0) sstr << "_SP" << speedup;
    if (fixed_gaussians) {
      sstr << "_FiG";
      if (narrow_gaussians) {
    	sstr << 0.05;
      } else { 
    	sstr << 0.10;
      }
    }
    sstr << "_FoG" << force_gaussians  
	 << "_NG" << Gaussian_dims; // << "_MR" << maxRho << "_MHW" << maxHalfMuRange;
    if (NH0>1) sstr << "_Nrep" << NH0;
  } else {
    if (dataset==0) {
      sstr << "/RB_HEPMASS_" << id << "_Ntr" << Ntrials 
	   << "_NS" << Nsignal << "_NB" << Nbackground << "_A" << Algorithm;
    } else if (dataset==1) {
      sstr << "/RB_miniBooNE_" << id << "_Ntr" << Ntrials 
	   << "_NS" << Nsignal << "_NB" << Nbackground << "_A" << Algorithm;
    } else if (dataset==2) {
      sstr << "/RB_fraud_" << id << "_maxNv" << Ntrials 
	   << "_NS" << Nsignal << "_NB" << Nbackground << "_A" << Algorithm;
    } else if (dataset==3) {
      sstr << "/RB_EEG_" << id << "_maxNv" << Ntrials 
	   << "_NS" << Nsignal << "_NB" << Nbackground << "_A" << Algorithm;
    } else if (dataset==4) {
      sstr << "/RB_mock_" << id << "_maxNv" << Ntrials 
	   << "_NS" << Nsignal << "_NB" << Nbackground << "_A" << Algorithm;
    } else if (dataset==5) {
      sstr << "/RB_LHCOlympics_" << id << "_maxNv" << Ntrials 
	   << "_NS" << Nsignal << "_NB" << Nbackground << "_A" << Algorithm;
    } else if (dataset==6) {
      sstr << "/RB_LHCOlympics_qqq_" << id << "_maxNv" << Ntrials 
	   << "_NS" << Nsignal << "_NB" << Nbackground << "_A" << Algorithm;
    }
  }
  if (Algorithm>0) sstr << "_SP" << speedup;
  if (useZPL) { 
    sstr << "_ZPL";
  } else {
    sstr << "_R2";
  }
  if (useSB) {
    sstr << "_SB";
  } else {
    sstr << "_Vol";
  }
  string rootfile       = outputPath + sstr.str() + ".root";
  string asciifile      = asciiPath + sstr.str() + ".asc";
  string summaryfile;
  string zplfile;
  if (dataset==0) {
    summaryfile = asciiPath + "/SummaryRB_HEPMASS.asc";
  } else if (dataset==1) {
    summaryfile = asciiPath + "/SummaryRB_miniBooNE.asc";
  } else if (dataset==2) {
    summaryfile = asciiPath + "/SummaryRB_fraud.asc";
  } else if (dataset==3) {
    summaryfile = asciiPath + "/SummaryRB_EEG.asc";
  } else if (dataset==4) {
    summaryfile = asciiPath + "/SummaryRB_mock.asc";
  } else if (dataset==5) {
    summaryfile = asciiPath + "/SummaryRB_LHCOlympics.asc";
  } else if (dataset==6) {
    summaryfile = asciiPath + "/SummaryRB_LHCOlympics_qqq.asc";
  }
  ofstream results; // output file
  ofstream summary; // summary file 
  ofstream zpl;

  // Create big array to store event kinematics
  // ------------------------------------------
  float * feature_all = new float[(maxEvents)*ND];
  int * order_ind_all = new int[(maxEvents)*ND];
  float ** feature = new float*[ND];
  int ** order_ind = new int*[ND];
  for (int i=0; i<ND; i++) {
    feature[i]   = feature_all+(maxEvents)*i;
    order_ind[i] = order_ind_all+(maxEvents)*i;
  }
  double * xtmp = new double[maxEvents];
  bool * isSignal = new bool[maxEvents];
  int * cum_ind = new int[maxEvents];
  
  // Arrays used for cluster search
  // ------------------------------
  int * Closest = new int[maxEvents];  
  int * PB_all = new int[(maxEvents)*maxClosest];
  int ** PointedBy = new int*[maxClosest];
  for (int i=0; i<maxClosest; i++) {
    PointedBy[i] = PB_all+(maxEvents)*i;
  }
  int * AmClosest = new int[maxEvents];
  
  // Header Printout
  // ---------------
  cout << endl;
  cout << "  -------------------------------------------------------------------------------------------" << endl;
  cout << endl;
  cout << "                                           R  A  N  B  O  X" << endl;
  cout << "                                           ----------------" << endl;
  cout << endl;
  
  // Set up ascii output
  // -------------------
  results.open(asciifile);
  summary.open(summaryfile, std::ios::out | std::ios::app);
  if (NH0>1) zpl.open(zplfile, std::ios::out | std::ios::app);

  results << "  ----------------------------- " << endl;
  results << "           R A N B O X          " << endl;
  results << "  ----------------------------- " << endl << endl << endl;
  results << "  Parameters: " << endl;
  results << "  ----------------------------- " << endl << endl;
  results << "  Dataset       = ";
  if (dataset==0) results << "HEPMASS data" << endl;
  if (dataset==1) results << "miniBooNE data " << endl;
  if (dataset==2) results << "Credit card frauds" << endl;
  if (dataset==3) results << "EEG data" << endl;
  if (dataset==4) results << "MOCK data" << endl;
  if (dataset==5) results << "LHC olymplics data" << endl;
  if (dataset==6) results << "LHC olymplics data qqq" << endl;
  results << "  Nsignal      = " << Nsignal << endl;
  results << "  Nbackground  = " << Nbackground << endl;
  results << "  NAD          = " << NAD << endl;
  results << "  PCA          = " << PCA << endl;
  if (PCA) results 
	     << "  NSEL         = " << NSEL << endl;
  results << "  Nvar         = " << Nvar << endl;
  results << "  Algorithm    = " << Algorithm << endl;
  results << "  Speedup      = " << speedup << endl;
  results << "  useSB        = " << useSB <<endl;
  results << "  useZPL       = " << useZPL << endl;
  if (useZPL) {
    results << "  Syst         = " << syst << endl;
  } else {
    results << "  RegFactor    = " << RegFactor << endl;
  }
  results << "  Root file    = " << rootfile << endl;
  results << "  maxBoxVolume = " << maxBoxVolume <<endl;
  results << "  maxGDLoops   = " << maxGDLoops << endl;
  if (mock) {
    results << "  Fixed gauss  = " << fixed_gaussians << endl;
    results << "  Narrow gauss = " << narrow_gaussians << endl;
    results << "  Gaussian dims= " << Gaussian_dims << endl;
    results << "  Force gauss  = " << force_gaussians << endl;
    results << "  maxRho       = " << maxRho << endl;
    results << "  maxHalfMuR   = " << maxHalfMuRange << endl;
  }
  results << "  Nremoved     = " << Nremoved << endl;
  results << "  id           = " << id << endl;
  results << endl;
  
  // Control histograms
  // ------------------
  TH1F * Zvalue_in       = new TH1F     ( "Zvalue_in",       "", 200, 0., 50.);
  TH1F * Zvalue_fi       = new TH1F     ( "Zvalue_fi",       "", 200, 0., 50.);
  TH2F * Bounds_in       = new TH2F     ( "Bounds_in",       "", 60, -0.1, 1.1, 60, -0.1, 1.1);
  TH2F * Bounds_fi       = new TH2F     ( "Bounds_fi",       "", 60, -0.1, 1.1, 60, -0.1, 1.1);
  TH2F * Drift           = new TH2F     ( "Drift",           "", 50, -1., 1., 50, -1., 1.);
  TH1F * NGDsteps        = new TH1F     ( "NGDsteps",        "", maxGDLoops+1, -0.5, maxGDLoops+0.5);
  TH1F * Ninbox_in       = new TH1F     ( "Ninbox_in",       "", 100, 0., 200.);
  TH1F * Ninbox_fi       = new TH1F     ( "Ninbox_fi",       "", 100, 0., 200.);
  TH2F * Ninbox_in_vs_fi = new TH2F     ( "Ninbox_in_vs_fi", "", 50, 0., 200., 50, 0., 200.);
  TH1F * InitialDensity  = new TH1F     ( "InitialDensity",  "", 100, 0., 5.);
  TH1F * InitialVolume   = new TH1F     ( "InitialVolume",   "", 100, 0., 0.5);
  TH1F * NClomax         = new TH1F     ( "NClomax",         "", 40, 0., 40. );
  TH1F * ZH0             = new TH1F     ( "ZH0",             "", 1000, 0., 100.);
  TProfile * ZvsOrder    = new TProfile ("ZvsOrder",         "", 20, 0., 2., 0., 30.); 

  TCanvas * PP;
  TCanvas * OP;
  TCanvas * UP;
  TCanvas * P2;
  TCanvas * OP2;
  TCanvas * UP2;
  
  // Construct plots of marginals for the considered features in the best box
  // ------------------------------------------------------------------------
  // If PCA, transformed components; otherwise original
  TH1D * Plot_al[maxNvar];
  TH1D * Plot_in[maxNvar];
  TH1D * Plot_ex[maxNvar];
  TH1D * Plot_si[maxNvar];
  // Original coordinates - NB the PCA untransformed space has all coordinates
  TH1D * OPlot_al[maxNvar];
  TH1D * OPlot_in[maxNvar];
  TH1D * OPlot_si[maxNvar];
  // no exclusive n-1 plot since this has no meaning in backtransformed space
  // Uniform marginals transforms (of PCA or normal, depending if PCA)
  TH1D * UPlot_al[maxNvar];
  TH1D * UPlot_in[maxNvar];
  TH1D * UPlot_ex[maxNvar];
  TH1D * UPlot_si[maxNvar];
  // Scatterplots now
  // ----------------
  TH2D * SCP_al[N2var];
  TH2D * SCP_in[N2var];
  TH2D * SCP_ex[N2var];
  TH2D * OSCP_al[N2var];
  TH2D * OSCP_in[N2var];
  // for original space n-2 plots make no sense
  TH2D * USCP_al[N2var];
  TH2D * USCP_in[N2var];
  TH2D * USCP_ex[N2var];

  // Generic variable names
  // ----------------------
  TString varname[ND] = { "V00", "V01", "V02", "V03", "V04", "V05", "V06", "V07", "V08", "V09", 
			  "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", 
			  "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "V29", 
			  "V30", "V31", "V32", "V33", "V34", "V35", "V36", "V37", "V38", "V39", 
			  "V40", "V41", "V42", "V43", "V44", "V45", "V46", "V47", "V48", "V49" };
  TString varname_mock[ND];
  
  // Variables to print out progress information during run time
  // -----------------------------------------------------------
  char progress[53] = {"[--10%--20%--30%--40%--50%--60%--70%--80%--90%-100%]"};
  int currchar;
  int block;
  
  bool messaged = false;
  ifstream events; // input file
  
  // Other vars and counters
  // -----------------------
  float Xmin[ND];
  float Xmax[ND];
  float OXmin[ND];
  float OXmax[ND];
  double mean[ND];
  double sigma[ND];
  bool Gaussian[ND];
  double gauss[ND];
 
  // Variables used for best box
  // ---------------------------
  int Ivar[maxNvar]; // Nvar is the number of features used to create a box
  double Blockmin[maxNvar]; // Blockmin is the left boundary of the block
  double Blockmax[maxNvar]; // Blockmax is the right boundary of the block
  double Sidemin[maxNvar];
  double Sidemax[maxNvar];
  // And for the best block found:
  int Ivar_best[maxNvar];
  double Blockmin_best[maxNvar];
  double Blockmax_best[maxNvar];
  double Zval_best      = -bignumber; // Absolute best Z value in all trials
  double Nin_best       = 0; // N in best box
  double Nexp_best      = 0; // expected N from volume
  int gd_best           = 0; // Loops for best Z region
  int trial_best        = 0;
  double ID_best        = 0; // initial density of box
  int Ivar_int[8]       = {0,1,2,3,4,5,6,7}; // orig vars for scatterplots

  // Check if Ntrials is consistent
  // ------------------------------
  bool doall = false;
  double comb   = Factorial(NSEL)/(Factorial(Nvar)*Factorial(NSEL-Nvar));

  if (Ntrials>comb/3) {
    cout << "  Ntrials large for random subspace search:" << endl;
    cout << "  Number of combinations is " << comb << " - will do all" << endl;
    doall = true;
    for (int k=0; k<Nvar; k++) { Ivar[k] = k; }; // if we are doing the combinatorial, we need to init
    Ntrials = comb;
  }
  // But don't forget the max dimensions!
  // ------------------------------------
  if (Ntrials>maxNtr) {
    cout << "  max number of trials is " << maxNtr 
	 << " - setting Ntrials to " << maxNtr << endl;
    Ntrials = maxNtr;
  }  
  
  // If we are testing the accuracy of different seeding algorithms, we need to
  // keep track of the following:
  // --------------------------------------------------------------------------
  double Aver_SF_caught     = 0.;
  double Aver2_SF_caught    = 0.;
  double Aver_1s_contained  = 0.;
  double Aver2_1s_contained = 0.;
  if (!mock) NseedTrials    = 1;
  int goodevents;
  
  // Variables used to keep track of results
  // ---------------------------------------
  int * bv_all = new int[maxNvar*maxNtr];
  int ** BoxVar = new int*[maxNtr];
  for (int i=0; i<maxNtr; i++) {
    BoxVar[i] = bv_all+(maxNvar)*i;
  }
  // int BoxVar[maxNtr][maxNvar];
  double * BoxVol = new double[maxNtr];
  double * BoxNin = new double[maxNtr];
  double * BoxNex = new double[maxNtr];
  double * BoxZpl = new double[maxNtr];
  double * BoxSFr = new double[maxNtr];
  int    * BoxInd = new int[maxNtr];  
  for (int i=0; i<maxNtr; i++) {
    BoxVol[i] = 1.;
    BoxNin[i] = 0.;
    BoxNex[i] = 0.;
    BoxZpl[i] = 0.;
    BoxSFr[i] = 0.;
    BoxInd[i] = 0;
    for (int j=0; j<maxNvar; j++) {
      BoxVar[j][i] = 0;
    }
  }

  // Loop for test of cluster seeding (uncomment if necessary)
  // ---------------------------------------------------------
  // for (int Ntestseed=0; Ntestseed<NseedTrials; Ntestseed++) {
  
  string alphabet [] = {"a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p",
			"q","r","s","t","u","v","w","x","y","z"};

  int save_p  = 0;
  int save_q  = 0; 
  bool closed = true;
  ifstream file;
  string dataname;

  // BIG LOOP FOR H0 //////////////////////////////////////////////////////////////////

  for (int IH0=0; IH0<NH0; IH0++) {
    
    results << "\n\nCycle n: " << IH0 << endl << endl; 

    Zval_best  = -bignumber;
    Nin_best   = 0;
    Nexp_best  = 0;
    gd_best    = 0;
    ID_best    = 0;    
    int NSread = 0;
    int NBread = 0;
    
    // This macro can process data from input files, or generate mock data from flat and gaussians
    // -------------------------------------------------------------------------------------------
    if (mock) {
      
      int Nmock = Nsignal+Nbackground;
      if (Gaussian_dims==0) {
        // Generate flat data for calibration of Z-values
        // ----------------------------------------------
        for (int i=0; i<Nmock; i++) {
            for (int dim=0; dim<NAD; dim++) {
                feature[dim][i] = gRandom->Uniform();
            }
        }
        NBread = Nmock;
        NSread = 0;
        for (int dim=0; dim<NAD; dim++) {
            varname_mock[dim] = "Flat";
        }
      } else { // >0 gaussian dims
	
        // First of all, generate multivariate gaussian means and covariance matrix, and decomposition
        // -------------------------------------------------------------------------------------------      
        // Generate covariance matrix
        // First the diagonal terms
        for (int dimx=0; dimx<Gaussian_dims; dimx++) {
            if (fixed_gaussians) {
                if (narrow_gaussians) {
                    sigma[dimx] = 0.05;
                } else {
                    sigma[dimx] = 0.10;
                }
            } else {
                if (narrow_gaussians) {
                    sigma[dimx] = gRandom->Uniform(0.01,0.1); 
                } else {
                    sigma[dimx] = gRandom->Uniform(0.05,0.15);
                }
            }
            if (fixed_gaussians) {
                mean[dimx] = gRandom->Uniform(0.5-maxHalfMuRange,0.5+maxHalfMuRange);
            } else {
                mean[dimx] = gRandom->Uniform(3*sigma[dimx],1.-3*sigma[dimx]); 
            }
            A[dimx][dimx] = sigma[dimx]*sigma[dimx]; 
        }
        if (Multivariate) {
            // Then the off-diagonal terms
            double maxcorr = 1.;
            int Nattempts  = 0;
            bool success   = false;
            do {
                for (int dimx=0; dimx<Gaussian_dims-1; dimx++) {
                    for (int dimy = dimx+1; dimy<Gaussian_dims; dimy++) {
                        if (fixed_gaussians) {
                            double rnd = gRandom->Uniform();
                            double rho = 2.*(rnd-0.5)*maxRho;
                            /* 		  if (rnd<1/3.) { */
                            /* 		    rho = -maxRho; */
                            /* 		  } else if (rnd<2/3.) { */
                            /* 		    rho = 0.; */
                            /* 		  } else { */
                            /* 		    rho = maxRho; */
                            A[dimx][dimy] = rho * sqrt(A[dimx][dimx]*A[dimy][dimy]);
                        } else {
                            A[dimx][dimy] = gRandom->Uniform(-maxcorr,maxcorr) * sqrt(A[dimx][dimx]*A[dimy][dimy]);
                        }
                        A[dimy][dimx] = A[dimx][dimy];
                    }
                }	
                // The following is necessary in order to keep a good chance 
                // that a positive-defined matrix is found, when Gaussian_dims is large (NB it's not useful for fixed_gaussians=true)
                maxcorr = maxcorr/1.001;
                Nattempts++;
                success = Cholesky(Gaussian_dims);
            } while (!success && Nattempts<10000);
            // Find Cholesky decomposition A = L * L^T
            if (success) {
                cout << "  Cholesky decomposition of covariance done after " << Nattempts 
                << " attempts, maxcorr = " << maxcorr << endl;
            } else {
                cout << "  Could not decompose covariance, stopping." << endl;
                return 0;
            }
        } // end if Multivariate

        // Generate flat data plus a Gaussian multivariate in some of the features
        // -----------------------------------------------------------------------
        for (int dim=0; dim<NAD; dim++) {
            if (dim>=Gaussian_dims) {
                Gaussian[dim] = false;
                varname_mock[dim] = "Flat";
            } else {
                Gaussian[dim] = true;
                varname_mock[dim] = "Gaus";
            }
        }
        NSread = 0;
        NBread = 0;
        for (int i=0; i<Nmock; i++) {
            bool unif;
            double f;
            if (i<Nbackground) {
                unif = true;
                NBread++;
                isSignal[i] = false;
            } else { 
                unif = false;
                NSread++;
                isSignal[i] = true;
            }
            // Vector of normals 	  
            int gausdim = 0;
            for (int dim=0; dim<NAD; dim++) {
                if (unif || !Gaussian[dim]) { 
                    feature[dim][i] = gRandom->Uniform();
                } else {
                    if (Multivariate) {
                        // Generate multivariate or single variate normal data
                        // ---------------------------------------------------
                        f = mean[gausdim];
                        do {
                        for (int dim1=0; dim1<Gaussian_dims; dim1++) {
                            gauss[dim1] = gRandom->Gaus();
                        }
                        for (int dim1=0; dim1<Gaussian_dims; dim1++) {
                            f += L[gausdim][dim1]*gauss[dim1];
                        }
                        } while (f<0. || f>1.);
                        // In alternative to do/while, one could do:
                        // If outside [0,1], mirror around mean! As simple as that, keeps intervals having same integral
                        // across whole range. However, distributions are not smooth anymore...
                        // if (f<0. || f>1.) f = 2*mean[gausdim]-f;
                        feature[dim][i] = f;
                        if (f!=f) {
                            cout << "nan - stopping" << endl;
                            return 0;
                        }
                    } else { // no correlations
                        do {
                            f = gRandom->Gaus(mean[gausdim],sigma[gausdim]);
                        } while (f<0. || f>1.);
                            // if (f<0. || f>1.) f = 2*mean[gausdim]-f;
                            feature[dim][i] = f;
                            if (f!=f) {
                            cout << "nan - stopping" << endl;
                            return 0;
                        }
                    }
                    gausdim++;
                }
            }
        } // end i->Nmock
      } // end if >0 gaussian_dims
      
      cout << "  Generated in total " << NBread << ", " << NSread 
   	       << " Flat and Gauss-distributed events." << endl;
      if (PCA) cout << "  "; // leave space for PCA cout
      goodevents = NSread+NBread; // just to make sure we do not overflow it

      if (PCA) {

        ////////////////////////////////////////////////////////////////////////////
        // Start of principal component analysis
        // -------------------------------------
        // Initialize bounds in both frames
        for (int dim=0; dim<ND; dim++) {
            Xmin[dim]  = bignumber;
            Xmax[dim]  = -bignumber;
            OXmin[dim] = bignumber;
            OXmax[dim] = -bignumber;
        }
        TPrincipal* principal = new TPrincipal(NAD,"NAD");
        Double_t * data = new Double_t[ND];
        for (int i=0; i<goodevents; i++) {
            for (int dim=0; dim<NAD; dim++) {
                data[dim] = feature[dim][i];
            }
            principal->AddRow(data);
        }
        delete [] data;
        principal->MakePrincipals();
        principal->Test();
        principal->MakeHistograms();
        principal->MakeCode();
        for (int i=0; i<goodevents; i++) {
            Double_t * dataP = new Double_t[ND];
            Double_t * dataX = new Double_t[ND];
            for (int dim=0; dim<NAD; dim++) {
                dataX[dim] = feature[dim][i];
            }
            X2P(dataX,dataP); // Do PCA transformation
            
            // Find min and max of each component in both systems
            for (int dim=0; dim<NAD; dim++) {
                if (dataX[dim]<OXmin[dim]) OXmin[dim] = dataX[dim];
                if (dataX[dim]>OXmax[dim]) OXmax[dim] = dataX[dim];
                if (dataP[dim]<Xmin[dim])  Xmin[dim]  = dataP[dim];
                if (dataP[dim]>Xmax[dim])  Xmax[dim]  = dataP[dim];
                
                feature[dim][i] = dataP[dim]; // From now on, feature is the PCA transformed variable
            }
            delete [] dataP;
            delete [] dataX;
        }
        // Now we consider only the leading NSEL of the NAD variables
        
        principal->~TPrincipal();
	
      } else {
        // We need these to be initialized even if we don't do PCA, for plots
        // ------------------------------------------------------------------
        for (int dim=0; dim<NAD; dim++) {
        Xmin[dim]  = 0.;
        Xmax[dim]  = 1.;
        OXmin[dim] = 0.;
        OXmax[dim] = 1.;
        }
      } // end if PCA or not
      
      // -----------------------------------
      // End of principal component analysis
      
      ////////////////////////////////////////////////////////////////////////////      
      // Sort mock data
      // --------------
      cout << "  Sorting mock data: " << progress[0];
      int itmp;
      double dtmp;
      currchar = 1;
      int incr = 50/NAD;
      for (int dim=0; dim<NAD; dim++) {		
        // Print out progress in nice format
        // ---------------------------------
        if (currchar>52) currchar -= 53;
        for (int ch=1; ch<=incr; ch++) { cout << progress[currchar+ch]; };
        currchar += incr;
        for (int i=0; i<Nmock; i++) {
            xtmp[i] = feature[dim][i];
            cum_ind[i] = i;
        }
        for (int times=0; times<=Nmock; times++) {  
            for (int i=Nmock-1; i>0; i--) {
                if (xtmp[i]<xtmp[i-1]) {
                    dtmp = xtmp[i-1];
                    xtmp[i-1] = xtmp[i];
                    xtmp[i] = dtmp;
                    itmp = cum_ind[i-1];
                    cum_ind[i-1] = cum_ind[i];
                    cum_ind[i] = itmp;
                }
            }	  
        } // End times loop
        for (int i=0; i<Nmock; i++) {
            order_ind[dim][i] = -1;
            for (int j=0; j<Nmock && order_ind[dim][i]==-1; j++) {
                if (cum_ind[j]==i) {
                    order_ind[dim][i] = j;
                }
            }
        }	  
      } // End of sorting loop
      for (int c=currchar; c<52; c++) cout << progress[c]; 
      cout << endl << endl;
      
    } else { // Not mock data: read real data 
      
      // Setup printout of processing progress bar
      // -----------------------------------------
      cout << "  Processing data:  " << progress[0];
      currchar = 1;

      // Start of loop
      // -------------
      goodevents = 0;
      NSread     = 0;
      NBread     = 0;
      block      = (Nsignal+Nbackground)/50; 
      double f[ND];
      int Nvoided = 0;
      if (dataset==0) { // HEPMASS dataset

        int maxletter1 = 6; // a-f
        int maxletter2 = 26; // ea-ez, fa-fk 
        bool readok;
        for (int p=save_p; p<maxletter1; p++) {
            if (p==5) maxletter2 = 11; // only files up to fk are available
            for (int q=save_q; q<maxletter2; q++) { //when we are at the end of a file we go back here      
                if (closed) {
                    dataname = "/lustre/cmswork/dorigo/RanBox/datafiles/HEPMASS/whiteson_" + alphabet[p] + alphabet[q];
                    file.open(dataname);
                    cout << "  Now reading file " << dataname << endl;
                    closed = false;
                }
                std::string data;
                std::string::size_type sz;
                do { // prime the loop
                    readok = (bool) std::getline(file,data,','); 
                    if (readok) {
                        double s = std::stod(data,&sz);
                        for (int dim=0; dim<NAD-1; dim++) {
                            std::getline(file, data,',');
                            f[dim] = std::stod(data,&sz);
                        }
                        std::getline(file, data, '\n');
                        f[NAD-1] = std::stod(data,&sz);
                        if (s==1 && NSread<Nsignal) {
                            isSignal[goodevents] = true;
                            for (int dim=0; dim<NAD; dim++) {
                                feature[dim][goodevents] = f[dim];
                            }
                            NSread++;
                            goodevents++;
                        } 
                        if (s==0 && NBread<Nbackground) {
                            isSignal[goodevents] = false;
                            for (int dim=0; dim<NAD; dim++) {
                                feature[dim][goodevents] = f[dim];
                            }
                            NBread++;
                            goodevents++;
                        }
                    }
                } while (readok && (NSread<Nsignal || NBread<Nbackground)); // end while
                if (!readok) {
                    cout << "  Closing file " << dataname << endl;
                    file.close();
                    closed = true;
                    if (p==maxletter1-1 && q==maxletter2-1) { // reset files if we got to the end of the list
                        save_p = 0;
                        save_q = 0;
                        p = 0;
                        q = -1; // so that it goes back to 0,0
                    }
                    if (save_q==maxletter2-1) save_q = 0; //reset save_q value
                } else {
                    // ensure we exit loop, as we have read enough data
                    save_p = p;
                    save_q = q;
                    q = maxletter2-1;
                    p = maxletter1-1;
                }
            } // end q
        } // end p

      } else if (dataset==1) { // miniBooNE dataset

        // Start of loop
        // -------------
        int Nsiginfile, Nbgrinfile;
        dataname = "/lustre/cmswork/dorigo/RanBox/datafiles/MiniBooNE/miniBooNE_PID.txt";
        events.open(dataname);
        // First line has number of signal and background events contained in file
        events >> Nsiginfile >> Nbgrinfile;
        if (Nsignal>Nsiginfile || Nbackground>Nbgrinfile) {
            cout << "   Sorry file does not contain enough signal or background to satisfy request, exiting." << endl;
            return 0;
        }
        block = (Nsiginfile+Nbackground)/50; // file has all signal and then all background - we need to read all the signal always
        int Ntot = Nsignal;
        if (Nbackground>0) Ntot = Nsiginfile+Nbackground; 
        for (int i=0; i<Ntot; i++) {
            // Print out progress in nice format
            // ---------------------------------
            if (i%block == 0) {
                if (currchar > 52) currchar -= 53;
                cout << progress[currchar];
                currchar++;
            }
            float f[ND];
            bool voidit = true; // We have to prune events with -999 on all features from file
            for (int dim=0; dim<NAD; dim++) {
                events >> f[dim];
                if (f[dim]!=-0.999E+03) voidit = false; 
            }
            if (voidit) {
                Nvoided++;
                i--;
            } else {
                if (i<Nsiginfile && goodevents<Nsignal) {
                    for (int dim=0; dim<NAD; dim++) {
                        feature[dim][goodevents] = f[dim];
                    }
                    isSignal[goodevents] = true;
                    NSread++;
                    goodevents++;
                } else if (i>=Nsiginfile && goodevents<Nsignal+Nbackground) {
                    for (int dim=0; dim<NAD; dim++) {
                        feature[dim][goodevents] = f[dim];
                    }
                    isSignal[goodevents] = false;
                    NBread++;
                    goodevents++;
                }
            }
        } // End event loop
        events.close();
        goodevents = Nsignal+Nbackground;

      } else if (dataset==2) { // credit card data
        // Start of loop
        // -------------
        dataname = "/lustre/cmswork/dorigo/RanBox/datafiles/Fraud/creditcard.asc";
        file.open(dataname);
        bool readok;
        std::string data;
        std::string::size_type sz;
        readok = (bool) std::getline(file,data,'\n'); 	// First line has description of variables
        do {
        for (int dim=0; dim<NAD; dim++) {
            std::getline(file,data,',');
            f[dim] = std::stod(data,&sz);
        }
        std::getline(file,data,'"');
        std::getline(file,data,'"');
        double s = std::stod(data,&sz);
        std::getline(file,data,'\n');
        if (s==1 && NSread<Nsignal) {
            isSignal[goodevents]= true;
            for (int dim=0; dim<NAD; dim++) {
                feature[dim][goodevents] = f[dim];
            }
            NSread++;
            goodevents++;
        } 
        if (s==0 && NBread<Nbackground) {
            isSignal[goodevents] = false;
            for (int dim=0; dim<NAD; dim++) {
                feature[dim][goodevents] = f[dim];
            }
            NBread++;
            goodevents++;	
        }
        
        } while (NSread<Nsignal || NBread<Nbackground);

        file.close();
        goodevents = Nsignal+Nbackground;
      } else if (dataset==3) { // EEG data
        // Start of loop
        // -------------
        dataname = "/lustre/cmswork/dorigo/RanBox/datafiles/EEG/EEG_eye_detection.asc";
        file.open(dataname);
        bool readok;
        std::string data;
        std::string::size_type sz;
        do {
            for (int dim=0; dim<NAD; dim++) {
                std::getline(file,data,',');
                f[dim] = std::stod(data,&sz);
            }
            std::getline(file,data,'\n');
            double s = std::stod(data,&sz);
            if (s==1 && NSread<Nsignal) {
                isSignal[goodevents]= true;
                for (int dim=0; dim<NAD; dim++) {
                    feature[dim][goodevents] = f[dim];
                }
                NSread++;
                goodevents++;
            } 
            if (s==0 && NBread<Nbackground) {
                isSignal[goodevents] = false;
                for (int dim=0; dim<NAD; dim++) {
                    feature[dim][goodevents] = f[dim];
                }
                NBread++;
                goodevents++;	
            }
            
        } while (NSread<Nsignal || NBread<Nbackground);

        file.close();
        goodevents = Nsignal+Nbackground;
      } else if (dataset==5) { // LHC Olympics dataset

        // Start of loop
        // -------------
        dataname = "/lustre/cmswork/dorigo/RanBox/datafiles/LHCOlympics/events_anomalydetection_v2.features.csv";
        file.open(dataname);
        bool readok;
        std::string data;
        std::string::size_type sz;
        do {
        std::getline(file,data,','); // first int per line is event number
        for (int dim=0; dim<NAD; dim++) { 
            std::getline(file,data,',');
            f[dim] = std::stod(data,&sz);
        }
        std::getline(file,data,'\n');
        double s = std::stod(data,&sz);
        if (s==1 && NSread<Nsignal) {
            isSignal[goodevents]= true;
            for (int dim=0; dim<NAD; dim++) {
                feature[dim][goodevents] = f[dim];
            }
            NSread++;
            goodevents++;
        } 
        if (s==0 && NBread<Nbackground) {
            isSignal[goodevents] = false;
            for (int dim=0; dim<NAD; dim++) {
                feature[dim][goodevents] = f[dim];
            }
            NBread++;
            goodevents++;	
        }
        
        } while (NSread<Nsignal || NBread<Nbackground);

        file.close();
        goodevents = Nsignal+Nbackground;
      } else if (dataset==6) { // LHC Olympics qqq data
        // Start of loop
        // -------------
        dataname = "/lustre/cmswork/dorigo/RanBox/datafiles/LHCOlympics/events_anomalydetection_Z_XY_qqq.features.csv";
        file.open(dataname);
        bool readok;
        std::string data;
        std::string::size_type sz;
        do {
            std::getline(file,data,','); // first int per line is event number
            for (int dim=0; dim<NAD; dim++) { 
                std::getline(file,data,',');
                f[dim] = std::stod(data,&sz);
            }
            // In this file there is no S/B information! We use the kludge of generating the info at random
            // std::getline(file,data,'\n');
            // double s = std::stod(data,&sz);
            double s = 0;
            if (myRNG->Uniform()>Nbackground/(Nbackground+Nsignal)) s=1; // We draw s=0,1 with same proportion of data to be read
            if (s==1 && NSread<Nsignal) {
                isSignal[goodevents]= true;
                for (int dim=0; dim<NAD; dim++) {
                    feature[dim][goodevents] = f[dim];
                }
                NSread++;
                goodevents++;
            } 
            if (s==0 && NBread<Nbackground) {
                isSignal[goodevents] = false;
                for (int dim=0; dim<NAD; dim++) {
                    feature[dim][goodevents] = f[dim];
                }
                NBread++;
                goodevents++;	
            }
            
        } while (NSread<Nsignal || NBread<Nbackground);

        file.close();
        goodevents = Nsignal+Nbackground;

      }
      ///////////////////////////////////////////////////////////


      if (currchar<52) cout << progress[51]; 
      cout << endl << endl; // End of progress string

      cout << "  Nsignal = " << NSread << " Nbackground = " << NBread << endl;
      cout << "  Total: " << goodevents << " good events read" << endl;
      if (dataset==1) cout << "  N voided events read = " << Nvoided << endl;

      if (PCA) cout << "  "; // offsets printout of pca file
      
      // Initialize bounds in both frames
      double X0min[ND];
      double X0max[ND];
      for (int dim=0; dim<NAD; dim++) {
        X0min[dim] = bignumber;  
        X0max[dim] = -bignumber;
      }
      
      // Before doing PCA on real data, we need to transform all features 
      // such that they have equal span. This is because PCA is sensitive 
      // to the span of the variables
      // ----------------------------------------------------------------
      for (int i=0; i<goodevents; i++) {
        for (int dim=0; dim<NAD; dim++) {
            if (feature[dim][i]<X0min[dim]) X0min[dim] = feature[dim][i];
            if (feature[dim][i]>X0max[dim]) X0max[dim] = feature[dim][i];
        }
      }
      for (int i=0; i<goodevents; i++) {
            for (int dim=0; dim<NAD; dim++) {
                feature[dim][i] = (feature[dim][i]-X0min[dim])/(X0max[dim]-X0min[dim]);
            }
      }
      
      // Additional preprocessing step to avoid large discontinuities in support affecting the PCA step:
      // We loop on each feature and create N-bin histograms, then if a bin is empty we move all data
      // to the right of the bin down by the bin width, "filling the gap". This is iterated until all
      // empty space (within a coarseness of a factor 2^13 from original [0,1] support of standardized
      // features is removed from the support of the features.
      // ----------------------------------------------------------------------------------------------
      if (compactify) {
        int c[4096];
        for (int Niter=1; Niter<13; Niter++) {
            float binwidth=1./pow(2,Niter);
            for (int dim=0; dim<NAD; dim++) {
                for (int n=0; n<pow(2,Niter); n++) { c[n]=0; };
                for (int i=0; i<goodevents; i++) {
                    int ibin = (int)(feature[dim][i]/binwidth);
                    if (ibin==pow(2,Niter)) ibin -= 1; // avoid boundary problems
                    c[ibin]++;
                }
                for (int n=1; n<pow(2,Niter)-1; n++) {
                    if (c[n]==0) { // bin is empty, need to compactify data support
                        for (int i=0; i<goodevents; i++) {
                            if (feature[dim][i]>binwidth*n) feature[dim][i] -= binwidth*n;
                        }
                    }
                }
            }
        }
        // Redo standardization now that we have compactified the intervals
        // ----------------------------------------------------------------
        for (int dim=0; dim<NAD; dim++) {
            X0min[dim] = bignumber;
            X0max[dim] = -bignumber;
        }
        for (int i=0; i<goodevents; i++) {
            for (int dim=0; dim<NAD; dim++) {
                if (feature[dim][i]<X0min[dim]) X0min[dim] = feature[dim][i];
                if (feature[dim][i]>X0max[dim]) X0max[dim] = feature[dim][i];	  
            }
        }
        for (int i=0; i<goodevents; i++) {
            for (int dim=0; dim<ND; dim++) {
                feature[dim][i] = (feature[dim][i] - X0min[dim]) / (X0max[dim] - X0min[dim]);
            }
        }
      } // end if compactify
     
      if (PCA) {

        ////////////////////////////////////////////////////////////////////////////
        // Start of principal component analysis
        // -------------------------------------
        // Initialize bounds in both frames
        for (int dim=0; dim<ND; dim++) {
            Xmin[dim]  = bignumber;
            Xmax[dim]  = -bignumber;
            OXmin[dim] = bignumber;
            OXmax[dim] = -bignumber;
        }
        TPrincipal* principal = new TPrincipal(NAD,"NAD");
        Double_t * data = new Double_t[ND];
        for (int i=0; i<goodevents; i++) {
            for (int dim=0; dim<NAD; dim++) {
                data[dim] = feature[dim][i];
            }
            principal->AddRow(data);
        }
        delete [] data;
        principal->MakePrincipals();
        // principal->Test();
        // principal->MakeHistograms();
        principal->MakeCode();
        for (int i=0; i<goodevents; i++) {
            Double_t * dataP = new Double_t[ND];
            Double_t * dataX = new Double_t[ND];
            for (int dim=0; dim<NAD; dim++) {
                dataX[dim] = feature[dim][i];
            }
            X2P(dataX,dataP); // Do PCA transformation
            
            // Find min and max of each component in both systems
            for (int dim=0; dim<ND; dim++) {
                if (dataX[dim]<OXmin[dim]) OXmin[dim] = dataX[dim];
                if (dataX[dim]>OXmax[dim]) OXmax[dim] = dataX[dim];
                if (dataP[dim]<Xmin[dim])  Xmin[dim]  = dataP[dim];
                if (dataP[dim]>Xmax[dim])  Xmax[dim]  = dataP[dim];
                
                feature[dim][i] = dataP[dim]; // From now on, feature is the PCA transformed variable
            }
            delete [] dataP;
            delete [] dataX;
        }
        principal->~TPrincipal();
	
      } else {
        // We need these to be initialized even if we don't do PCA, for plots
        // ------------------------------------------------------------------
        for (int dim=0; dim<NAD; dim++) {
            Xmin[dim]  = 0.;
            Xmax[dim]  = 1.;
            OXmin[dim] = 0.;
            OXmax[dim] = 1.;
        }
      } // end if PCA or not
      
      // Now we consider only the leading NSEL of the NAD variables
      
      // -----------------------------------
      // End of principal component analysis
      ////////////////////////////////////////////////////////////////////////////
      
      // We may decide to remove the Nremoved<NAD variables which have the highest correlation, by identifying the
      // set which, once removed, leaves the smallest highest correlation among the remaining ones
      // ---------------------------------------------------------------------------------------------------------
      if (RemoveHighCorr) {
        // Compute means
        // -------------
        float meanvar[ND];
        for (int dim=0; dim<NAD; dim++) {
            meanvar[dim] = 0.;
        }
        for (int i=0; i<goodevents; i++) {
            for (int dim=0; dim<NAD; dim++) {
                meanvar[dim] += feature[dim][i];
            }
        }
        for (int dim=0; dim<NAD; dim++) {
            meanvar[dim] = meanvar[dim]/goodevents;
        }
        
        // Compute correlation coefficients
        // --------------------------------
        float corr[ND][ND];
        float sumsq[ND];
        for (int j=0; j<NAD; j++) {
            sumsq[j] = 0.;
            for (int k=0; k<NAD; k++) {
                corr[j][k] = 0.;
            }
        }
        for (int iev=0; iev<goodevents; iev++) {
            for (int j=0; j<NAD; j++) {
                sumsq[j] += pow(feature[j][iev]-meanvar[j],2);
                if (j<NAD-1) {
                    for (int k=j+1; k<NAD; k++) {
                        corr[j][k] += (feature[j][iev]-meanvar[j])*(feature[k][iev]-meanvar[k]);
                    }
                }
            }
        }
        int rhoindex = 0;
        float corrvec[ND*(ND-1)];
        int varind1[ND*(ND-1)];
        int varind2[ND*(ND-1)];
        for (int j=0; j<NAD; j++) {
            corr[j][j] = 1.;
            if (j<NAD-1) {
                for (int k=j+1; k<NAD; k++) {
                    if (sumsq[j]*sumsq[k]!=0) {
                        corr[j][k] = fabs(corr[j][k])/(sqrt(sumsq[j])*sqrt(sumsq[k]));
                    } else {
                        corr[j][k] = fabs(corr[j][k]);
                    }
                    corr[k][j] = fabs(corr[j][k]);
                    corrvec[rhoindex] = corr[j][k];
                    varind1[rhoindex] = j;
                    varind2[rhoindex] = k;
                    rhoindex++;
                }
            }
        }
        // Order vector of correlation coefficients
        // ----------------------------------------
        for (int times=0; times<rhoindex; times++) {
            for (int i=rhoindex-1; i>0; i--) {
                if (corrvec[i]>corrvec[i-1]) {
                    float tmp = corrvec[i];
                    corrvec[i] = corrvec[i-1];
                    corrvec[i-1] = tmp;
                    int itmp = varind1[i];
                    varind1[i] = varind1[i-1];
                    varind1[i-1] = itmp;
                    itmp = varind2[i];
                    varind2[i] = varind2[i-1];
                    varind2[i-1] = itmp;
                }
            }
        }
        for (int i=0; i<rhoindex; i++) {
            cout << "   Vars " << varind1[i] << "," << varind2[i] << ": rho = " << corrvec[i] << endl;
        }

        // New attempt
        // -----------
        int Nlists = pow(2,NAD-NSEL);
        int voidedind[ND];
        int voidedindbest[ND];
        int maxhighest=0;
        bool voidthis;
        for (int ilist=0; ilist<Nlists; ilist++) {
            int ihighest = 0;
            int Nvoided  = 0;
            do {
                do {
                    // Check if this rho is already excluded in this list
                    // --------------------------------------------------
                    voidthis = false;
                    // Loop on excluded variables from this list, to find first unexcluded element
                    // ---------------------------------------------------------------------------
                    for (int ivoided=0; ivoided<Nvoided; ivoided++) {
                        if (voidedind[ivoided]==varind1[ihighest] || 
                            voidedind[ivoided]==varind2[ihighest]) voidthis = true;
                    }
                    if (voidthis) ihighest++;
                } while (voidthis && ihighest<rhoindex);
                if (ihighest==rhoindex) continue; // check later
                // Ok, we found a living corr. coefficient. Now we
                // increment the list 
                // -----------------------------------------------
                if (bitIsOn(ilist,Nvoided)) {
                   voidedind[Nvoided] = varind2[ihighest];
                } else {
                    voidedind[Nvoided] = varind1[ihighest];
                }
                Nvoided++;
            } while (Nvoided<NAD-NSEL);
            // Check if this list has the best result
            // --------------------------------------
            if (ihighest>maxhighest) {
                maxhighest = ihighest;
                for (int iv=0; iv<Nvoided; iv++) {
                    voidedindbest[iv] = voidedind[iv];
                }
            }
        } // end ilist

        // So now the variables to remove are those identified by voidedind[0:Nvoided-1][bestlist].
        // Hence we compactify them, removing NAD-NSEL of them
        // ----------------------------------------------------------------------------------------
        bool kept[ND];
        for (int dim=0; dim<NAD; dim++) {
            kept[dim] = true;
            for (int iv=0; iv<NAD-NSEL; iv++) {
                if (voidedindbest[iv]==dim) kept[dim] = false;
            }
        }
        cout << "   Correlated variables removal:" << endl;
        cout << "   -----------------------------" << endl;
        cout << "   Number of voided variables = " << NAD-NSEL << " Voided correlations = " << maxhighest 
             << " max corr = " << corrvec[maxhighest] << endl;
        cout << "   Removed variables: ";
        for (int i=0; i<NAD-NSEL; i++) {
            cout << voidedindbest[i] << " ";
            //if (!kept[i]) cout << i << " ";
        }
        cout << endl << endl;
        int nkept = -1;
        for (int dim=0; dim<NAD; dim++) {
            if (kept[dim]) {
                nkept++;
                for (int i=0; i<goodevents; i++) {
                feature[nkept][i] = feature[dim][i];
                }
            }
        }

      } // end RemoveHigCorr loop

      cout << endl;
      // Now we consider only the leading NSEL of the NAD variables
      // --------------------------------------------------------
      // End of variable selection / principal component analysis
      ////////////////////////////////////////////////////////////////////////////

      // Sort data
      // ---------
      cout << "  Sorting data:     " << progress[0];
      currchar = 1;
      int itmp;
      double dtmp;
      for (int dim=0; dim<NAD; dim++) {
        // Print out progress in nice format
        // ---------------------------------
        if (currchar>52) currchar -= 53;
        cout << progress[currchar];
        currchar++;
        
        for (int i=0; i<goodevents; i++) {
        xtmp[i] = feature[dim][i];
        cum_ind[i] = i;
        }
        for (int times=0; times<goodevents; times++) {
            for (int i=goodevents-1; i>0; i--) {
                if (xtmp[i]<xtmp[i-1]) {
                    dtmp      = xtmp[i-1];
                    xtmp[i-1] = xtmp[i];
                    xtmp[i]   = dtmp;
                    itmp         = cum_ind[i-1];
                    cum_ind[i-1] = cum_ind[i];
                    cum_ind[i]   = itmp;
                }
            }
        
        } // End times loop
        for (int i=0; i<goodevents; i++) {
            order_ind[dim][i] = -1;
            for (int j=0; j<goodevents && order_ind[dim][i]==-1; j++) {
                if (cum_ind[j]==i) {
                order_ind[dim][i] = j;
                }
            }
            }
      } // End of sorting loop
      for (int c=currchar; c<52; c++) cout << progress[c]; 
      cout << endl << endl; // End of progress string
      
    } // End if !mock (loop on data and standardization, which for flat mock data is superfluous)
    // ------------------------------------------------------------------------------------------

    // Start of trials
    // ---------------
    for (int trial=0; trial<Ntrials; trial++) {
      
      // Print out progress in nice format
      // ---------------------------------
      // if (Ntrials>50) {
      //   if (trial%block==0) {
      //	if (currchar>52) currchar-=53;
      //  cout << progress[currchar];
      //	currchar++;
      //}
      //} else {
      //  cout << progress[currchar];
      //  currchar++;
      //}
      
      // Decide what variables will be looked at in this trial
      // -----------------------------------------------------
      
      if (one_off) {
        // Fix ivar list, for debug purposes
        for (int k=0; k<Nvar; k++) {
            Ivar[k] = Ivar_fix[k];
        }
      } else if (doall) { // scan all possible combinations of Nvar in NSEL
        if (trial>0) { // // the first trial has ivar already set	  
            if (Ivar[Nvar-1]<NSEL-1) {
                Ivar[Nvar-1]++; // this is all in this case
            } else {
                bool onemore = true;
                // if one of the digits has reached its maximum it's time to move 
                // up the previous digit and reset all following ones
                for (int k=1; k<Nvar && onemore; k++) { 
                    if (Ivar[k]==NSEL-1-(Nvar-1-k)) { // it's at the end
                    Ivar[k-1]++;
                    for (int kk=k; kk<Nvar; kk++) { Ivar[kk] = Ivar[kk-1]+1; };
                        onemore = false;
                    }
                }
            }
        }
        // print variable list    
        for (int k=0; k<Nvar; k++) { cout << " " << Ivar[k];};
        cout << endl;
      }	else if (NseedTrials==1) {
        bool used[ND];
        for (int k=0; k<ND; k++) { 
            used[k] = false; 
        }
        for (int k=0; k<Nvar; k++) {
            if (mock && PCA==false && Nsignal>0 && force_gaussians==true) { // Pick subspace with gaussians
                Ivar[k] = (int)gRandom->Uniform(0.,Gaussian_dims-smallnumber);
            } else { // Pick subspace at random
                Ivar[k] = (int)gRandom->Uniform(0.,NSEL-smallnumber);
            }
            if (!used[Ivar[k]]) {
                used[Ivar[k]] = true;
            } else {
                k = k-1; // This can only happen for k>0
                continue;
            }
        }
      } else { // If NseedTrials>1 we test the same condition with different settings
        for (int k=0; k<Nvar; k++) {
            Ivar[k] = k;
        }
      }
      
      ////////////////////////////////
      // Choose initial box
      // -----------------------------
      
      double BmiIn[maxNvar]; // Track initial left bounds of signal box
      double BmaIn[maxNvar]; // Track initial right bounds of signal box
      double Bmin[maxNvar][7];
      double Bmax[maxNvar][7];
      double Smin[maxNvar][7][maxNvar]; // For each direction there are seven moves, and to each correspond a sideband
      double Smax[maxNvar][7][maxNvar];     
      
      //////////////////////////////
      // Box initialization
      // ------------------
      
      for (int i=0; i<goodevents; i++) { AmClosest[i]=0; };
      double position_i[maxNvar];
      double VolumeOrig;
      double SidebandsVolume;
      sidewidth = 0.5*(pow(2,1./Nvar)-1.); // the sideband is in Nvar dimensions. 
      
      // Choose initial box around these vars  
      // ------------------------------------
      if (Algorithm==0) {

        // - The random way      
        // ----------------
        // Random interval in [0,1] has length 1/3
        // we choose random box to have expected volume
        // such that on average 10 events are in it.
        // This means using [0,1] range in all coords but
        // using x = log(10/Ntot) / log(1/3) random intervals
        // --------------------------------------------------
        double Nvar_box = (log(10./goodevents)/log(1./3));
        if (Nvar_box>Nvar) Nvar_box = Nvar;
        VolumeOrig = 1.;
        int nonfull = (int)gRandom->Uniform(0.,Nvar-epsilon);
        for (int k=0; k<Nvar; k++) {

            // The one below ensures that a move may catch a signal even if it's in the boundaries.
            if (k==nonfull) {
                Blockmin[k] = epsilon*(int)(InvEpsilon*gRandom->Uniform(0.,0.5));
                Blockmax[k] = 0.5+Blockmin[k];
                VolumeOrig  = VolumeOrig*fabs(Blockmax[k]-Blockmin[k]);
            } else {
                Blockmin[k] = 0.;
                Blockmax[k] = 1.;
            }
        }

      } else if (Algorithm==1) {
	
        // Algorithm 1
        // -----------
        
        for (int i=0; i<goodevents; i+=speedup) {
            double d2min = bignumber;
            for (int k=0; k<Nvar; k++) {
                position_i[k] = (0.5+(double)order_ind[Ivar[k]][i])/goodevents;
            }
            for (int ii=0; ii<goodevents; ii+=speedup) {
                if (ii==i) continue;
                if (AmClosest[ii]>maxClosest-2) continue; // We cap the max associations
                double d2[maxNvar];
                for (int k=0; k<Nvar; k++) {
                    double position_j = (0.5+(double)order_ind[Ivar[k]][ii])/goodevents;
                    d2[k] = pow(position_i[k]-position_j,2);
                }
                double sumd2 = 0.;
                double tmp;
                for (int times=0; times<Nvar; times++) {
                    for (int m=Nvar-1; m>0; m--) {
                        if (d2[m]<d2[m-1]) {
                            tmp     = d2[m-1];
                            d2[m-1] = d2[m];
                            d2[m]   = tmp;
                        }
                    }
                }
                // Only consider the half of coordinates where the points are closest
                for (int m=0; m<Nvar/2; m++) { sumd2 += d2[m];};
                if (sumd2<d2min) {
                    d2min = sumd2;
                    Closest[i] = ii;
                }
            }
            if (AmClosest[Closest[i]]<maxClosest-1) {
                PointedBy[AmClosest[Closest[i]]][Closest[i]] = i;
                AmClosest[Closest[i]]++;
            }
        }
        // Compute how crowded is the area around each event, using AmClosest as metric
        double Nclomax = 0;
        int Imax       = -1;
        for (int ii=0; ii<goodevents; ii+=speedup) {
            int Nclo = 0;
            for (int m=0; m<AmClosest[ii]; m++) {
                Nclo += AmClosest[PointedBy[m][ii]];
            }
            if (Nclo>Nclomax) {
                Nclomax = Nclo;
                Imax    = ii;
            }
        }
        NClomax->Fill((double)Nclomax);
        for (int k=0; k<Nvar; k++) {
            Blockmin[k] = 1.;
            Blockmax[k] = 0.;
        }
        //Now use box dimensions given by neighbors location
        VolumeOrig = 1.;
        for (int k=0; k<Nvar; k++) {
            for (int m=0; m<AmClosest[Imax]; m++) {
                for (int mm=0; mm<AmClosest[PointedBy[m][Imax]]; mm++) {
                    int index = PointedBy[mm][PointedBy[m][Imax]]; // or use PointedBy[m][Imax] and no mm loop
                    double pos_this = (0.5+(double)order_ind[Ivar[k]][index])/goodevents;
                    if (Blockmin[k]>pos_this) {
                        Blockmin[k] = pos_this;
                    } 
                    if (Blockmax[k]<pos_this) {
                        Blockmax[k] = pos_this;
                    }
                }
            }
            // Round off boundaries
            Blockmin[k] = epsilon*(int)(InvEpsilon*(Blockmin[k]-epsilon));
            if (Blockmin[k]<0.) Blockmin[k] = 0.;
            Blockmax[k] = epsilon*(int)(InvEpsilon*(Blockmax[k]+epsilon));
            if (Blockmax[k]>1.) Blockmax[k] = 1.;
            VolumeOrig = VolumeOrig*fabs(Blockmax[k]-Blockmin[k]);
        } // end k
	
      } else if (Algorithm==2) { // end algorithm 1
	
        // Algorithm 2
        // -----------
        // Here we try to minimize the volume of collective neighbors
        // by using as association criterion the linear sum of components
        
        for (int i=0; i<goodevents; i+=speedup) {
        double d1min = bignumber;
        for (int k=0; k<Nvar; k++) {
            position_i[k] = (0.5+(double)order_ind[Ivar[k]][i])/goodevents;
        }
        for (int ii=0; ii<goodevents; ii+=speedup) {
            if (ii==i) continue;
            if (AmClosest[ii]>maxClosest-2) continue; // We cap the max associations
            double d1[maxNvar];
            for (int k=0; k<Nvar; k++) {
            double position_j = (0.5+(double)order_ind[Ivar[k]][ii])/goodevents;
            d1[k] = fabs(position_i[k]-position_j);
            }
            double sumd1 = 0;
            double tmp;
            for (int times=0; times<Nvar; times++) {
            for (int m=Nvar-1; m>0; m--) {
                if (d1[m]<d1[m-1]) {
                tmp     = d1[m-1];
                d1[m-1] = d1[m];
                d1[m]   = tmp;
                }
            }
            }
            // Only consider the half of coordinates where the points are closest
            for (int m=0; m<Nvar/2; m++) { sumd1 += d1[m];};
            if (sumd1<d1min) {
            d1min = sumd1;
            Closest[i] = ii;
            }
        }
        if (AmClosest[Closest[i]]<maxClosest-1) {
            PointedBy[AmClosest[Closest[i]]][Closest[i]] = i;
            AmClosest[Closest[i]]++;
        }
        }
        // Compute how crowded is the area around each event, using AmClosest as metric
        int Nclomax = 0;
        int Imax = -1;
        for (int i=0; i<goodevents; i+=speedup) {
        int Nclo = 0;
        for (int m=0; m<AmClosest[i]; m++) {
            Nclo += AmClosest[PointedBy[m][i]];
        }
        if (Nclo>Nclomax) {
            Nclomax = Nclo;
            Imax = i;
        }
        }
        NClomax->Fill((double)Nclomax);
        for (int k=0; k<Nvar; k++) {
        Blockmin[k] = 1.;
        Blockmax[k] = 0.;
        }
        //Now use box dimensions given by neighbors location
        VolumeOrig = 1.;
        for (int k=0; k<Nvar; k++) {
        for (int m=0; m<AmClosest[Imax]; m++) {
            for (int mm=0; mm<AmClosest[PointedBy[m][Imax]]; mm++) {
            int index = PointedBy[mm][PointedBy[m][Imax]]; // PointedBy[m][Imax] and no mm loop
            double pos_this = (0.5+(double)order_ind[Ivar[k]][index])/goodevents;
            if (Blockmin[k]>pos_this) {
                Blockmin[k] = pos_this;
            } 
            if (Blockmax[k]<pos_this) {
                Blockmax[k] = pos_this;
            }
            }
        }
        // Round off boundaries
        Blockmin[k] = epsilon*(int)(InvEpsilon*(Blockmin[k]-epsilon));
        if (Blockmin[k]<0.) Blockmin[k] = 0.;
        Blockmax[k] = epsilon*(int)(InvEpsilon*(Blockmax[k]+epsilon));
        if (Blockmax[k]>1.) Blockmax[k] = 1.;
        VolumeOrig = VolumeOrig*fabs(Blockmax[k]-Blockmin[k]);
        }
	
      } else if (Algorithm==3) {
	
        // Algorithm 3
        // -----------
        // Here we use the volume as metric of distance
        
        for (int i=0; i<goodevents; i+=speedup) {
            double dxmin = bignumber;
            for (int k=0; k<Nvar; k++) {
                position_i[k] = (0.5+(double)order_ind[Ivar[k]][i])/goodevents;
            }
            for (int ii=0; ii<goodevents; ii+=speedup) {
                if (ii==i) continue;
                if (AmClosest[ii]>maxClosest-2) continue; // We cap the max associations
                double dx[maxNvar];
                for (int k=0; k<Nvar; k++) {
                    double position_j = (0.5+(double)order_ind[Ivar[k]][ii])/goodevents;
                    dx[k] = fabs(position_i[k]-position_j);
                    if (dx[k]<epsilon) dx[k] = epsilon;
                }
                double proddx = 1.;
                double tmp;
                for (int times=0; times<Nvar; times++) {
                    for (int m=Nvar-1; m>0; m--) {
                        if (dx[m]<dx[m-1]) {
                            tmp     = dx[m-1];
                            dx[m-1] = dx[m];
                            dx[m]   = tmp;
                        }
                    }
                }
                // Only consider the half of coordinates where the points are closest
                for (int m=0; m<Nvar/2; m++) { proddx *= dx[m]; };
                if (proddx<dxmin) {
                    dxmin      = proddx;
                    Closest[i] = ii;
                }
            } // end ii loop
            if (AmClosest[Closest[i]]<maxClosest-1) {
                PointedBy[AmClosest[Closest[i]]][Closest[i]] = i;
                AmClosest[Closest[i]]++; // it can get to be equal to maxClosest-1, length of dimension 0 of pointedby[..][]
            }
        } // end i loop

        // Compute how crowded is the area around each event, using AmClosest as metric
        int Nclomax = 0;
        int Imax    = -1;
        for (int ii=0; ii<goodevents; ii+=speedup) {
            int Nclo = 0;
            for (int m=0; m<AmClosest[ii]; m++) {
                Nclo += AmClosest[PointedBy[m][ii]];
            }
            if (Nclo>Nclomax) {
                Nclomax = Nclo;
                Imax    = ii;
            }
        }
        NClomax->Fill((double)Nclomax);
        for (int k=0; k<Nvar; k++) {
            Blockmin[k] = 1.;
            Blockmax[k] = 0.;
        }
        // Now use box dimensions given by neighbors location
        VolumeOrig = 1.;
        for (int k=0; k<Nvar; k++) {
            for (int m=0; m<AmClosest[Imax]; m++) {
                for (int mm=0; mm<AmClosest[PointedBy[m][Imax]]; mm++) {
                    int index = PointedBy[mm][PointedBy[m][Imax]]; // PointedBy[m][Imax] and no mm loop
                    double pos_this = (0.5+(double)order_ind[Ivar[k]][index])/goodevents;
                    if (Blockmin[k]>pos_this) {
                        Blockmin[k] = pos_this;
                    } 
                    if (Blockmax[k]<pos_this) {
                        Blockmax[k] = pos_this;
                    }
                }
            }
            // Round off boundaries
            Blockmin[k] = epsilon*(int)(InvEpsilon*(Blockmin[k]-epsilon));
            if (Blockmin[k]<0.) Blockmin[k] = 0.;
            Blockmax[k] = epsilon*(int)(InvEpsilon*(Blockmax[k]+epsilon));
            if (Blockmax[k]>1.) Blockmax[k] = 1.;
            VolumeOrig = VolumeOrig*fabs(Blockmax[k]-Blockmin[k]);
        } // end k
	
      } else if (Algorithm==4) { // end algorithm 3
	
        // Algorithm 4: minimize largest direction
        // ---------------------------------------
        for (int i=0; i<goodevents; i+=speedup) {
            double d0min = bignumber;
            for (int k=0; k<Nvar; k++) {
                position_i[k] = (0.5+(double)order_ind[Ivar[k]][i])/goodevents;
            }
            for (int ii=0; ii<goodevents; ii+=speedup) {
                if (ii==i) continue;
                if (AmClosest[ii]>maxClosest-2) continue; // We cap the max associations
                double d0max_dir = 0;
                for (int k=0; k<Nvar; k++) {
                    double position_j = (0.5+(double)order_ind[Ivar[k]][ii])/goodevents;
                    double d0 = fabs(position_i[k]-position_j);
                    if (d0>d0max_dir) d0max_dir = d0;
                }
                if (d0max_dir<d0min) {
                    d0min = d0max_dir;
                    Closest[i] = ii;
                }
            }
            if (AmClosest[Closest[i]]<maxClosest-1) {
                PointedBy[AmClosest[Closest[i]]][Closest[i]] = i;
                AmClosest[Closest[i]]++; 
            }
        }
        // Compute how crowded is the area around each event, using AmClosest as metric
        int Nclomax = 0;
        int Imax    = -1;
        for (int i=0; i<goodevents; i+=speedup) {
            int Nclo = 0;
            for (int m=0; m<AmClosest[i]; m++) {
                Nclo += AmClosest[PointedBy[m][i]];
            }
            if (Nclo>Nclomax) {
                Nclomax = Nclo;
                Imax    = i;
            }
        }
        NClomax->Fill((double)Nclomax);
        for (int k=0; k<Nvar; k++) {
            Blockmin[k] = 1.;
            Blockmax[k] = 0.;
        }
        // Now use box dimensions given by neighbors location
        VolumeOrig = 1.;
        for (int k=0; k<Nvar; k++) {
            for (int m=0; m<AmClosest[Imax]; m++) {
                for (int mm=0; mm<AmClosest[PointedBy[m][Imax]]; mm++) {
                    int index = PointedBy[mm][PointedBy[m][Imax]]; // PointedBy[m][Imax] and no mm loop
                    double pos_this = (0.5+(double)order_ind[Ivar[k]][index])/goodevents;
                    if (Blockmin[k]>pos_this) {
                        Blockmin[k] = pos_this;
                    } 
                    if (Blockmax[k]<pos_this) {
                        Blockmax[k] = pos_this;
                    }
                }
            }
            // Round off boundaries
            Blockmin[k] = epsilon*(int)(InvEpsilon*(Blockmin[k]-epsilon));
            if (Blockmin[k]<0.) Blockmin[k] = 0.;
            Blockmax[k] = epsilon*(int)(InvEpsilon*(Blockmax[k]+epsilon));
            if (Blockmax[k]>1.) Blockmax[k] = 1.;
            VolumeOrig = VolumeOrig*fabs(Blockmax[k]-Blockmin[k]);
        }
	
      } else if (Algorithm==5) {
	
        // Algorithm 5 - choose box center as point with max local density, 
        // density is computed using Gaussian kernels
        // ----------------------------------------------------------------
        double maxSum     = 0.;
        int inbest        = -1;
        double halfwidth  = 0.2;
        double d2[maxNvar];
        double sumd2;
        for (int i=0; i<goodevents; i+=speedup) {
            double SumKernels = 0.;
            for (int k=0; k<Nvar; k++) {
                position_i[k] = (0.5+(double)order_ind[Ivar[k]][i])/goodevents;
            }
            for (int ii=0; ii<goodevents; ii+=speedup) {
                sumd2 = 0.;
                if (ii==i) continue;
                for (int k=0; k<Nvar; k++) {
                    double position_j = (0.5+(double)order_ind[Ivar[k]][ii])/goodevents;
                    d2[k] = pow(position_i[k]-position_j,2);
                    sumd2 += d2[k];
                }
                SumKernels += exp(-sumd2/D2kernel); 
            } // end ii loop
            if (SumKernels>maxSum) {
                maxSum = SumKernels;
                inbest = i;
            }
        }
        // Now use fixed box dimensions around chosen point
        VolumeOrig = 1.;
        for (int k=0; k<Nvar; k++) {
            double pos_this = (0.5+(double)order_ind[Ivar[k]][inbest])/goodevents;
            Blockmin[k] = epsilon*(int)(InvEpsilon*(pos_this-halfwidth));
            Blockmax[k] = epsilon*(int)(InvEpsilon*(pos_this+halfwidth));
            if (Blockmin[k]<0.) {
                Blockmin[k] = 0.;
                Blockmax[k] = 2*halfwidth;
            } else if (Blockmax[k]>1.) {
                Blockmin[k] = 1.-2*halfwidth;
                Blockmax[k] = 1.;
            }
            VolumeOrig *= fabs(Blockmax[k]-Blockmin[k]);
        } 
      } else {
          cout << "  Choice of algorithm not allowed. Exiting. " << endl;
          return 0;
      } // end choice of algorithm
        
      // If Volume is too large, reduce uniformly
      // Delta' = Delta/RF
      // RF = (Vol/0.5)^(1/N)
      // x'max,min = xmax,min -+ delta
      // Delta' = Delta - 2 delta
      // from which delta = 0.5Delta(1-1/RF)
      // -----------------------------------------
      if (VolumeOrig>0.5) {
        double redfactor = pow(VolumeOrig/0.5,1./Nvar);
        for (int k=0; k<Nvar; k++) {
            double Delta = Blockmax[k]-Blockmin[k];
            double delta = 0.5*Delta*(1.-1./redfactor);
            Blockmin[k] = Blockmin[k] + delta;
            Blockmax[k] = Blockmax[k] - delta;
            // Round off boundaries
            Blockmin[k] = epsilon*(int)(InvEpsilon*(Blockmin[k]-epsilon));
            if (Blockmin[k]<0.) Blockmin[k] = 0.;
            Blockmax[k] = epsilon*(int)(InvEpsilon*(Blockmax[k]+epsilon));
            if (Blockmax[k]>1.) Blockmax[k] = 1.;
            if (Blockmin[k]==Blockmax[k]) {
                if (Blockmin[k]>0) {
                    Blockmin[k] = Blockmin[k]-epsilon;
                } else {
                Blockmax[k] = Blockmax[k]+epsilon;
                } 
            }
            VolumeOrig *= fabs(Blockmax[k]-Blockmin[k]);
        }
      }

      if (debug) {
        for (int k=0; k<Nvar; k++) {
        cout << "  " << k << "th dir: " << Ivar[k] << "- initial bounds [" << Blockmin[k] 
            <<"," << Blockmax[k] << "]";
        if (mock) {
            if (Gaussian[Ivar[k]]) {
            cout << " - gen: G(" << mean[Ivar[k]] << "," << sigma[Ivar[k]] << ")" << endl;
            } else {
            cout << " - gen: Flat direction " << endl;
            }
        }
        }
      }
      
      ////////////////////////////////////// End of box initialization 
      
      for (int k=0; k<Nvar; k++) {
        // Store initial values to compute Drift histogram of boundaries due to GD minimization
        BmiIn[k] = Blockmin[k];
        BmaIn[k] = Blockmax[k];
        Bounds_in->Fill(Blockmin[k],Blockmax[k]);
      }
      
      // Initialize variables before loop over data
      // ------------------------------------------
      double lambda[maxNvar][6];
      // double lambdatmp[maxNvar][4];
      for (int k=0; k<Nvar; k++) {
        for (int m=0; m<6; m++) {
        lambda[k][m] = InitialLambda; // 0.2;
        // lambdatmp[k][m] = 0.2;
        }
      }
      int igrad       = -1;
      int dirgrad     = 0;
      double Zvalbest = -bignumber;
      int Ninbest;
      int Nsidebest;
      double Nexpbest;
      double S0[maxNvar];
      double S1[maxNvar];
      double B0[maxNvar];
      double B1[maxNvar];
      int Nin;
      double Nexp;
      int Nin0 = 0;
      int jumps[maxNvar];
      int Nin_grad[maxNvar][7]; // events in box moved right or left
      int Nside; // sideband events 
      int Nside_grad[maxNvar][7];    // sideband events for each feature, corr. to moved box right or left
      double VolumeMod[maxNvar][7];  // modified box volume following a step
      int donedir[maxNvar];
      for (int k=0; k<Nvar; k++) {
        donedir[k] = false;
        jumps[k]   = 0;
      }
      bool doloop; // control if we can get out of gd loop
      bool okmove[maxNvar][7]; // control if a move is allowed
      
      int gd2; // use it to keep track of number of loops
      double Initial_density = 0.;

      //////////////////////////////////////////////////////////////////////////
      // Gradient descent loop
      // ---------------------
      for (int gd=0; gd<maxGDLoops; gd++) {
	
        determineSB(Sidemin,Sidemax,Blockmin,Blockmax,Nvar);
        VolumeOrig      = 1.;
        SidebandsVolume = 1.;
        for (int k=0; k<Nvar; k++) {
            VolumeOrig      *= fabs(Blockmax[k]-Blockmin[k]);
            SidebandsVolume *= fabs(Sidemax[k]-Sidemin[k]);
        }
        doloop = false;
        gd2    = gd;
        Nin    = 0;   // events in original box
        Nside  = 0;   // events in sideband
        Nexp   = 0;   // expected events from volume considered (which may differ from boxvol)
        for (int k=0; k<Nvar; k++) {
            for (int m=0; m<7; m++) {
                okmove[k][m]     = true;
                Nin_grad[k][m]   = 0; 
                Nside_grad[k][m] = 0; 
                Bmin[k][m] = 0.;
                Bmax[k][m] = 1.;
            }	    
            if (lambda[k][0]>epsilon || lambda[k][1]>epsilon || 
                lambda[k][2]>epsilon || lambda[k][3]>epsilon || 
                lambda[k][4]>epsilon || lambda[k][5]>epsilon) doloop = true; 
            // if all lambdas are small we will get out
            }
            
            // Get out if this is off
            // ----------------------
            if (!doloop) {
                gd = maxGDLoops;
                continue;
            }
            
            // Compute modified boundaries and volumes following a move in box space 
            // ---------------------------------------------------------------------
            for (int k=0; k<Nvar; k++) { 
                if (donedir[k]) continue;
                double multiplier = VolumeOrig/(Blockmax[k]-Blockmin[k]);
                
                if (Blockmin[k]==0.) {
                    okmove[k][0] = false;
                } else {
                    Bmin[k][0] = epsilon*(int)(InvEpsilon*(Blockmin[k]-lambda[k][0]));
                    if (Bmin[k][0]<0.) Bmin[k][0] = 0.;
                    Bmax[k][0] = Blockmax[k];
                    VolumeMod[k][0] = multiplier*(Bmax[k][0]-Bmin[k][0]);
                    if (VolumeMod[k][0]>maxBoxVolume) okmove[k][0] = false;
                }
                
                Bmin[k][1] = epsilon*(int)(InvEpsilon*(Blockmin[k]+lambda[k][1]));
                Bmax[k][1] = Blockmax[k];
                if (Bmin[k][1]>Bmax[k][1]-epsilon) Bmin[k][1] = Bmax[k][1]-epsilon;
                VolumeMod[k][1] = multiplier*(Bmax[k][1]-Bmin[k][1]);
                if (VolumeMod[k][1]>maxBoxVolume) okmove[k][1] = false;	    
                
                Bmax[k][2] = epsilon*(int)(InvEpsilon*(Blockmax[k]-lambda[k][2]));
                Bmin[k][2] = Blockmin[k];
                if (Bmax[k][2]<Bmin[k][2]+epsilon) Bmax[k][2] = Bmin[k][2]+epsilon;
                VolumeMod[k][2] = multiplier*(Bmax[k][2]-Bmin[k][2]);
                if (VolumeMod[k][2]>maxBoxVolume) okmove[k][2] = false;	    
                
                if (Blockmax[k]==1.) {
                    okmove[k][3] = false;
                } else {
                    Bmax[k][3] = epsilon*(int)(InvEpsilon*(Blockmax[k]+lambda[k][3]));
                    Bmin[k][3] = Blockmin[k];
                    if (Bmax[k][3]>1.) Bmax[k][3] = 1.;
                    VolumeMod[k][3] = multiplier*(Bmax[k][3]-Bmin[k][3]);
                    if (VolumeMod[k][3]>maxBoxVolume) okmove[k][3] = false;	    
                }
                
                if (Blockmin[k]==0.) {
                    okmove[k][4] = false;
                } else {
                    Bmax[k][4] = epsilon*(int)(InvEpsilon*(Blockmax[k]-lambda[k][4]));
                    Bmin[k][4] = epsilon*(int)(InvEpsilon*(Blockmin[k]-lambda[k][4]));
                    if (Bmin[k][4]<0.) Bmin[k][4] = 0.;
                    if (Bmax[k][4]<=Bmin[k][4]) Bmax[k][4] = Bmin[k][4]+epsilon;
                    VolumeMod[k][4] = multiplier*(Bmax[k][4]-Bmin[k][4]);
                    if (VolumeMod[k][4]>maxBoxVolume) okmove[k][4] = false;	    
                }
                
                if (Blockmax[k]==1.) {
                    okmove[k][5] = false;
                } else {
                    Bmax[k][5] = epsilon*(int)(InvEpsilon*(Blockmax[k]+lambda[k][5]));
                    Bmin[k][5] = epsilon*(int)(InvEpsilon*(Blockmin[k]+lambda[k][5]));
                    if (Bmax[k][5]>1.) Bmax[k][5] = 1.;
                    if (Bmin[k][5]>=Bmax[k][5]) Bmin[k][5] = Bmax[k][5]-epsilon;
                    VolumeMod[k][5] = multiplier*(Bmax[k][5]-Bmin[k][5]);
                    if (VolumeMod[k][5]>maxBoxVolume) okmove[k][5] = false;	    
                }
                
                // Random jump of boundaries
                do {
                    Bmin[k][6] = epsilon*(int)(InvEpsilon*(gRandom->Uniform(0.,1.-epsilon)));
                    Bmax[k][6] = epsilon*(int)(InvEpsilon*(gRandom->Uniform(epsilon,1.)));
                } while (Bmax[k][6]<Bmin[k][6]+epsilon);
                VolumeMod[k][6] = multiplier*(Bmax[k][6]-Bmin[k][6]);
                if (VolumeMod[k][6]>maxBoxVolume) okmove[k][6] = false;	    	
            } // end k

            if (debug) {
            for (int k=0; k<Nvar; k++) {
                for (int m=0; m<7; m++) {
                    if (okmove[k][m]) {
                        cout << "k,m= " << k << " " << m << " Bmin,max = [" 
                             << Bmin[k][m] << "," << Bmax[k][m] << "]" << endl;
                    }
                }
            }
        } // end k
        
        // Determine sidebands
        // We are investigating Nvar*7 different moves -> as many box boundaries
        // ---------------------------------------------------------------------
        for (int k=0; k<Nvar; k++) {
            if (donedir[k]) continue;
            for (int m=0; m<7; m++) {
                if (okmove[k][m]) {
                    B0[k] = Bmin[k][m];
                    B1[k] = Bmax[k][m];
                    double VolMod = B1[k]-B0[k];
                    for (int k2=0; k2<Nvar; k2++) {
                        if (k2!=k) { // not a move direction
                        B0[k2] = Blockmin[k2];
                        B1[k2] = Blockmax[k2];
                        VolMod *= (Blockmax[k2]-Blockmin[k2]);
                        }
                    }
                    determineSB(S0,S1,B0,B1,Nvar);
                    double SideVolMod = 1.;
                    for (int k2=0; k2<Nvar; k2++) {
                    Smin[k][m][k2] = S0[k2];
                    Smax[k][m][k2] = S1[k2];
                    SideVolMod *= (S1[k2]-S0[k2]);
                    }
                } // if okmove
            } // end m
        } // end k
        
        int in_;
        int side_;
        bool in[maxNvar];
        bool in_grad[maxNvar][7];
        bool side[maxNvar];
        int ns[maxNvar][7];

        // Loop on all data
        // ----------------
        for (int i=0; i<goodevents; i++) {	  
            in_   = 0;   // number of dimensions along which event is inside signal box
            side_ = 0;   // dimensions along which event is in sideband box
            for (int k=0; k<Nvar; k++) {
                in[k]           = false;
                side[k]         = false;
                for (int m=0; m<7; m++) {
                    in_grad[k][m] = false;
                    ns[k][m]      = 0;
                }
                
                // Determine if event is in starting box or sideband
                // -------------------------------------------------
                double position = (0.5+order_ind[Ivar[k]][i])/goodevents; // pos of ev in this var
                if (position>Blockmin[k] && position<=Blockmax[k]) { 
                    in[k] = true; // this var is in 1d-box
                    in_++;
                } 
                if (position>Sidemin[k] && position<=Sidemax[k]) {
                    side[k] = true; // this var is in sideband in this direction
                    side_++;
                }
                for (int m=0; m<7; m++) {
                    if (position>Bmin[k][m] && position<=Bmax[k][m]) {
                        in_grad[k][m] = true; 
                    } 
                }	    
            } // k loop on Nvar
            
            // Determine if event is in modified box or sideband
            // -------------------------------------------------
            for (int k=0; k<Nvar; k++) {
                if (donedir[k]) continue;
                // To fill side_grad, we consider the sidebands corresponding to all k*m kth-var moves
                // and track the inclusion of position in the intervals, spanned by the third component of the matrix
                for (int m=0; m<7; m++) {
                    if (okmove[k][m]) {
                        for (int k2=0; k2<Nvar; k2++) {
                            double position = (0.5+order_ind[Ivar[k2]][i])/goodevents; // pos of ev in this var
                            if (position>Smin[k][m][k2] && position<=Smax[k][m][k2]) {
                                ns[k][m]++;
                            }
                        }
                    }
                }
            }

            // Increment counters for all considered boxes
            // -------------------------------------------
            if (in_==Nvar) {
                Nin++; // event is in signal box
            } else if (side_==Nvar) {
                Nside++; // event is in sideband but not in signal box
            }
            // Update # events in varied boxes
            // -------------------------------
            for (int k=0; k<Nvar; k++) {  
                if (donedir[k]) continue;
                if (in_==Nvar || (!in[k] && in_==Nvar-1)) { // either the event is in original box 
                    // or it fails only in the gradient dir
                    for (int m=0; m<7; m++) {
                        if (okmove[k][m]) {
                            if (in_grad[k][m]) Nin_grad[k][m]++;  // event is in modified box along var k
                        }
                    }
                }
            }
            
            // Compute Nside_grad
            // ------------------
            for (int k=0; k<Nvar; k++) {
                if (donedir[k]) continue;
                for (int m=0; m<7; m++) {
                    if (okmove[k][m]) {
                        if (ns[k][m]==Nvar) { 
                            // This event is in sideband of grad modified interval along k
                            Nside_grad[k][m]++;
                        } // if in k*m-th sideband
                    } // move is ok
                } // m loop
            } // k loop
        } // end of i loop on data. All boxes are filled

        for (int k=0; k<Nvar; k++) {
            if (donedir[k]) continue;
            for (int m=0; m<7; m++) {
                if (okmove[k][m]) {
                    Nside_grad[k][m] -= Nin_grad[k][m]; // remove box events
                    /* 	      if (Nin_grad[k][m]>10*(Nside_grad[k][m]+10)) { */
                    /* 		cout << Nin_grad[k][m] << " " << Nside_grad[k][m] << endl; */
                    /* 		for (int k2=0; k2<Nvar; k2++) { */
                    /* 		  if (k==k2) { */
                    /* 		    cout << "B[" << k2 << "] = [" << Bmin[k2][m] << "," << Bmax[k2][m] << "],"; */
                    /* 		  } else { */
                    /* 		    cout << "B[" << k2 << "] = [" << Blockmin[k2] << "," << Blockmax[k2] << "],"; */
                    /* 		  } */
                    /* 		  cout << ", S[" << k2 << "] = ["  */
                    /* 		       << Smin[k][m][k2] << "," << Smax[k][m][k2] << "]" << endl; */
                    /* 		} */
                    /* 	      } */
                }
            }
        }

        if (gd==0) {
            Initial_density = log((double)Nin/VolumeOrig/goodevents);
            InitialDensity->Fill(Initial_density);
            InitialVolume->Fill(VolumeOrig);
        }
        
        if (Nin<1) { // The box is too small, or the seeding went bad for some reason
            cout << "  Nin<1, volume = " << VolumeOrig << " donedir = " 
                << donedir[0] << donedir[1] << donedir[2] << donedir[3] 
                << donedir[4] << donedir[5] << " - rethrowing box" << endl;
            
            // Rethrow intervals and go back to start of loop
            // ----------------------------------------------
            //double Nvar_box = (log(10./goodevents)/log(1./3));
            //if (Nvar_box>Nvar) Nvar_box = Nvar;
            // do {
            VolumeOrig = 1.;
            int nonfull = (int)gRandom->Uniform(0.,Nvar-epsilon);
            for (int k=0; k<Nvar; k++) { 
                
                if (k==nonfull) {
                    Blockmin[k] = epsilon*(int)(InvEpsilon*gRandom->Uniform(0.,0.5));
                    Blockmax[k] = 0.5+Blockmin[k];
                    VolumeOrig *= (Blockmax[k]-Blockmin[k]);	     
                } else {
                    Blockmin[k] = 0.;
                    Blockmax[k] = 1.;
                }	    	    
                // Start fresh with search
                donedir[k] = false;
                jumps[k] = 0;
                // Reset lambdas too
                for (int m=0; m<6; m++) {
                    lambda[k][m] = InitialLambda; // 0.2;  // MUST CHECK IF THIS CAN BE BROUGHT TO 0.3 with same power for A0
                }
            } // end k	    
            determineSB(Sidemin,Sidemax,Blockmin,Blockmax,Nvar);
            VolumeOrig      = 1.;
            SidebandsVolume = 1.;
            for (int k=0; k<Nvar; k++) {
                VolumeOrig      *= fabs(Blockmax[k]-Blockmin[k]);
                SidebandsVolume *= fabs(Sidemax[k]-Sidemin[k]);
            }
            
            for (int k=0; k<Nvar; k++) {
                // Store initial values to compute Drift histogram of boundaries due to GD minimization
                BmiIn[k] = Blockmin[k];
                BmaIn[k] = Blockmax[k];  
            }
            gd = -1;
            continue;
        } // end if Nin=0
        

        // The first time it passes through, Nin is not zero anymore, due to the above
        if (gd==0) {
            Ninbox_in->Fill((double)Nin);
            Nin0 = Nin; // Initial number of events in signal box, for (scatter)plotting purposes
        }
        
        // Compute Z-value of hyp that Nin is compatible with box volume, in Poisson approx.
        // --------------------------------------------------------------------------------
        double Zval[7] = { 0.,0.,0.,0.,0.,0.,0.};
        double Zval_start;
        if (useZPL) {
            if (useSB && Nside>0 && VolumeOrig<0.5) { 
                Nexp       = Nside;
                Zval_start = ZPLtau(Nin,Nside,1.);
            } else { // When sidebands have no events, the signal box volume is used
                Nexp       = goodevents*VolumeOrig;
                Zval_start = ZPL(Nin,goodevents,VolumeOrig);
            }	  
        } else {
            if (useSB && Nside>0) {
                Nexp       = Nside;
                Zval_start = R2(Nin,Nexp); 
            } else { // When sidebands have no events, the signal box volume is used
                Nexp       = goodevents*VolumeOrig;
                Zval_start = R2(Nin,Nexp);
            }
        }
        // Starting values before box move
        if (gd==0) Zvalue_in->Fill(Zval_start);
        Zvalbest = Zval_start;
        Ninbest  = Nin;
        Nexpbest = Nexp;
        igrad    = -1;
        dirgrad  = 0;
        if (debug) cout << "Zstart= " << Zval_start << "Ni,e=" << Nin << " " << Nexp << endl;
        for (int k=0; k<Nvar; k++) {
            if (donedir[k]) continue;
            for (int m=0; m<7; m++) {
                if (okmove[k][m]) {
                    if (useZPL) {
                        if (useSB && Nside_grad[k][m]>0 && VolumeMod[k][m]<0.5) {
                        Nexp = Nside_grad[k][m];
                        Zval[m] = ZPLtau(Nin_grad[k][m],Nside_grad[k][m],1.);
                        } else {
                        Nexp = VolumeMod[k][m]*goodevents;
                        Zval[m] = ZPL(Nin_grad[k][m],goodevents,VolumeMod[k][m]);
                        }	  
                    } else {
                        if (useSB && Nside_grad[k][m]>0 && VolumeMod[k][m]<0.5) {
                            Nexp = Nside_grad[k][m];
                            Zval[m] = R2(Nin_grad[k][m],Nexp);
                        } else {
                            Nexp = VolumeMod[k][m]*goodevents;
                            Zval[m] = R2(Nin_grad[k][m],Nexp);
                        }
                    }
                    if (debug) cout << "Z[" << k << "," << m << "]=" << Zval[m] 
                            << " [" << Bmin[k][m] << "," << Bmax[k][m] << "]" 
                            << " N,S,E= " << Nin_grad[k][m] << "," << Nside_grad[k][m] << "," << Nexp 
                            << " Vm=" << VolumeMod[k][m] << endl;
                    
                    // Find the highest Z value among these
                    // ------------------------------------
                    if (Zval[m]>Zvalbest+epsilon) { // the move must improve and be the best
                        igrad     = k;
                        dirgrad   = m;
                        Zvalbest  = Zval[m];
                        Ninbest   = Nin_grad[igrad][dirgrad];
                        Nexpbest  = Nexp;
                        Nsidebest = Nside_grad[igrad][dirgrad];
                    } 
                }  // ok move
            } // end m
        } // end k

        // If the best move is a jump, update counter
        // ------------------------------------------
        if (dirgrad==6) jumps[igrad]++;
        if (jumps[igrad]>=maxJumps) okmove[igrad][6] = false;

        // Debug
        // -----
        if (debug) {
            int totjumps = 0;
            double logprodlambdas = 0.;
            for (int k=0; k<Nvar; k++) {
                for (int m=0; m<6; m++) {
                    logprodlambdas += log(lambda[k][m]);
                }
                totjumps += jumps[k];
            }
            cout << " k,g=" << igrad << " " << dirgrad;
            if (igrad>-1) { 
                cout << " [" << Bmin[igrad][dirgrad] << "," << Bmax[igrad][dirgrad] << "]";
            } else {
                cout << "prod lambda = " << logprodlambdas << " totjumps = " << totjumps;
            }
            if (Zvalbest>Zval_start) {
                cout << " Zbest=" << Zvalbest << "(was " << Zval_start <<"); Ni,e=" << Ninbest << " " 
                << Nexpbest << " vol= " << VolumeMod[igrad][dirgrad] << " ig,dg= " 
                << igrad << " " << dirgrad << " ";
            } 
            if (igrad>-1) {
                if (dirgrad<6) {
                cout << " l=" << lambda[igrad][dirgrad];
                } else {
                cout << " j=" << jumps[igrad];
                }
            }
            cout << endl;
        }	
        
        // Decision on this move is taken, update values
        // ---------------------------------------------
        if (igrad==-1) { // Are we in a minimum?
            for (int k=0; k<Nvar; k++) {
                if (donedir[k]) continue;
                for (int m=0; m<6; m++) {
                    // Attempt to explore other minima
                    // if (gd/2==0) {
                    //   lambdatmp[k][m] = lambda[k][m];
                    //   lambda[k][m] = epsilon*(int)(InvEpsilon*gRandom->Uniform(epsilon,0.2));
                    // } else {
                    // recall the previous time we were here, and decrease by epsilon all steps
                    // lambda[k][m] = epsilon*(int)(InvEpsilon*(lambdatmp[k][m]-epsilon)); 
                    
                    // lambda[k][m] = epsilon*(int)(InvEpsilon*(lambda[k][m]-epsilon));
                    lambda[k][m] = epsilon*(int)(InvEpsilon*(lambda[k][m]*shrinking_factor));
                    if (lambda[k][m]<epsilon) lambda[k][m] = epsilon;
                    // lambdatmp[k][m] = lambda[k][m];
                    // }
                }
                if (lambda[k][0]<=epsilon && lambda[k][1]<=epsilon && 
                lambda[k][2]<=epsilon && lambda[k][3]<=epsilon && 
                lambda[k][4]<=epsilon && lambda[k][5]<=epsilon) {
                    donedir[k] = true; // stop descent in this direction 
                }
            }	  
        } else { // we can improve the Z value by moving
            // Update box boundaries
            Blockmin[igrad] = Bmin[igrad][dirgrad];
            Blockmax[igrad] = Bmax[igrad][dirgrad];
            for (int k2=0; k2<Nvar; k2++) {
                Sidemin[k2]  = Smin[igrad][dirgrad][k2];
                Sidemax[k2]  = Smax[igrad][dirgrad][k2];
            }
            // Now update lambdas. If we have made a jump, we allow lambdas to not be too small
            // such that the neighborhood can be studied in the jump direction; otherwise we
            // shrink step directions not used
            if (dirgrad==6) {
                for (int m=0; m<6; m++) {
                if (lambda[igrad][m]<InitialLambda/4) lambda[igrad][m] = InitialLambda/4;
                }
            } else {
                for (int m=0; m<6; m++) { 
                    if (m!=dirgrad) {
                        // shrink steps in directions not used at this iteration
                        lambda[igrad][m] = epsilon*(int)(InvEpsilon*shrinking_factor*lambda[igrad][m]); 
                        if (lambda[igrad][m]<epsilon) lambda[igrad][m] = epsilon;
                        //lambdatmp[igrad][m] = lambda[igrad][m];
                    }
                }
            }
            // Accelerate step if proceeding in same direction!?
            if (widening_factor>1) {
                if (dirgrad==0 || dirgrad==4) {
                    if (Blockmin[igrad]>0) {
                        // accelerated gradient
                        lambda[igrad][dirgrad] = epsilon*(int)(InvEpsilon*(lambda[igrad][dirgrad]*widening_factor)); 
                        if (lambda[igrad][dirgrad]>Blockmin[igrad]) {
                            lambda[igrad][dirgrad] = Blockmin[igrad];
                        }
                    } 
                } else if (dirgrad==1 || dirgrad==2) {
                    if (Blockmax[igrad]-Blockmin[igrad]>epsilon) {
                        // accelerated gradient
                        lambda[igrad][dirgrad] = epsilon*(int)(InvEpsilon*(lambda[igrad][dirgrad]*widening_factor)); 
                        if (lambda[igrad][dirgrad]>Blockmax[igrad]-Blockmin[igrad]-epsilon) { 
                            lambda[igrad][dirgrad] = Blockmax[igrad]-Blockmin[igrad]-epsilon;
                        } 
                    } 
                } else if (dirgrad==3 || dirgrad==5) {
                    if (Blockmax[igrad]<1) {
                        // accelerated gradient
                        lambda[igrad][dirgrad] = epsilon*(int)(InvEpsilon*(lambda[igrad][dirgrad]*widening_factor)); 
                        if (lambda[igrad][dirgrad]>1.-Blockmax[igrad]) {
                            lambda[igrad][dirgrad] = 1.-Blockmax[igrad];
                        }
                    }
                }
                if (dirgrad<6) {
                    if (lambda[igrad][dirgrad]<epsilon) lambda[igrad][dirgrad] = epsilon;
                }
                // lambdatmp[igrad][dirgrad] = lambda[igrad][dirgrad];
            } // end if igrad =/!= -1
        } // end move
	
      } // end of gd loop /////////////////////////////////////////////////////////////////      

	  // Check for subspace result
	  // -------------------------
      double ThisVolume = 1.;
      for (int k=0; k<Nvar; k++) {
    	ThisVolume = ThisVolume*(Blockmax[k]-Blockmin[k]);
      }
      // Fill variables to keep track of results
      // ---------------------------------------
      BoxVol[trial] = ThisVolume;
      BoxNin[trial] = Ninbest;
      BoxNex[trial] = Nsidebest;
      BoxZpl[trial] = Zvalbest;
      BoxInd[trial] = trial;
      for (int k=0; k<Nvar; k++) {
	    BoxVar[trial][k] = Ivar[k];
      }
      int NSIB = 0;
      if (Nsignal>0) {
        for (int i=0; i<goodevents; i++) {
            if (isSignal[i]) {
                bool NotIn = false;	    
                for (int k=0; k<Nvar && !NotIn; k++) {
                    double position = (0.5+order_ind[Ivar[k]][i])/goodevents; // pos in this var
                    if (position<=Blockmin[k] || position>Blockmax[k]) { 
                        NotIn = true;
                    } 
                }
                if (!NotIn) NSIB++;
            }
        }
      }
      
      BoxSFr[trial] = 0.;
      if (Nsignal>0) BoxSFr[trial] = 100.*NSIB/Nsignal;
      
      if (debug) {
        cout << " D1[" << Blockmin[0] << "," << Blockmax[0] << "]" 
             << " D2[" << Blockmin[1] << "," << Blockmax[1] << "]" 
             << " No, Ne " << Ninbest << " " << Nsidebest;
        cout << " V= " << ThisVolume << " Z = " << Zvalbest << endl;
      }
      
      // Printout per trial
      // ------------------
      cout << "  Trial # " << trial << " Z= " << Zvalbest << " Nin/exp= " 
	   << Ninbest << "/" << Nexpbest << " Ns="<< NSIB;
      if (useSB) {
    	cout << " VB=" << ThisVolume << endl;
      } else {
    	cout << endl;
      }

      // If this is a better box than all others, adjourn vars
      // ----------------------------------------------------- 
      if (Zval_best<Zvalbest) {
        Zval_best  = Zvalbest;
        Nin_best   = Ninbest;
        Nexp_best  = Nexpbest;
        gd_best    = gd2;
        trial_best = trial;
        ID_best    = Initial_density;
        for (int k=0; k<Nvar; k++) {
            Ivar_best[k] = Ivar[k];
            // The boundaries are the best ones as they get updated every time igrad!=-1
            Blockmin_best[k] = Blockmin[k];
            Blockmax_best[k] = Blockmax[k];
        }
        if (debug) cout << "Updated values: " << Zval_best << " " << Nin_best 
                << " " << Nexp_best << endl;
        
        // Printout
        int NSignalInBox = 0;
        if (Nsignal>0) {
            for (int i=0; i<goodevents; i++) {
                if (isSignal[i]) {
                        bool NotIn = false;
                        for (int k=0; k<Nvar && !NotIn; k++) {
                            double position = (0.5+order_ind[Ivar_best[k]][i])/goodevents; // pos of ev in this var
                            if (position<=Blockmin_best[k] || position>Blockmax_best[k]) { 
                                NotIn = true;
                            }
                        }
                    if (!NotIn) NSignalInBox++;
                }
            }	  
            double sbgain = 0;
            if (Nin_best>0) 
                sbgain = ((double)NSignalInBox/(double)Nin_best) * goodevents/ (double)Nsignal; 
            cout << "  Z=" << Zval_best << " Nin, Nexp = " << Nin_best << ", " 
                 << Nexp_best;
            cout << " - SFR= " << (double)NSignalInBox/Nsignal*100. << " - " << "Ns_in = " << NSignalInBox; 
            cout << "; vars = ";
            for (int k=0; k<Nvar; k++) { cout << Ivar[k] << " "; }
            cout << " - SB gain = " << sbgain << endl << endl;;
            results << "  Trial " << trial << ": Z = " << Zval_best << "; Nin, Nexp = " 
                    << Nin_best << ", " << Nexp_best;
            results << " - Ns_in = " << NSignalInBox << "; SFR = " 
                    << (double)NSignalInBox/Nsignal*100. 
                    << "; vars= "; 
            for ( int k=0; k<Nvar; k++) { results << Ivar[k] << " ";}
            results << " - SB gain = " << sbgain << endl << endl;;
        }  
      } // Zval_best updated
      
      Ninbox_fi->Fill(Ninbest);
      Ninbox_in_vs_fi->Fill(Nin0,Ninbest);
      NGDsteps->Fill((double)gd2);
      
      if (Ninbest>0) {
        Zvalue_fi->Fill(Zvalbest);
      }
      // Fill some histograms
      // --------------------
      double log_final_volume = 0.;
      for (int k=0; k<Nvar; k++) {
        Bounds_fi->Fill(Blockmin[k],Blockmax[k]);
        Drift->Fill(Blockmin[k]-BmiIn[k],Blockmax[k]-BmaIn[k]);
      }
      
      // Compute order statistic based on which PC were chosen at this trial
      double order = 0;
      for (int dim=0; dim<Nvar; dim++) {
        order += exp(-Ivar[dim]);
      }
      ZvsOrder->Fill(order,Zvalbest);
      
    } // End loop on Ntrials 
    if (Ntrials>50) cout << progress[51] << endl << endl; // End of progress string
    
    //////////////////////////////////////////////////////////////////////////////////////
    // --------------------------------------------------------------------------
    // FINAL PRINTOUT
    // --------------
    // For toys and injected signal: 
    // determine how many signal events were contained in the best box
    // ---------------------------------------------------------------    
    // If we are testing accuracy of different seeding algorithms, we keep
    // track of average SF caught in box, and average number of 1-sigma Gauss marginals in boundaries    
    int NSignalInBox = 0;
    if (Nsignal>0) {
      for (int i=0; i<goodevents; i++) {
        if (isSignal[i]) {
            bool NotIn = false;
            for (int k=0; k<Nvar && !NotIn; k++) {
                double position = (0.5+order_ind[Ivar_best[k]][i])/goodevents; // pos of ev in this var
                if (position<=Blockmin_best[k] || position>Blockmax_best[k]) { 
                    NotIn = true;
                }
            }
            if (!NotIn) NSignalInBox++;
        }
      }
      
      Aver_SF_caught  += (double)NSignalInBox/Nsignal;
      Aver2_SF_caught += pow((double)NSignalInBox/Nsignal,2);
      
      // Count how many Gaussian means are included in the best interval
      // ---------------------------------------------------------------
      double ndimin = 0;
      for (int k=0; k<Nvar; k++) {
        if (Gaussian[Ivar_best[k]]) {
            if (mean[Ivar_best[k]]-0.5*sigma[Ivar_best[k]]>=Blockmin_best[k] && 
                mean[Ivar_best[k]]+0.5*sigma[Ivar_best[k]]<Blockmax_best[k]) {
                ndimin++;
            }
        }
      }
      if (NseedTrials>1) cout << "  This time AverSF=" <<  (double)NSignalInBox/Nsignal 
			      << "  Aver_1s= " << ndimin << endl;
      Aver_1s_contained  += ndimin;
      Aver2_1s_contained += pow(ndimin,2);
    }
    
    //} // End Test of Cluster, Ntestseed loop - turn on if needed
    
    if (mock && NseedTrials>1) { // Nseed tests only for toy data  
      Aver_SF_caught     = Aver_SF_caught/NseedTrials;
      Aver2_SF_caught    = Aver2_SF_caught/NseedTrials;
      Aver_1s_contained  = Aver_1s_contained/NseedTrials;
      Aver2_1s_contained = Aver2_1s_contained/NseedTrials;
      double sqm_Aver_SF_caught    = sqrt(Aver2_SF_caught-pow(Aver_SF_caught,2));
      double sqm_Aver_1s_contained = sqrt(Aver2_1s_contained-pow(Aver_1s_contained,2));
      cout << endl;
      cout << "  Average SF caught by box: " << Aver_SF_caught;
      if (NseedTrials>1) cout << "+-" 
			      << sqm_Aver_SF_caught/sqrt(NseedTrials-1);
      cout << endl;
      cout << "  Average 1sigma contained: " << Aver_1s_contained;
      if (NseedTrials>1) cout << "+-" 
			      << sqm_Aver_1s_contained/sqrt(NseedTrials-1); 
      cout << endl << endl;
    }
    
    NSignalInBox = 0;
    if (Nsignal>0) {
      //if ((mock && Gaussian_dims>0) || fakefrac>0) {
      for (int i=0; i<goodevents; i++) {
        if (isSignal[i]) {
            bool NotIn = false;
            for (int k=0; k<Nvar && !NotIn; k++) {
                double position = (0.5+order_ind[Ivar_best[k]][i])/goodevents; // pos of ev in this var
                if (position<=Blockmin_best[k] || position>Blockmax_best[k]) { 
                    NotIn = true;
                }
            }
            if (!NotIn) NSignalInBox++;
        }
      }
      cout << "  Caught " << (double)NSignalInBox/Nsignal*100. << " % of injected signal" << endl;
    }
    
    for (int k=0; k<Nvar; k++) {
        if (mock) {
            cout << "  Var " << Ivar_best[k] << ":" << varname_mock[Ivar_best[k]] << 
                 " Original bounds: [" << Blockmin_best[k] << "," << Blockmax_best[k] << "]" << endl;
        } else {
            cout << "  Var " << Ivar_best[k] << ":" << varname[Ivar_best[k]] << 
            " Original bounds: [" << Blockmin_best[k] << "," << Blockmax_best[k] << "]" << endl;
        }
    }
    
    //////////////////////////////////////////////////////////////////
    // Only for toys:  determine absolute optimal box boundaries
    // Here we assume that Gaussian_dims>Nvar, and compute the 
    // optimal conditions using expectation value of signal in box
    // with total volume (omitting calculation of signal in sidebands)
    // ---------------------------------------------------------------
    if (mock && Gaussian_dims>0) {
      double ProdSigma = 1.;
      for (int k=0; k<Nvar; k++) {
        if (Gaussian[Ivar_best[k]]) {
            ProdSigma *= sigma[Ivar_best[k]];
        }
      }
      double maxZ      = -bignumber;
      double x_maxZ    = 0.;
      double Nexp_maxZ = 0.;
      double Nobs_maxZ = 0.;
      double Vol_maxZ  = 0.;
      for (int ix=0; ix<3000; ix++) {
        double x = (double)ix/1000.; // number of sigmas extension of half-intervals
        double boxvolume = pow(2*x,Gaussian_dims)*ProdSigma;
        double Nobs = Nsignal*pow(TMath::Erf(x/sqrt(2.)),Gaussian_dims) + Nbackground*boxvolume;
        double Nexp = goodevents*boxvolume; // Exp bgr is calculated with all events
        double Z;
        if (useZPL) { // useSB is implied false here
            Z = ZPL(Nobs,goodevents,boxvolume); 
        } else {
            Z = R2(Nobs,Nexp);
        }
        if (Z>maxZ) {
            maxZ      = Z;
            Nexp_maxZ = Nexp;
            Nobs_maxZ = Nobs;
            Vol_maxZ  = boxvolume;
            x_maxZ    = x;
        }
      }
      cout << endl;
      cout << "  Optimal Nsig  = " 
	   << Nsignal * pow(TMath::Erf(x_maxZ/sqrt(2)),Gaussian_dims) << " Volume = " << 0.00001*((int)(100000*Vol_maxZ))
	   << " Nobs = " << Nobs_maxZ << " Nexp = " << Nexp_maxZ << "; x = " << x_maxZ << " Z = " << maxZ << endl;
      results << endl;
      results << "  Optimal Nsig  = " 
	      << Nsignal * pow(TMath::Erf(x_maxZ),Gaussian_dims) << " Volume = " << 0.00001*((int)(100000*Vol_maxZ))
	      << " Nobs = " << Nobs_maxZ << " Nexp = " << Nexp_maxZ << "; x = " << x_maxZ << " Z = " << maxZ << endl;
    }
  
    cout << endl;
    if (NH0>1) cout << "  H0 test # " << IH0 << endl;
    cout << "  Best Z after " << gd_best << " loops = " << 0.1*(int)(10*Zval_best) 
	 << " Nin = " << Nin_best << " Nexp = " << 0.01*(int)(100*Nexp_best) 
	 << " Initial Dens. = " << 0.01*(int)(100*ID_best) << endl;  
    cout << endl;
    results << endl;
    if (NH0>1) results << "  H0 test # " << IH0 << endl;  
    results << "  Best Z after " << gd_best << " loops = " << 0.1*(int)(10*Zval_best) 
	    << " Nin = " << Nin_best << " Nexp = " << 0.01*(int)(100*Nexp_best) 
      //    << " SF = " << 
	    << " Initial Dens. = " << 0.01*(int)(100*ID_best);  
    results << endl;
    
    // Summary printout
    // ----------------
    if (IH0==0) summary << "id= " << id << " NAD=" << NAD << " NSEL=" << NSEL << " PCA=" << PCA << " RF=" << RegFactor;
    if (IH0==0 && mock) summary << " FixG=" << fixed_gaussians << " NarrG=" 
				<< narrow_gaussians << " forceG=" << force_gaussians << " maxR=" << maxRho 
				<< " maxHMR=" << maxHalfMuRange;
    if (IH0==0) summary << endl;
    if (IH0==0) summary << "id  Ntr  NS  NB  Alg maxZ useSB Nvar spup Z  Nin Nex Ns Vol BoxSF SBg BoxVars" << endl;
    if (IH0==0) summary << "---------------------------------------------------------------------------------------" << endl;
    summary << id << " " << Ntrials << "  " << Nsignal << " " << Nbackground << "  " << Algorithm << "  " 
	    << useZPL << "   " << useSB << "   " << Nvar << "   " << speedup << "  ";
    summary << 0.01*(int)(100*BoxZpl[trial_best]) 
	    << "  " << Nin_best  
	    << "  " << 0.01*(int)(Nexp_best*100)
	    << "  " << 0.1*(int)(10*BoxSFr[trial_best]*Nsignal/100.)
	    << "  " << 0.00001*(int)(100000*BoxVol[trial_best])
	    << "  " << 0.1*(int)(10*BoxSFr[trial_best]) << "%"
	    << "  " << 0.01*BoxSFr[trial_best]/Nin_best*goodevents;
    for (int ivar=0; ivar<Nvar; ivar++) {
      summary << " " << BoxVar[trial_best][ivar];
    }
    summary << endl;
    
    // Printout of best boxes, if Signal>0
    // -----------------------------------
    if (Nsignal>0) {
      for (int times=0; times<Ntrials; times++) {
        for (int i=Ntrials-1; i>0; i--) {
            if (BoxZpl[BoxInd[i]]>BoxZpl[BoxInd[i-1]]) {
                int tmp = BoxInd[i];
                BoxInd[i] = BoxInd[i-1];
                BoxInd[i-1] = tmp;
            }
        }
      }
      for (int i=0; i<20 && i<Ntrials; i++) {
        int ind = BoxInd[i];
        double sbgain = 0;
        sbgain = 0.01*BoxSFr[ind]/BoxNin[ind] * goodevents; 
        cout << "  Z=" << 0.01*(int)(100*BoxZpl[ind]) 
            << "  Nin=" << BoxNin[ind]
            << "  Nex=" << 0.01*(int)(BoxNex[ind]*100)
            << "  Ns=" << 0.1*(int)(10*BoxSFr[ind]*Nsignal/100.)
            << "  V=" << 0.0001*(int)(10000*BoxVol[ind])
            << "  SF=" << 0.1*(int)(10*BoxSFr[ind])
            << "  SBg= " << sbgain << "  Vars= ";
        for (int ivar=0; ivar<Nvar; ivar++) {
            cout << " " << BoxVar[ind][ivar];
        }
        cout << endl;
        results << "  Z=" << 0.01*(int)(100*BoxZpl[ind]) 
            << "  Nin=" << BoxNin[ind]
            << "  Nex=" << 0.01*(int)(BoxNex[ind]*100)
            << "  Ns=" << 0.1*(int)(10*BoxSFr[ind]*Nsignal/100.)
            << "  V=" << 0.0001*(int)(10000*BoxVol[ind])
            << "  SF=" << 0.1*(int)(10*BoxSFr[ind])
            << "  SBg= " << sbgain << "  Vars= ";
        for (int ivar=0; ivar<Nvar; ivar++) {
            results << " " << BoxVar[ind][ivar];
        }
        results << endl;
      }
    }
    summary << endl << endl;
    
    if (NH0>1) {
      zpl << Zval_best << "\t\t" << NSignalInBox << endl;
    }
    
    // End loop on IH0, if testing NH0 datasets for test statistic distributions under the null
    // ----------------------------------------------------------------------------------------
    ZH0->Fill(BoxZpl[trial_best]);
    // if (PCA && NH0>1) 	~principal();   
  } // end IH0 loop
  
  delete [] PB_all;
  delete [] PointedBy;
  delete [] BoxVar;
  
  ////////////////////////////////////////////////////////////////////////////////////////////
  // Construct plots of marginals for the considered features in the best box
  // ------------------------------------------------------------------------
  Double_t * dataP = new Double_t[ND];
  Double_t * dataX = new Double_t[ND];  
  if (plots) {

    int Nbins1D = 50;
    if (goodevents<4000) Nbins1D = 25;
    char nameplotal[30];
    char nameplotin[30];
    char nameplotex[30];
    char nameplotsi[30];
    double sidew = 0.5/Nbins1D;
    // original vars live in NAD space (box selection after PCA backtransform spans full space)
    for (int k=0; k<NAD; k++) {
      sprintf (nameplotal, "OPlot_al%d", k);
      sprintf (nameplotin, "OPlot_in%d", k);
      sprintf (nameplotsi, "OPlot_si%d", k);
      double xmin = OXmin[k]; // (1.+sidew)*OXmin[k]-sidew*OXmax[k]; 
      double xmax = OXmax[k]; // (1.+sidew)*OXmax[k]-sidew*OXmin[k];
      OPlot_al[k] = new TH1D (nameplotal, nameplotal, Nbins1D, xmin, xmax); 
      OPlot_in[k] = new TH1D (nameplotin, nameplotin, Nbins1D, xmin, xmax);
      OPlot_si[k] = new TH1D (nameplotsi, nameplotsi, Nbins1D, xmin, xmax);
      OPlot_in[k]->SetFillColor(kGreen);
      OPlot_in[k]->SetLineColor(kGreen);
      OPlot_al[k]->SetLineColor(kBlue);
      OPlot_si[k]->SetLineColor(kBlack);
      OPlot_si[k]->SetLineWidth(2);
      OPlot_al[k]->SetLineWidth(2);
      OPlot_in[k]->SetLineWidth(2);

      sprintf (nameplotal, "Plot_al%d", k);
      sprintf (nameplotin, "Plot_in%d", k);
      sprintf (nameplotex, "Plot_ex%d", k);
      sprintf (nameplotsi, "Plot_si%d", k);
      xmin = Xmin[Ivar_best[k]]; // (1.+sidew)*Xmin[Ivar_best[k]]-sidew*Xmax[Ivar_best[k]];
      xmax = Xmax[Ivar_best[k]]; // (1.+sidew)*Xmax[Ivar_best[k]]-sidew*Xmin[Ivar_best[k]];
      Plot_al[k] = new TH1D (nameplotal, nameplotal, Nbins1D, xmin, xmax); 
      Plot_in[k] = new TH1D (nameplotin, nameplotin, Nbins1D, xmin, xmax);
      Plot_ex[k] = new TH1D (nameplotex, nameplotex, Nbins1D, xmin, xmax);
      Plot_si[k] = new TH1D (nameplotsi, nameplotsi, Nbins1D, xmin, xmax);
      Plot_in[k]->SetFillColor(kGreen);
      Plot_in[k]->SetLineColor(kGreen);
      Plot_al[k]->SetLineColor(kBlue);
      Plot_ex[k]->SetFillColor(kRed);
      Plot_ex[k]->SetLineColor(kRed);
      Plot_si[k]->SetLineColor(kBlack);
      Plot_si[k]->SetLineWidth(2);
      Plot_in[k]->SetLineWidth(2);
      Plot_ex[k]->SetLineWidth(2);
      Plot_al[k]->SetLineWidth(2);

      sprintf (nameplotal, "UPlot_al%d", k);
      sprintf (nameplotin, "UPlot_in%d", k);
      sprintf (nameplotex, "UPlot_ex%d", k);
      sprintf (nameplotsi, "UPlot_si%d", k);
      UPlot_al[k] = new TH1D (nameplotal, nameplotal, Nbins1D, 0., 1.); // -sidew, 1.+sidew);
      UPlot_in[k] = new TH1D (nameplotin, nameplotin, Nbins1D, 0., 1.); // -sidew, 1.+sidew);
      UPlot_ex[k] = new TH1D (nameplotex, nameplotex, Nbins1D, 0., 1.); // -sidew, 1.+sidew);
      UPlot_si[k] = new TH1D (nameplotsi, nameplotsi, Nbins1D, 0., 1.); // -sidew, 1.+sidew);
      UPlot_in[k]->SetFillColor(kGreen);
      UPlot_al[k]->SetLineColor(kBlue);
      UPlot_ex[k]->SetFillColor(kRed);
      UPlot_in[k]->SetLineColor(kGreen);
      UPlot_ex[k]->SetLineColor(kRed);
      UPlot_si[k]->SetLineColor(kBlack);
      UPlot_si[k]->SetLineWidth(2);
      UPlot_in[k]->SetLineWidth(2);
      UPlot_ex[k]->SetLineWidth(2);
      UPlot_al[k]->SetLineWidth(2);
    }

    // Fill 1D plots
    // -------------
    double pos;
    int Nin1  = 0;
    int NUin1 = 0;
    
    for (int i=0; i<goodevents; i++) {      
      // If we did PCA, we want also the untransformed vars
      // --------------------------------------------------
      for (int dim=0; dim<NAD; dim++) {
        dataP[dim] = feature[dim][i];
        if (!PCA) dataX[dim] = dataP[dim];
      }
      if (PCA) P2X(dataP,dataX,NAD); 
      // Can now use dataX as original feature
      int Uin1_ = 0;
      for (int kk=0; kk<Nvar; kk++) {
        pos = (0.5+order_ind[Ivar_best[kk]][i])/goodevents;
        if (pos>Blockmin_best[kk] && pos<=Blockmax_best[kk]) {
        Uin1_++;
        }
      }
      // Original vars in NAD space
      for (int k=0; k<NAD; k++) {
        OPlot_al[k]->Fill(dataX[k]);
        if (Uin1_==Nvar)  {
            OPlot_in[k]->Fill(dataX[k]);
            if (isSignal[i]) OPlot_si[k]->Fill(dataX[k]);
        }
      }

      // PCA and UNIF space use Nvar
      for (int k=0; k<Nvar; k++) {
        Plot_al[k]->Fill(dataP[Ivar_best[k]]);
        pos = (0.5+order_ind[Ivar_best[k]][i])/goodevents;
        UPlot_al[k]->Fill(pos);
        // Fill plots for n cuts
        if (Uin1_==Nvar)  {
            Plot_in[k]->Fill(dataP[Ivar_best[k]]);
            if (isSignal[i]) Plot_si[k]->Fill(dataP[Ivar_best[k]]);
        }
        if (Uin1_==Nvar) {
            UPlot_in[k]->Fill(pos);
            if (isSignal[i]) UPlot_si[k]->Fill(pos);
        }
        // Now fill plots of n-1 cuts
        int Uin2_ = 0;
        for (int kk=0; kk<Nvar; kk++) {
            if (kk!=k) {
                pos = (0.5+order_ind[Ivar_best[kk]][i])/goodevents;
                if (pos>Blockmin_best[kk] && pos<=Blockmax_best[kk]) Uin2_++;
            }
        }
        if (Uin2_==Nvar-1) {
            Plot_ex[k]->Fill(dataP[Ivar_best[k]]);
        }
        pos = (0.5+order_ind[Ivar_best[k]][i])/goodevents;
        if (Uin2_==Nvar-1) UPlot_ex[k]->Fill(pos);
      } // end k
      if (Uin1_==Nvar) NUin1++;
    }

    // Scatterplots now
    // ----------------
    char nameplot2al[30];
    char nameplot2in[30];
    char nameplot2ex[30];
    int ind = 0;
    int Nbins2D = 20;
    sidew = 0.5/Nbins2D;
    for (int k=0; k<Nvar-1; k++) {
      for (int kk=k+1; kk<Nvar; kk++) {
        sprintf(nameplot2al,"SCP_al%d",ind);
        sprintf(nameplot2in,"SCP_in%d",ind);
        sprintf(nameplot2ex,"SCP_ex%d",ind);
        SCP_al[ind] = new TH2D(nameplot2al, nameplot2al, Nbins2D, 
                    (1.+sidew)*Xmin[Ivar_best[k]]-sidew*Xmax[Ivar_best[k]], 
                    (1.+sidew)*Xmax[Ivar_best[k]]-sidew*Xmin[Ivar_best[k]], 
                    Nbins2D, (1.+sidew)*Xmin[Ivar_best[kk]]-sidew*Xmax[Ivar_best[kk]], 
                    (1.+sidew)*Xmax[Ivar_best[kk]]-sidew*Xmin[Ivar_best[kk]]);
        SCP_in[ind] = new TH2D(nameplot2in, nameplot2in, Nbins2D, 
                    (1.+sidew)*Xmin[Ivar_best[k]]-sidew*Xmax[Ivar_best[k]], 
                    (1.+sidew)*Xmax[Ivar_best[k]]-sidew*Xmin[Ivar_best[k]], 
                    Nbins2D, (1.+sidew)*Xmin[Ivar_best[kk]]-sidew*Xmax[Ivar_best[kk]], 
                    (1.+sidew)*Xmax[Ivar_best[kk]]-sidew*Xmin[Ivar_best[kk]]);
        SCP_ex[ind] = new TH2D(nameplot2ex, nameplot2ex, Nbins2D, 
                    (1.+sidew)*Xmin[Ivar_best[k]]-sidew*Xmax[Ivar_best[k]], 
                    (1.+sidew)*Xmax[Ivar_best[k]]-sidew*Xmin[Ivar_best[k]], 
                    Nbins2D, (1.+sidew)*Xmin[Ivar_best[kk]]-sidew*Xmax[Ivar_best[kk]], 
                    (1.+sidew)*Xmax[Ivar_best[kk]]-sidew*Xmin[Ivar_best[kk]]);
        SCP_in[ind]->SetLineColor(kGreen);
        SCP_ex[ind]->SetLineColor(kRed);	
        
        sprintf(nameplot2al,"USCP_al%d",ind);
        sprintf(nameplot2in,"USCP_in%d",ind);
        sprintf(nameplot2ex,"USCP_ex%d",ind);
        USCP_al[ind] = new TH2D (nameplot2al, nameplot2al, Nbins2D, -sidew, 1.+sidew, 
                    Nbins2D, -sidew, 1.+sidew);
        USCP_in[ind] = new TH2D (nameplot2in, nameplot2in, Nbins2D, -sidew, 1.+sidew, 
                    Nbins2D, -sidew, 1.+sidew);
        USCP_ex[ind] = new TH2D (nameplot2ex, nameplot2ex, Nbins2D, -sidew, 1.+sidew, 
                    Nbins2D, -sidew, 1.+sidew);
        USCP_in[ind]->SetLineColor(kGreen);
        USCP_ex[ind]->SetLineColor(kRed);

        sprintf(nameplot2al,"OSCP_al%d",ind);
        sprintf(nameplot2in,"OSCP_in%d",ind);
        OSCP_al[ind] = new TH2D(nameplot2al, nameplot2al, Nbins2D, 
                    (1.+sidew)*OXmin[Ivar_best[k]]-sidew*OXmax[Ivar_best[k]], 
                    (1.+sidew)*OXmax[Ivar_best[k]]-sidew*OXmin[Ivar_best[k]], 
                    Nbins2D, (1.+sidew)*OXmin[Ivar_best[kk]]-sidew*OXmax[Ivar_best[kk]], 
                    (1.+sidew)*OXmax[Ivar_best[kk]]-sidew*OXmin[Ivar_best[kk]]);
        OSCP_in[ind] = new TH2D(nameplot2in, nameplot2in, Nbins2D, 
                    (1.+sidew)*OXmin[Ivar_best[k]]-sidew*OXmax[Ivar_best[k]], 
                    (1.+sidew)*OXmax[Ivar_best[k]]-sidew*OXmin[Ivar_best[k]], 
                    Nbins2D, (1.+sidew)*OXmin[Ivar_best[kk]]-sidew*OXmax[Ivar_best[kk]], 
                    (1.+sidew)*OXmax[Ivar_best[kk]]-sidew*OXmin[Ivar_best[kk]]);
        OSCP_in[ind]->SetLineColor(kGreen);

        ind++;
      }
    }

    // Fill scatterplots
    // -----------------
    for (int i=0; i<goodevents; i++) {
      // If we did PCA, we want also the untransformed vars
      // --------------------------------------------------
      for (int dim=0; dim<NAD; dim++) {
        dataP[dim] = feature[dim][i];
        if (!PCA) dataX[dim] = dataP[dim];
      }
      if (PCA) P2X(dataP,dataX,NAD); 
      // Can now use dataX as original feature
      
      // Untransformed var scatterplots - NB variables are scrambled by PCA if used
      // --------------------------------------------------------------------------
      int Uin_=0;
      for (int kkk=0; kkk<Nvar; kkk++) {
        pos = (0.5+order_ind[Ivar_best[kkk]][i])/goodevents;
        if (pos>Blockmin_best[kkk] && pos<=Blockmax_best[kkk]) {
            Uin_++;
        }
      }
      ind = 0;
      for (int k=0; k<Nvar-1; k++) {
        for (int kk=k+1; kk<Nvar; kk++) {
        Uin_ = 0;
        for (int kkk=0; kkk<Nvar; kkk++) {
            pos = (0.5+order_ind[Ivar_best[kkk]][i])/goodevents;
            if (pos>Blockmin_best[kkk] && pos<=Blockmax_best[kkk]) {
                Uin_++;
            }
        }
        double posx = (0.5+order_ind[Ivar_best[k]][i])/goodevents;
        double posy = (0.5+order_ind[Ivar_best[kk]][i])/goodevents;	
        SCP_al[ind]->Fill(feature[Ivar_best[k]][i],feature[Ivar_best[kk]][i]);
        USCP_al[ind]->Fill(posx, posy);
        OSCP_al[ind]->Fill(dataX[Ivar_best[k]],dataX[Ivar_best[kk]]);
        if (Uin_==Nvar) {
            SCP_in[ind]->Fill(feature[Ivar_best[k]][i],
                    feature[Ivar_best[kk]][i]);
            USCP_in[ind]->Fill(posx,posy);
            OSCP_in[ind]->Fill(dataX[Ivar_best[k]],dataX[Ivar_best[kk]]);
        }

        // Fill plots for n-2 cuts
        // -----------------------
        Uin_ = 0;
        for (int kkk=0; kkk<Nvar; kkk++) {
            if (kkk!=k && kkk!=kk) {
                pos = (0.5+order_ind[Ivar_best[kkk]][i])/goodevents;
                if (pos>Blockmin_best[kkk] && pos<=Blockmax_best[kkk]) {
                    Uin_++;
                }
            }
        }
        if (Uin_==Nvar-2) {
            SCP_ex[ind]->Fill(feature[Ivar_best[k]][i],
                    feature[Ivar_best[kk]][i]);	  
            USCP_ex[ind]->Fill(posx,posy);
        }

        ind++; // scatterplot index
        }
      }
    }
    
    // Draw all the stuff now. First, free up some memory
    // --------------------------------------------------
    delete [] feature;
    delete [] feature_all;
    delete [] order_ind;    
    delete [] order_ind_all;
    
    // Plot chosen features for all events and chosen events - 1D plots 
    // ----------------------------------------------------------------
    int NADplot = NAD;
    if (NADplot>30) NADplot = 30; // can't have too many

    PP = new TCanvas ("PP", "PP", 1000, 700);
    if (Nvar<7) {
      PP->Divide(3,2);
    } else if (Nvar<9) {
      PP->Divide(4,2);
    } else if (Nvar==9) {
      PP->Divide(3,3);
    } else if (Nvar<11) {
      PP->Divide(5,2);
    } else if (Nvar<13) {
      PP->Divide(4,3);
    } else if (Nvar<16) {
      PP->Divide(5,3);
    } else if (Nvar==16) {
      PP->Divide(4,4);
    } else {
      PP->Divide(5,4);
    }
    OP = new TCanvas ("OP", "OP", 1000, 700);
    if (NADplot<16) {
      OP->Divide(5,3);
    } else if (NADplot==16) {
      OP->Divide(4,4);
    } else if (NADplot<21) {
      OP->Divide(5,4);
    } else {
      OP->Divide(6,5);
    }
    UP = new TCanvas ("UP", "", 1000, 700);
    if (Nvar<7) {
      UP->Divide(3,2);
    } else if (Nvar<9) {
      UP->Divide(4,2);
    } else if (Nvar==9) {
      UP->Divide(3,3);
    } else if (Nvar<11) {
      UP->Divide(5,2);
    } else if (Nvar<13) {
      UP->Divide(4,3);
    } else if (Nvar<16) {
      UP->Divide(5,3);
    } else if (Nvar==16) {
      UP->Divide(4,4);
    } else {
      UP->Divide(5,4);
    }
    P2 = new TCanvas ("P2", "", 1000, 700);
    OP2 = new TCanvas ("OP2", "", 1000, 700);
    UP2 = new TCanvas ("UP2", "", 1000, 700);
        
    for (int k=0; k<Nvar; k++) {
      // Get histogram boundary
      double m = Plot_ex[k]->GetMaximum();
      double m2= 0;
      if (Plot_al[k]->Integral()>0 && Plot_in[k]->Integral()>0) 
        Plot_al[k]->Scale(Plot_in[k]->Integral()/Plot_al[k]->Integral());
      m2 = Plot_al[k]->GetMaximum();      
      if (m<m2) m = m2;
      Plot_ex[k]->SetMaximum(1.1*m);
      // Plot them
      PP->cd(k+1);
      Plot_ex[k]->Draw();
      Plot_in[k]->Draw("SAME");
      Plot_si[k]->Draw("SAME");
      Plot_al[k]->Draw("SAME");
    }
    
    for (int k=0; k<NADplot; k++) {
      // Get histogram boundary
      double m = OPlot_in[k]->GetMaximum();
      double m2= 0;
      if (OPlot_al[k]->Integral()>0 && OPlot_in[k]->Integral()>0) 
    	OPlot_al[k]->Scale(OPlot_in[k]->Integral()/OPlot_al[k]->Integral());
      m2 = OPlot_al[k]->GetMaximum();
      if (m<m2) m = m2;
      OPlot_in[k]->SetMaximum(1.1*m);
      // Plot them
      OP->cd(k+1);
      OPlot_in[k]->Draw();
      OPlot_si[k]->Draw("SAME");
      OPlot_al[k]->Draw("SAME");
    }
    
    for (int k=0; k<Nvar; k++) {
      // Get histogram boundary
      double m = UPlot_ex[k]->GetMaximum();
      double m2= 0;
      if (UPlot_al[k]->Integral()>0 && UPlot_in[k]->Integral()>0) 
    	UPlot_al[k]->Scale(UPlot_in[k]->Integral()/UPlot_al[k]->Integral());
      m2 = UPlot_al[k]->GetMaximum();
      if (m<m2) m = m2;
      UPlot_ex[k]->SetMaximum(1.1*m);
      // Plot them
      UP->cd(k+1);
      UPlot_ex[k]->Draw();
      UPlot_in[k]->Draw("SAME");
      UPlot_si[k]->Draw("SAME");
      UPlot_al[k]->Draw("SAME");
    }
    
    // Plot chosen features for all events and chosen events - 2D plots 
    // ----------------------------------------------------------------
    if (ind>98) ind=98; // max number of plots
    if (ind<=15) {
      P2->Divide(6,5);
    } else if (ind<=30) {
      P2->Divide(10,6);
    } else if (ind<=50) {
      P2->Divide(10,10);
    } else {
      P2->Divide(14,15); // max is 15x14 variables to plot       
    } 
    for (int i=0; i<ind; i++) {
      P2->cd(2*i+1);
      SCP_al[i]->Draw("BOX");
      P2->cd(2*i+2);
      SCP_ex[i]->Draw("BOX");
      SCP_in[i]->Draw("BOXSAME");
    }
    
    if (ind<=15) {
      OP2->Divide(6,5);
    } else if (ind<=30) {
      OP2->Divide(10,6);
    } else if (ind<=50) {
      OP2->Divide(10,10);
    } else {
      OP2->Divide(14,15); // max is 15x14 variables to plot
    } 
    for (int i=0; i<ind; i++) {
      OP2->cd(2*i+1);
      OSCP_al[i]->Draw("BOX");
      OP2->cd(2*i+2);
      OSCP_in[i]->Draw("BOXSAME");
    }
    
    if (ind<=15) {
      UP2->Divide(6,5);
    } else if (ind<=30) {
      UP2->Divide(10,6);
    } else if (ind<=50) {
      UP2->Divide(10,10);
    } else {
      UP2->Divide(14,15);
    } 
    for (int i=0; i<ind; i++) {
      UP2->cd(2*i+1);
      USCP_al[i]->Draw("BOX");
      UP2->cd(2*i+2);
      USCP_ex[i]->Draw("BOX");
      USCP_in[i]->Draw("BOXSAME");
    }  
    
    TCanvas * Vars = new TCanvas ("Vars","", 500,500);
    Vars->Divide(3,4);
    Vars->cd(1);
    Zvalue_in->Draw();
    Vars->cd(2);
    Zvalue_fi->Draw();
    Vars->cd(3);
    InitialDensity->Draw();
    Vars->cd(4);
    NClomax->Draw();
    Vars->cd(5);
    Bounds_in->Draw();
    Vars->cd(6);
    Bounds_fi->Draw();
    Vars->cd(7);
    Drift->Draw();
    Vars->cd(8);
    NGDsteps->Draw();
    Vars->cd(9);
    Ninbox_in->Draw();
    Ninbox_fi->SetLineColor(kRed);
    Ninbox_fi->Draw("SAME");
    Vars->cd(10);
    Ninbox_in_vs_fi->Draw();
    Vars->cd(11);
    InitialDensity->Draw();
    Vars->cd(12);
    InitialVolume->Draw();
    
    // Write histograms to root file
    // -----------------------------
    TFile * out = new TFile(rootfile.c_str(),"RECREATE");   
    out->cd();
    Zvalue_in->Write();
    Zvalue_fi->Write();
    Bounds_in->Write();
    Bounds_fi->Write();
    Drift->Write();
    NGDsteps->Write();
    Ninbox_in->Write();
    Ninbox_fi->Write();
    Ninbox_in_vs_fi->Write();
    InitialDensity->Write();
    InitialVolume->Write();
    NClomax->Write();
    ZH0->Write();
    ZvsOrder->Write();
    //Z_maxR->Write();
    //maxR_vs_Z->Write();
    
    for (int k=0;k<NAD; k++) {
      Plot_al[k]->Write();
      Plot_in[k]->Write();
      Plot_ex[k]->Write();
      Plot_si[k]->Write();
      UPlot_al[k]->Write();
      UPlot_in[k]->Write();
      UPlot_ex[k]->Write();
      UPlot_si[k]->Write();
      OPlot_al[k]->Write();
      OPlot_in[k]->Write();
      OPlot_si[k]->Write();
    }
    for (int k=0; k<Nvar*(Nvar-1)/2; k++) {
      SCP_al[k]->Write();
      SCP_in[k]->Write();
      SCP_ex[k]->Write();
      USCP_al[k]->Write();
      USCP_in[k]->Write();
      USCP_ex[k]->Write();
      OSCP_al[k]->Write();
      OSCP_in[k]->Write();
    }  
    // Canvases now
    PP->Write();
    OP->Write();
    UP->Write();
    P2->Write();
    OP2->Write();
    UP2->Write();
    Vars->Write();

    // Close root file
    // ---------------
    out->Close();
    
  } // end if plots 
  
  // Plot test statistic if requested
  // --------------------------------
  if (NH0>1) {
    TCanvas * H0 = new TCanvas ("H0", "", 500, 300);
    H0->cd();
    ZH0->Draw();
    
    // Determine quantiles of TS
    // -------------------------
    int    s        = 0;
    double alpha    = 0.05;
    double centrmin = 0.1586;
    double centrmax = 0.8414;
    double centr    = 0.5;
    double integral = 0.;
    double x        = 0.;
    double xcmin    = 0.;
    double xcmax    = 0.;
    double xc       = 0.;
    double xalpha   = 0.;
    for (int ix=0; ix<1000; ix++) {
      s += ZH0->GetBinContent(ix+1);
      x = 0.1*ix;
      if (s>=NH0*centrmin && xcmin==0.)   xcmin  = x;
      if (s>=NH0*centr && xc==0.)         xc     = x;
      if (s>=NH0*centrmax && xcmax==0.)   xcmax  = x;
      if (s>=NH0*(1-alpha) && xalpha==0.) xalpha = x;
    }
    cout << endl;
    cout << "  Central interval of Z distribution: [" << xcmin << " " << xc << " " << xcmax << "]" << endl;
    cout << "  Critical region: Z> " << xalpha << endl;
    cout << "  Mean Z = " << ZH0->GetMean() << " +- " << ZH0->GetRMS()/sqrt(NH0-1) << endl;
    cout << endl;
    results << endl;
    results << "  id = " << id;
    results << "  Central interval of Z distribution: [" << xcmin << " " << xc << " " << xcmax << "]" << endl;
    results << "  Critical region: Z> " << xalpha << endl;
    results << "  Mean Z = " << ZH0->GetMean() << " +- " << ZH0->GetRMS()/sqrt(NH0-1) << endl;
    results << endl;
    summary << endl;
    summary << "  id = " << id;
    summary << "  Central interval of Z distribution: [" << xcmin << " " << xc << " " << xcmax << "]" << endl;
    summary << "  Critical region: Z> " << xalpha << endl;
    summary << "  Mean Z = " << ZH0->GetMean() << " +- " << ZH0->GetRMS()/sqrt(NH0-1) << endl;
    summary << endl;
  }
  
  results.close();
  summary.close();
  
  gROOT->Time();
  return 0;
  
} // END 
