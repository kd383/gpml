// y = quadraticform(A, x)
#include "mex.h"

/*
Mx_in: Vector of length 4n
row_in: Vector of length 4n
ku_in: Vector of length m

Mx_in are the non-zero elements of the n-by-m interpolation matrix. Each row
has 4 non-zeros and these values are contigous chunks.
*/

/* Input Arguments */
#define Mx_in prhs[0]
#define inds_in prhs[1]
#define ku_in prhs[2]

/* Output Arguments */
#define diag plhs[0]

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mwSize mMx, nMx, n, mInds, nInds, mKu, nKu, m;
    double *Mx, *Ku;
    int *inds;

    if (nrhs != 3) {
        mexErrMsgTxt("Three input arguments required.");
    }
    else if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments.");
    }

    // Check Mx input
    mMx = mxGetM(Mx_in);
    nMx = mxGetN(Mx_in);
    if (mMx % 4 != 0 || nMx != 1) {
        mexErrMsgTxt("Mx must have size 4n x 1");
    }
    n = mMx/4;

    // Check inds input
    mInds = mxGetM(inds_in);
    nInds = mxGetN(inds_in);
    if (mInds != mMx || nInds != nMx) {
        mexErrMsgTxt("First and second argument must have the same size");
    }

    // Check inds input
    mKu = mxGetM(ku_in);
    nKu = mxGetN(ku_in);
    if (nKu != 1) {
        mexErrMsgTxt("Third argument must be a column vector");
    }
    m = mKu;

    Mx = mxGetPr(Mx_in);
    inds = mxGetPr(inds_in);
    Ku = mxGetPr(ku_in);

    // Compute quadratic forms
    diag = mxCreateNumericMatrix(n, 1, mxDOUBLE_CLASS, mxREAL);
    double q[4], y;
    int ind[4];
    int i, j, k;
    for (i = 0; i < n; ++i) {

        // Extract q and row indices
        for(j = 0; j < 4; j++) {
            q[j] = Mx[j+4*i];
            ind[j] = inds[j+4*i] - 1;
        }

        // Compute the quadratic form
        y = 0.0;
        for(j = 0; j < 4; j++) {
            for(k = 0; k < 4; k++) {
                y += Ku[abs(ind[j] - ind[k])]*q[j]*q[k];
            }
        }

        // Save to output array
        ((double*)mxGetPr(diag))[i] = y;
    }
}
