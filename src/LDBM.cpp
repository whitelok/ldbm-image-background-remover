#include <iostream>
#include <string>
#include <math.h>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <eigen3/Eigen/core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseQR>
#include <opencv2/core/eigen.hpp>

#include "LDBM.h"

using namespace std;
using namespace cv;
using namespace Eigen;

bool showImgLBDM(0), saveImgLBDM(0);

typedef Eigen::SparseMatrix<double> SpMat; 
typedef Eigen::Triplet<double> T;

Size imgSize;

Mat matlab_reshape(Mat mat, int rows){
    Mat res;
    res = mat.t();
    res = res.reshape(0, res.size().area() / rows);
    res = res.t();
    
    return res;
}

Mat matlab_colVector(Mat mat){
    Mat tmp = mat.clone();
    tmp = tmp.t();
    tmp = tmp.reshape(0, 1);
    tmp = tmp.t();
    return tmp;
}

void resizeCol(Mat& m, size_t sz, const Scalar& s)
{
    Mat tm(m.rows, m.cols + sz, m.type());
    tm.setTo(s);
    m.copyTo(tm(Rect(Point(0, 0), m.size())));
    m = tm;
}

Mat createPixInds(Size sz){
    Mat ap(sz.width, sz.height, CV_32SC1);
    
    Mat_<int>::iterator it = ap.begin<int>();
    Mat_<int>::iterator end = ap.end<int>();
    
    for (int i = 0; it != end; ++it, ++i){
        *it = i;
    }
    
    ap = ap.t();
    
    return ap;
}

Mat compLapCoeff(Mat winI, double lambda){
    
    Mat Xi, I;
    Xi = winI.clone();
    resizeCol(Xi, 1, Scalar::all(1));
    
    I = Mat::eye(Xi.rows, Xi.rows, CV_64FC1);
    I.at<double>(I.rows-1, I.cols-1) = 0;
    
    Mat fenmu, F, I_F, lapcoeff;
    
    Mat Xi_t = Xi.t();
    Mat tmp1 = Xi*Xi_t;
    Mat tmp2 = lambda*I;
    fenmu = tmp1 + tmp2;
    
    F = fenmu.inv(DECOMP_SVD)*tmp1;
    F = F.t();
    
    I_F = Mat::eye(F.rows, F.rows, CV_64FC1) - F;
    
    lapcoeff = I_F.t() * I_F;
    
    Mat tmp = lapcoeff.clone();
    tmp.convertTo(tmp, CV_8S, 1, 0);
    
    return lapcoeff;
}


SpMat creEigenMat(vector<int> row_inds, vector<int> col_inds,
                  vector<double> vals, Size sz)
{
    std::vector<T> tripletList;
    
    for (int i = 0; i < (int)row_inds.size(); ++i){
        
        double x, y, v_xy;
        y = row_inds[i];
        x = col_inds[i];
        v_xy = vals[i];
        
        tripletList.push_back(T(x,y,v_xy));
    }
    
    SpMat mat(sz.area(), sz.area());
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    
    return mat;
}

SpMat getLap_iccv09_overlapping(Mat imdata, Size winsz, Mat mask, double lambda){
    imdata.convertTo(imdata, CV_64FC3, 1.0/255, 0);
    Size imsz = imdata.size();
    int d = imdata.channels();
    Mat pixInds = createPixInds(imsz);
    
    int numPixInWindow = winsz.area();
    Size halfwinsz = Size(winsz.width/2, winsz.height/2);
    
    Mat scribble_mask;
    scribble_mask = (abs(mask) != 0);
    
    Mat element(winsz, CV_8U, Scalar(1));
    erode(scribble_mask, scribble_mask, element);
    
    Mat tmp;
    Point p1, p2;
    p1 = Point(halfwinsz.width + 1,halfwinsz.height + 1);
    p2 = Point(scribble_mask.cols-halfwinsz.width-1, scribble_mask.rows-halfwinsz.height-1);
    tmp = scribble_mask(Rect(p1, p2));
    tmp = 1 - scribble_mask;
    
    int numPix4Training = sum( sum(tmp) )[0];
    int numNonzeroValue = numPix4Training * numPixInWindow * numPixInWindow;
    
    vector<int> row_inds;
    vector<int> col_inds;
    vector<double> vals;
    
    int len = 0;
    for (int j = halfwinsz.width; j < imsz.width - halfwinsz.width; ++j){
        for (int i = halfwinsz.height; i < imsz.height- halfwinsz.height; ++i){
            if (uchar a = scribble_mask.at<uchar>(i, j))
                continue;
            
            Point p1, p2;
            p1 = Point(j - halfwinsz.width, i - halfwinsz.height);
            p2 = Point(j + halfwinsz.width+1, i + halfwinsz.height+1);
            
            Mat winData = imdata(Rect(p1, p2)).clone();
            winData = matlab_reshape(winData, numPixInWindow);
            winData = winData.reshape(1);
            Mat lapcoeff = compLapCoeff(winData, lambda);
            
            Mat win_inds = pixInds(	Rect(p1,p2));
            Mat rep_row = repeat(matlab_colVector(win_inds), 1, numPixInWindow);
            Mat row_incre = matlab_reshape(rep_row, numPixInWindow*numPixInWindow);
            row_inds.insert(row_inds.end(), row_incre.begin<int>(), row_incre.end<int>());
            
            Mat rep_col = repeat(matlab_colVector(win_inds).t(), numPixInWindow, 1);
            Mat col_incre = matlab_reshape(rep_col, numPixInWindow*numPixInWindow);
            col_inds.insert(col_inds.end(), col_incre.begin<int>(), col_incre.end<int>());
            
            lapcoeff = lapcoeff.t();
            vals.insert(vals.end(), lapcoeff.begin<double>(), lapcoeff.end<double>());
        }
    }
    
    SpMat res;
    res = creEigenMat(row_inds, col_inds, vals, imsz);
    
    return res;
}

Mat getMask_onlineEvaluation(Mat I){
    Mat mask, fore, back;
    
    mask = Mat::zeros(I.size(), CV_32SC1);
    
    fore = (I == 255);
    back = (I == 0);
    
    mask.setTo(1, fore);
    mask.setTo(-1, back);
    
    return mask;
}

SpMat getLap(Mat imdata, Size winsz, Mat mask, double lambda){
    cout << "Computing Laplacian matrix ... ..." << endl;
    SpMat L = getLap_iccv09_overlapping(imdata, winsz, mask, lambda);
    
    return L;
}

SpMat getC(Mat mask, int c){
    cout << "Computing regularization matrix ... ..." << endl;
    
    Mat scribble_mask;
    scribble_mask = (abs(mask) != 0);
    scribble_mask.convertTo(scribble_mask, CV_64FC1, 1.0/255, 0);
    
    scribble_mask = matlab_colVector(scribble_mask);
    
    int numPix = mask.cols * mask.rows;
    
    SpMat diagnal(numPix, numPix);
    diagnal.reserve(Eigen::VectorXd::Constant(numPix, 2));
    
    for (int i = 0; i < numPix; ++i){
        diagnal.insert(i, i) = scribble_mask.at<double>(i);
    }
    
    SpMat C = (double)c * diagnal ;
    
    return C;
}

VectorXd getAlpha_star(Mat mask){
    cout << "Computing preknown alpha values ... ..." << endl;
    
    Mat alpha_star;
    alpha_star = Mat::zeros(mask.size(), CV_8SC1);
    
    alpha_star.setTo(1, mask>0);
    alpha_star.setTo(-1, mask<0);
    
    alpha_star = matlab_colVector(alpha_star);
    
    VectorXd alpha_star_vec;
    cv2eigen(alpha_star, alpha_star_vec);
    
    return alpha_star_vec;
}

Mat solveQurdOpt(SpMat L, SpMat C, VectorXd alpha_star){
    
    Eigen::setNbThreads(1);    
    
    cout << "solving quadratic optimization proble .............." << endl;
    double lambda = 0.000001;
    
    SpMat D(L.rows(), L.cols());
    D.setIdentity();
    
    SpMat b = alpha_star.sparseView();
    
    SpMat A, alpha;
    A = L + C + lambda * D;
    b = C * b;
    
    Eigen::SimplicialLDLT<SpMat> solver;
    cout << "begin to factorize A" << endl;
    solver.compute(A);
    if(solver.info() != Eigen::Success)
        cout << "decomposition failed" << endl;
    else
        cout << "decomposition success" << endl;
    
    cout << "begin to solve !" << endl;
    alpha = solver.solve(b);
    cout << "solve success" << endl;
    
    Mat alpha_mat;
    eigen2cv(MatrixXd(alpha), alpha_mat);
    alpha_mat = alpha_mat.t();
    alpha_mat = alpha_mat.reshape(0, imgSize.width).t();
    
    alpha_mat = alpha_mat * 0.5 + 0.5;
    
    alpha_mat = max(min(alpha_mat, 1), 0);
    
    return alpha_mat;
}

Mat learningBasedMatting(Mat imdata,Mat mask){
    Size winsz = Size(3, 3);
    int c = 800;
    double lambda = 0.000001;
    
    SpMat L, C;
    VectorXd alpha_star;
    
    L = getLap(imdata, winsz, mask, lambda);
    
    C = getC(mask, c);
    
    alpha_star = getAlpha_star(mask);
    
    Mat alpha = solveQurdOpt(L, C, alpha_star);
    
    return alpha;
}


Mat LBDM_Matting(Mat imdata, Mat raw_mask){
    imgSize = imdata.size();
    
    double duration;
    duration  = static_cast<double>(getTickCount());
    string file_name,
    path_im, path_mask, path_save, file_extend,
    fn_im, fn_mask, fn_save;
    
    path_save = "res";
    fn_save = path_save + file_name;
    
    Mat mask = getMask_onlineEvaluation(raw_mask);
    
    Mat alpha = learningBasedMatting(imdata, mask);
    
    duration = static_cast<double>(getTickCount())-duration;
    duration /=getTickFrequency();
    cout << "using:" << duration << "s" << endl;
    
    return alpha;
}
