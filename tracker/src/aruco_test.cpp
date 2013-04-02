/*****************************************************************************************
Copyright 2011 Rafael Muñoz Salinas. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY Rafael Muñoz Salinas ''AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Rafael Muñoz Salinas OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Rafael Muñoz Salinas.
********************************************************************************************/
#include <iostream>
#include <fstream>
#include <sstream>
#include "aruco.h"
#include "cvdrawingutils.h"
using namespace cv;
using namespace aruco;

string TheInputVideo;
string TheIntrinsicFile;
float TheMarkerSize=-1;
int ThePyrDownLevel;
MarkerDetector MDetector;
VideoCapture TheVideoCapturer;
vector<Marker> TheMarkers;
Mat TheInputImage,TheInputImageCopy;
CameraParameters TheCameraParameters;
void cvTackBarEvents(int pos,void*);
bool readCameraParameters(string TheIntrinsicFile,CameraParameters &CP,Size size);

pair<double,double> AvrgTime(0,0) ;//determines the average time required for detection
double ThresParam1,ThresParam2;
int iThresParam1,iThresParam2;
int waitTime=0;
int fps=15;
int width=320;
int height=240;

/************************************
 *
 *
 *
 *
 ************************************/
bool readArguments ( int argc,char **argv )
{
    if (argc<2) {
        cerr<<"Invalid number of arguments"<<endl;
        cerr<<"Usage: (in.avi|live) [fps] [width] [height] [intrinsics.yml] [size]"<<endl;
        return false;
    }
    TheInputVideo=argv[1];
    if (argc>=3)
        fps=atoi(argv[2]);
    if (argc>=4)
        width=atoi(argv[3]);
    if (argc>=5)
        height=atoi(argv[4]);

    if (argc>=6)
        TheIntrinsicFile=argv[5];
    if (argc>=7)
        TheMarkerSize=atof(argv[6]);

    if (argc==6)
        cerr<<"NOTE: You need makersize to see 3d info!!!!"<<endl;
    return true;
}
/************************************
 *
 *
 *
 *
 ************************************/
void printP(cv::Mat p) {
	for(int i=0; i < 3; i++)
		cout << p.at<float>(i, 0) << " ";
	cout << endl;
}

cv::Mat rmat(cv::Mat u, cv::Mat v, cv::Mat w) {
	cv::Mat m = cv::Mat::eye(4, 4, CV_32F);
	for(int i=0; i<3; i++) {
		m.at<float>(0, i) = u.at<float>(i,0);
		m.at<float>(1, i) = v.at<float>(i,0);
		m.at<float>(2, i) = w.at<float>(i,0);
	}
	return m;
}

cv::Mat tmat(cv::Mat t) {
	cv::Mat m = cv::Mat::eye(4, 4, CV_32F);
	m.at<float>(0, 3) = -t.at<float>(0,0);
	m.at<float>(1, 3) = -t.at<float>(1,0);
	m.at<float>(2, 3) = -t.at<float>(2,0);

	return m;
}

int main(int argc,char **argv)
{
    try
    {
        if (readArguments (argc,argv)==false) {
            return 0;
        }
        //parse arguments
        ;
        //read from camera or from  file
        if (TheInputVideo=="live") {
		cout << TheVideoCapturer.open(0) << endl;
            waitTime=100;

		TheVideoCapturer.set(CV_CAP_PROP_FPS, fps);
		TheVideoCapturer.set(CV_CAP_PROP_FRAME_WIDTH, width);
		TheVideoCapturer.set(CV_CAP_PROP_FRAME_HEIGHT, height);
		
        }
        else  TheVideoCapturer.open(TheInputVideo);
        //check video is open
        if (!TheVideoCapturer.isOpened()) {
            cerr<<"Could not open video"<<endl;
            return -1;

        }

        //read first image to get the dimensions
        TheVideoCapturer>>TheInputImage;

        //read camera parameters if passed
        if (TheIntrinsicFile!="") {
            TheCameraParameters.readFromXMLFile(TheIntrinsicFile);
            TheCameraParameters.resize(TheInputImage.size());
        }
        //Configure other parameters
        if (ThePyrDownLevel>0)
            MDetector.pyrDown(ThePyrDownLevel);


        //Create gui

        //cv::namedWindow("thres",1);
        //cv::namedWindow("in",1);
        MDetector.getThresholdParams( ThresParam1,ThresParam2);
        MDetector.setCornerRefinementMethod(MarkerDetector::LINES);
        iThresParam1=ThresParam1;
        iThresParam2=ThresParam2;
        //cv::createTrackbar("ThresParam1", "in",&iThresParam1, 13, cvTackBarEvents);
        //cv::createTrackbar("ThresParam2", "in",&iThresParam2, 13, cvTackBarEvents);

        ThresParam1=13;
        ThresParam2=2;
	MDetector.setThresholdParams(ThresParam1,ThresParam2);

        char key=0;
        int index=0;
        //capture until press ESC or until the end of the video

	double tick = (double)getTickCount();//for checking the speed
            
        while ( key!=27 && TheVideoCapturer.grab())
        {
            TheVideoCapturer.retrieve( TheInputImage);
            //copy image

            AvrgTime.first=((double)getTickCount()-tick)/getTickFrequency();
            cout<<"Time detection="<<1000*AvrgTime.first<<" milliseconds"<<endl;

            index++; //number of images captured
            tick = (double)getTickCount();//for checking the speed
            //Detection of markers in the image passed
            MDetector.detect(TheInputImage,TheMarkers,TheCameraParameters,TheMarkerSize);
            //chekc the speed by calculating the mean speed of all iterations

            //print marker info and draw the markers in image
	    float maxArea = 0;
	    float markerArea = 0;
	    Marker *chosen = NULL;

            //TheInputImage.copyTo(TheInputImageCopy);
	    for (unsigned int i=0;i<TheMarkers.size();i++) {
		    markerArea = TheMarkers[i].getArea();
		    if (markerArea > maxArea) {
			    chosen = &TheMarkers[i];
			    maxArea = markerArea;
		    }
	    }
//	    cout << "----------------------" << endl;
	    //cv::Point2f center = chosen.getCenter();                
	    if (chosen) {
		    //cout << chosen << endl;				
		    //cout<< "2D (" << center.x << ", " << center.y << ")" <<endl;
		    //cout<< "2D (" << center.x << ", " << center.y << ")" <<endl;
		    cv::Mat r(3,3,CV_32F);
		    cv::Mat coord(3,1,CV_32F);
		    cv::Mat coord_ = cv::Mat::zeros(4,1,CV_32F);
		    coord_.at<float>(3, 0) = 1;
		    cv::Mat u(3,1,CV_32F);
		    cv::Mat v(3,1,CV_32F);
		    cv::Mat w(3,1,CV_32F);
		    //cv::Mat u_(3,1,CV_32F);
		    //cv::Mat v_(3,1,CV_32F);
		    // cv::Mat w_(3,1,CV_32F);
		    u.at<float>(0,0) = 1;
		    u.at<float>(1,0) = 0;
		    u.at<float>(2,0) = 0;
		    v.at<float>(0,0) = 0;
		    v.at<float>(1,0) = 1;
		    v.at<float>(2,0) = 0;
		    w.at<float>(0,0) = 0;
		    w.at<float>(1,0) = 0;
		    w.at<float>(2,0) = 1;
		    coord.at<float>(0,0) = 0;
		    coord.at<float>(1,0) = 0;
		    coord.at<float>(2,0) = 0;
		    cv::Mat rotation = chosen->Rvec;
		    Rodrigues(rotation, r);
		    coord = r * coord;
		    coord = coord + chosen->Tvec;
		    //cout << "After tvec: ";
		    //printP(coord);
		    //cout << "After rvec: ";
		    //printP(coord);      
		    u = r * u;
		    u = u + chosen->Tvec;       
		    v = r * v;
		    v = v + chosen->Tvec; 
		    w = r * w;
		    w = w + chosen->Tvec;
		    u -= coord;
		    v -= coord;
		    w -= coord;
		    //cv::normalize(u, u_);
		    //cv::normalize(v, v_);
		    //cv::normalize(w, w_);
		    cv::Mat R = rmat(u, v, w);
		    cv::Mat T = tmat(coord);
		    cv::Mat M = R*T;
		    coord_ = M*coord_;
		    //printP(u);
		    //printP(v);
		    //printP(w);
		    //cout << "Origin: ";
		    //printP(coord);
		    //printP(u_);
		    //printP(v_);
		    //printP(w_);
//		    cout << "Perim " << chosen->getPerimeter() << endl; 
//		    cout << "Area " << chosen->getArea() << endl; 
		    int x = chosen->id % 6;
		    int y = chosen->id / 6;
		    cout<< "3D position [Fg] = (" << x-coord_.at<float>(2,0) << ", " << y - coord_.at<float>(0,0) << ", " << coord_.at<float>(1,0) << ")" <<endl;
//		    cout<< "3D position [Fm] = (" << coord_.at<float>(0,0) << ", " << coord_.at<float>(1,0) << ", " << coord_.at<float>(2,0) << ")" <<endl;
//		    cout<< "3D position [Fc] = (" << coord.at<float>(0,0) << ", " << coord.at<float>(1,0) << ", " << coord.at<float>(2,0) << ")" <<endl;
		    //chosen->draw(TheInputImageCopy,Scalar(0,0,255),1);
//		    cout << "----------------------" << endl;
		    //if (TheCameraParameters.isValid())
			    //CvDrawingUtils::draw3dAxis(TheInputImageCopy, *chosen, TheCameraParameters);
	    }
	    /*
            //print other rectangles that contains no valid markers
            for (unsigned int i=0;i<MDetector.getCandidates().size();i++) {
                aruco::Marker m( MDetector.getCandidates()[i],999);
                m.draw(TheInputImageCopy,cv::Scalar(255,0,0));
            }
	    */


            //draw a 3d cube in each marker if there is 3d info
	    
            

            //DONE! Easy, right?
            //cout<<endl<<endl<<endl;
            //show input with augmented information and  the thresholded image
            //cv::imshow("in",TheInputImageCopy);
            //cv::imshow("thres",MDetector.getThresholdedImage());

            //key=cv::waitKey(waitTime);//wait for key to be pressed
        }

    } catch (std::exception &ex)

    {
        cout<<"Exception :"<<ex.what()<<endl;
    }

}
/************************************
 *
 *
 *
 *
 ************************************/
/*
void cvTackBarEvents(int pos,void*)
{
    if (iThresParam1<3) iThresParam1=3;
    if (iThresParam1%2!=1) iThresParam1++;
    if (ThresParam2<1) ThresParam2=1;
    ThresParam1=iThresParam1;
    ThresParam2=iThresParam2;
    MDetector.setThresholdParams(ThresParam1,ThresParam2);
//recompute
    MDetector.detect(TheInputImage,TheMarkers,TheCameraParameters);
    TheInputImage.copyTo(TheInputImageCopy);
    for (unsigned int i=0;i<TheMarkers.size();i++)	TheMarkers[i].draw(TheInputImageCopy,Scalar(0,0,255),1);
    //print other rectangles that contains no valid markers
    /*for (unsigned int i=0;i<MDetector.getCandidates().size();i++) {
        aruco::Marker m( MDetector.getCandidates()[i],999);
        m.draw(TheInputImageCopy,cv::Scalar(255,0,0));
    }

//draw a 3d cube in each marker if there is 3d info
    if (TheCameraParameters.isValid())
        for (unsigned int i=0;i<TheMarkers.size();i++)
            CvDrawingUtils::draw3dCube(TheInputImageCopy,TheMarkers[i],TheCameraParameters);

    cv::imshow("in",TheInputImageCopy);
    cv::imshow("thres",MDetector.getThresholdedImage());
}

*/
