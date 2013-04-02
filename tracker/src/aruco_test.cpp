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

cv::Mat localize(cv::Mat frame)
{
	MDetector.detect(TheInputImage,TheMarkers,TheCameraParameters,TheMarkerSize);
	    
	float maxArea = 0;
	float markerArea = 0;
	Marker *chosen = NULL;
	cv::Mat pos3D = cv::Mat::zeros(4, 1, CV_32F);

	for (unsigned int i=0;i<TheMarkers.size();i++) {
		markerArea = TheMarkers[i].getArea();
		if (markerArea > maxArea) {
			chosen = &TheMarkers[i];
			maxArea = markerArea;
		}
	}

	if (chosen) {
		cv::Mat r(3,3,CV_32F);
		cv::Mat coord_ = cv::Mat::zeros(4,1,CV_32F);
		coord_.at<float>(3, 0) = 1;
		cv::Mat coord(3,1,CV_32F);
		cv::Mat u = cv::Mat::zeros(3,1,CV_32F);
		cv::Mat v = cv::Mat::zeros(3,1,CV_32F);
		cv::Mat w = cv::Mat::zeros(3,1,CV_32F);

		//Unit vectors of Fm
		u.at<float>(0,0) = 1;
		v.at<float>(1,0) = 1;
		w.at<float>(2,0) = 1;
		
		coord.at<float>(0,0) = 0;
		coord.at<float>(1,0) = 0;
		coord.at<float>(2,0) = 0;
		
		cv::Mat rotation = chosen->Rvec;
		Rodrigues(rotation, r);
		coord = r * coord;
		coord = coord + chosen->Tvec;
		
		// Find unit vectors of Fm wrt Fg
		u = r * u;
		u = u + chosen->Tvec;       
		v = r * v;
		v = v + chosen->Tvec; 
		w = r * w;
		w = w + chosen->Tvec;
		u -= coord;
		v -= coord;
		w -= coord;
		
		// Find translation, rotation, and transformation matrix for Fm -> Fg
		cv::Mat R = rmat(u, v, w);
		cv::Mat T = tmat(coord);
		cv::Mat M = R*T;
		coord_ = M*coord_;

		pos3D = coord_;
	}

	return pos3D;
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
        cv::namedWindow("in",1);
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
	Mat pos3D;
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
	    
	    pos3D = localize(TheInputImage);

	    int x = 0; //chosen->id % 6;
	    int y = 0; //chosen->id / 6;
		
	    cout<< "3D position [Fg] = (" << x-pos3D.at<float>(2,0) << ", " << y - pos3D.at<float>(0,0) << ", " << pos3D.at<float>(1,0) << ")" <<endl;

	    cv::imshow("in",TheInputImage);
	    waitKey(10);
		
	}

    } catch (std::exception &ex) {
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
