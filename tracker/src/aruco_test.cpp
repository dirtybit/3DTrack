#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <semaphore.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include "aruco.h"
#include "cvdrawingutils.h"
using namespace cv;
using namespace aruco;

#define PORT 9090

sem_t sem;
pthread_mutex_t mutex;

int width=320;
int height=240;

char pos_data[200];

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

cv::Mat localize(MarkerDetector &markerDetector, cv::Mat &frame, vector<Marker> &markers, CameraParameters cameraParameters, float markerSize)
{
	markerDetector.detect(frame, markers, cameraParameters, markerSize);
	    
	float maxArea = 0;
	float markerArea = 0;
	Marker *chosen = NULL;
	cv::Mat pos3D = cv::Mat::zeros(4, 1, CV_32F);

	for (unsigned int i=0;i<markers.size();i++) {
		markerArea = markers[i].getArea();
		if (markerArea > maxArea) {
			chosen = &markers[i];
			maxArea = markerArea;
		}
	}

	if (chosen) {
		cv::Mat r(3,3,CV_32F);
		cv::Mat coord = cv::Mat::zeros(3,1,CV_32F);
		cv::Mat coord_ = cv::Mat::zeros(4,1,CV_32F);
		coord_.at<float>(3, 0) = 1;
		cv::Mat u = cv::Mat::zeros(3,1,CV_32F);
		cv::Mat v = cv::Mat::zeros(3,1,CV_32F);
		cv::Mat w = cv::Mat::zeros(3,1,CV_32F);
		u.at<float>(0,0) = 1;
		v.at<float>(1,0) = 1;
		w.at<float>(2,0) = 1;
		cv::Mat rotation = chosen->Rvec;
		Rodrigues(rotation, r);
		coord = r * coord;
		coord = coord + chosen->Tvec; 
		u = r * u;
		u = u + chosen->Tvec;       
		v = r * v;
		v = v + chosen->Tvec; 
		w = r * w;
		w = w + chosen->Tvec;
		u -= coord;
		v -= coord;
		w -= coord;
		cv::Mat R = rmat(u, v, w);
		cv::Mat T = tmat(coord);
		cv::Mat M = R*T;
		coord_ = M*coord_;
		pos3D.at<float>(0, 0) = -coord_.at<float>(2, 0);
		pos3D.at<float>(1, 0) = -coord_.at<float>(0, 0);
		pos3D.at<float>(2, 0) = coord_.at<float>(1, 0);
	}

	return pos3D;
}

void* run_server(void *args)
{
	int sock, conn, bytes_received, flag = 1;
	struct sockaddr_in server_addr, client_addr;
	socklen_t sin_size;

	if ((sock = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
		perror("Socket");
		exit(1);
	}

	if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof (int)) == -1) {
		perror("Setsockopt");
		exit(1);
	}

	server_addr.sin_family = AF_INET;
	server_addr.sin_port = htons(PORT);
	server_addr.sin_addr.s_addr = INADDR_ANY;
	bzero(&(server_addr.sin_zero), 8);

	if (bind(sock, (struct sockaddr *) &server_addr, sizeof (struct sockaddr))
	    == -1) {
		perror("Unable to bind");
		exit(1);
	}

	if (listen(sock, 5) == -1) {
		perror("Listen");
		exit(1);
	}

	printf("\nTCPServer Waiting for client on port %d\n", PORT);
	fflush(stdout); 

	sin_size = sizeof(client_addr);
	conn = accept(sock, (struct sockaddr *) &client_addr, &sin_size);
	cout << "Connected" << endl;

	while(1) {
		sem_wait(&sem);
		pthread_mutex_lock(&mutex);
		send(conn, pos_data, strlen(pos_data), 0);
		pthread_mutex_unlock(&mutex);
	}
}

int main(int argc,char **argv)
{
	string inputSource;
	string intrinsicFile;
	float markerSize=-1;
	int pyrDownLevel = 0;
	MarkerDetector markerDetector;
	VideoCapture capture;
	vector<Marker> markers;
	Mat inputImage;
	CameraParameters cameraParameters;
	pair<double,double> avgTime(0,0) ;//determines the average time required for detection
	float thresParam1, thresParam2;		
	int index=0;
	Mat pos3D;
	double tick;

	// Initialize sync 	
	pthread_t child;
	pthread_attr_t attr;
	pthread_attr_init(&attr);

	pthread_mutex_init(&mutex, NULL);
	sem_init(&sem, 0, 0);


	pthread_create(&child, &attr, run_server, NULL);

	try {
		// Read command-line arguments
		if (argc<2) {
			cerr<<"Invalid number of arguments"<<endl;
			cerr<<"Usage: (in.avi|live) [width] [height] [intrinsics.yml] [size]"<<endl;
			return false;
		}

		inputSource=argv[1];

		if (argc > 2)
			width=atoi(argv[2]);
		if (argc > 3)
			height=atoi(argv[3]);

		if (argc > 4)
			intrinsicFile=argv[4];
		if (argc > 5)
			markerSize=atof(argv[5]);

		if (argc==5)
			cerr<<"NOTE: You need makersize to see 3d info!!!!"<<endl;

		//read from camera or from  file
		if (inputSource=="live") {
			cout << capture.open(0) << endl;

			capture.set(CV_CAP_PROP_FRAME_WIDTH, width);
			capture.set(CV_CAP_PROP_FRAME_HEIGHT, height);
		
		}
		else  
			capture.open(inputSource);
		
		if (!capture.isOpened()) {
			cerr << "Could not open video" << endl;
			return -1;

		}

		// Read camera paramters
		if (intrinsicFile!="") {
			capture >> inputImage;
			cameraParameters.readFromXMLFile(intrinsicFile);
			cameraParameters.resize(inputImage.size());
		}

		if (pyrDownLevel > 0)
			markerDetector.pyrDown(pyrDownLevel);

		cv::namedWindow("in", 1);
		
		//markerDetector.getThresholdParams(ThresParam1, ThresParam2);
		//markerDetector.setCornerRefinementMethod(MarkerDetector::LINES);
		
		thresParam1 = 13;
		thresParam2 = 2;
		markerDetector.setThresholdParams(thresParam1, thresParam2);

		tick = (double)getTickCount(); // Time of detection
            
		int k=0;
		while (capture.grab())
		{
			capture.retrieve(inputImage);

			avgTime.first=((double)getTickCount() - tick) / getTickFrequency();
			//cout << "Time detection = "<< 1000*avgTime.first << " milliseconds" << endl;

			index++; //number of images captured
			tick = (double)getTickCount();
	    
			pos3D = localize(markerDetector, inputImage, markers, cameraParameters, markerSize);

			float x = pos3D.at<float>(0, 0); //chosen->id % 6;
			float y = pos3D.at<float>(1, 0); //chosen->id / 6;
			float z = pos3D.at<float>(2, 0);
			pthread_mutex_lock(&mutex);
			sprintf(pos_data, "Fm = %.6f, %.6f, %.6f", x, y, z);
			cout << pos_data << endl;
			pthread_mutex_unlock(&mutex);			
			sem_post(&sem);
		}
	} catch (std::exception &ex) {
		cout << "Exception :" << ex.what() << endl;
	}

	return 0;
}

