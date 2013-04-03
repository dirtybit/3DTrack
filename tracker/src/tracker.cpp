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
#define NUM_OF_MARKERS 209

sem_t sem;
pthread_mutex_t mutex;

int width=320;
int height=240;

char pos_data[200];
int found;

struct coord {
	float x;
	float y;
} markerCoords[NUM_OF_MARKERS];

int verbose = 0;
int failedFrameNumber = 0;
int saveFailedFrames = 0;
char* savedFrame = new char[25];
char* outputPath = new char[200];

void initMarkerCoordinates()
{
	for (int i = 0; i < NUM_OF_MARKERS; i++) {
		if (i < 60) {
			markerCoords[i].x = i  % 6;
			markerCoords[i].y = i / 6;
		} else if (i < 65) {
			markerCoords[i].x = (i - 60.0) + 0.5;
			markerCoords[i].y = 0.0;
		} else if (i < 76) {
			markerCoords[i].x = (i - 65.0) * 0.5;
			markerCoords[i].y = 0.5;
		} else if (i < 81) {
			markerCoords[i].x = (i - 76.0) + 0.5;
			markerCoords[i].y = 1.0;
		} else if (i < 92) {
			markerCoords[i].x = (i - 81.0) * 0.5;
			markerCoords[i].y = 1.5;
		} else if (i < 97) {
			markerCoords[i].x = (i - 92.0) + 0.5;
			markerCoords[i].y = 2.0;
		} else if (i < 108) {
			markerCoords[i].x = (i - 97.0) * 0.5;
			markerCoords[i].y = 2.5;
		} else if (i < 113) {
			markerCoords[i].x = (i - 108.0) + 0.5;
			markerCoords[i].y = 3.0;
		} else if (i < 124) {
			markerCoords[i].x = (i - 113.0) * 0.5;
			markerCoords[i].y = 3.5;
		} else if (i < 129) {
			markerCoords[i].x = (i - 124.0) + 0.5;
			markerCoords[i].y = 4.0;
		} else if (i < 140) {
			markerCoords[i].x = (i - 129.0) * 0.5;
			markerCoords[i].y = 4.5;
		} else if (i < 145) {
			markerCoords[i].x = (i - 140.0) + 0.5;
			markerCoords[i].y = 5.0;
		} else if (i < 156) {
			markerCoords[i].x = (i - 145.0) * 0.5;
			markerCoords[i].y = 5.5;
		} else if (i < 161) {
			markerCoords[i].x = (i - 156.0) + 0.5;
			markerCoords[i].y = 6.0;
		} else if (i < 172) {
			markerCoords[i].x = (i - 161.0) * 0.5;
			markerCoords[i].y = 6.5;
		} else if (i < 177) {
			markerCoords[i].x = (i - 172.0) + 0.5;
			markerCoords[i].y = 7.0;
		} else if (i < 188) {
			markerCoords[i].x = (i - 177.0) * 0.5;
			markerCoords[i].y = 7.5;
		} else if (i < 193) {
			markerCoords[i].x = (i - 188.0) + 0.5;
			markerCoords[i].y = 8.0;
		} else if (i < 204) {
			markerCoords[i].x = (i - 193.0) * 0.5;
			markerCoords[i].y = 8.5;
		} else {
			markerCoords[i].x = (i - 204.0) + 0.5;
			markerCoords[i].y = 9.0;

		}
	}
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

int localize(MarkerDetector &markerDetector, cv::Mat &frame, vector<Marker> &markers, CameraParameters cameraParameters, float markerSize, cv::Mat &pos3D)
{
	markerDetector.detect(frame, markers, cameraParameters, markerSize);
	    
	float maxArea = 0;
	float markerArea = 0;
	Marker *chosen = NULL;

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

		float x = markerCoords[chosen->id].x;
		float y = markerCoords[chosen->id].y;
		float z = coord_.at<float>(1, 0);
		pos3D.at<float>(0, 0) = x - coord_.at<float>(2, 0);
		pos3D.at<float>(1, 0) = y - coord_.at<float>(0, 0);
		pos3D.at<float>(2, 0) = z;
		pos3D.at<float>(3, 0) = 1;
		
		return 1;
	}
	else
		return 0;
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
		if (found)
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
	Mat pos3D = cv::Mat::zeros(4, 1, CV_32F);
	double tick;

	// Initialize sync 	
	pthread_t child;
	pthread_attr_t attr;
	pthread_attr_init(&attr);

	pthread_mutex_init(&mutex, NULL);
	sem_init(&sem, 0, 0);

	initMarkerCoordinates();


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
		if (argc > 6)
			verbose=atoi(argv[6]);
		if (argc > 7)
			saveFailedFrames=atoi(argv[7]);
		if (argc > 8)
			outputPath=argv[8];

		if (argc == 5)
			cerr<<"NOTE: You need makersize to see 3d info!!!!"<<endl;

		if (argc == 8) {
			cerr<<"NOTE: You have to specify output path!!!!"<<endl;
			exit(0);
		}

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
		
		int res;
		while (capture.grab())
		{
			capture.retrieve(inputImage);

			avgTime.first=((double)getTickCount() - tick) / getTickFrequency();
			//cout << "Time detection = "<< 1000*avgTime.first << " milliseconds" << endl;

			index++; //number of images captured
			tick = (double)getTickCount();
	    
			res = localize(markerDetector, inputImage, markers, cameraParameters, markerSize, pos3D);

			if(verbose) {
				if (res == 0) // no marker in the current frame
					failedFrameNumber++;

				cerr << "No markers in " << failedFrameNumber << " frames out of " << index << endl;
			}

			if(saveFailedFrames){
				if (res == 0) {
					sprintf(savedFrame, "%sframe_%d.png", outputPath, index);
					imwrite(savedFrame, inputImage);
					cout << "Frame " << index << " saved to: " << savedFrame << endl;
				}

			}

			float x = pos3D.at<float>(0, 0);
			float y = pos3D.at<float>(1, 0);
			float z = pos3D.at<float>(2, 0);
			pthread_mutex_lock(&mutex);
			found = res;
			sprintf(pos_data, "%.6f, %.6f, %.6f, %.6f", x, y, z, 1000*avgTime.first);
			cout << "Fm = " << pos_data << endl;
			pthread_mutex_unlock(&mutex);			
			sem_post(&sem);
		}
	} catch (std::exception &ex) {
		cout << "Exception :" << ex.what() << endl;
	}

	return 0;
}

