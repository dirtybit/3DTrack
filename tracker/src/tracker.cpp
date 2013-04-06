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
#include <getopt.h>			
#include "aruco.h"
#include "cvdrawingutils.h"
using namespace cv;
using namespace aruco;

#define NUM_OF_MARKERS 209

struct coord {
    float x;
    float y;
} markerCoords[NUM_OF_MARKERS];

struct option opts[] = {
    {"input", 1, 0, 'i'},
    {"size", 1, 0, 's'},
    {"calib", 1, 0, 'c'},
    {"port", 1, 0, 'p'},
    {"width", 1, 0, 'w'},
    {"height", 1, 0, 'h'},
    {"logframe", 2, 0, 'l'},
    {"verbose", 0, 0, 'v'},
    {"gui", 0, 0, 'g'},
    {0, 0, 0, 0}
};

string inputSource;
string calibFile;
int port = 0;
int width = 640;
int height = 480;
float markerSize = 0;
int guiEnabled = 0;
int verboseEnabled = 0;
int logframeEnabled = 0;
char outputPath[200];
int pyrDownLevel = 0;

sem_t sem;
pthread_mutex_t mutex;
char posData[200];
int found;

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

Mat rmat(Mat u, Mat v, Mat w) {
    Mat m = Mat::eye(4, 4, CV_32F);
	for(int i=0; i<3; i++) {
		m.at<float>(0, i) = u.at<float>(i,0);
		m.at<float>(1, i) = v.at<float>(i,0);
		m.at<float>(2, i) = w.at<float>(i,0);
	}
	return m;
}

Mat tmat(Mat t) {
    Mat m = Mat::eye(4, 4, CV_32F);
	m.at<float>(0, 3) = -t.at<float>(0,0);
	m.at<float>(1, 3) = -t.at<float>(1,0);
	m.at<float>(2, 3) = -t.at<float>(2,0);

	return m;
}

Mat computeTransMat(Mat rvec, Mat tvec)
{
    Mat r(3,3,CV_32F);
    Mat coord = Mat::zeros(3,1,CV_32F);
    Mat u = Mat::zeros(3,1,CV_32F);
    Mat v = Mat::zeros(3,1,CV_32F);
    Mat w = Mat::zeros(3,1,CV_32F);
    Mat rotation = rvec;
    u.at<float>(0,0) = 1;
    v.at<float>(1,0) = 1;
    w.at<float>(2,0) = 1;
    Rodrigues(rotation, r);
    coord = r * coord;
    coord = coord + tvec;
    u = r * u;
    u = u + tvec;
    v = r * v;
    v = v + tvec;
    w = r * w;
    w = w + tvec;
    u -= coord;
    v -= coord;
    w -= coord;
    Mat R = rmat(u, v, w);
    Mat T = tmat(coord);
    Mat M = R*T;

    return M;
}

int localize(MarkerDetector &markerDetector, Mat &frame, vector<Marker> &markers, CameraParameters cameraParameters, float markerSize, Mat &pos3D)
{
	markerDetector.detect(frame, markers, cameraParameters, markerSize);
	    
	float maxArea = 0;
	float markerArea = 0;
	Marker *chosen = NULL;

    for (unsigned int i=0; i < markers.size(); i++) {
		markerArea = markers[i].getArea();
		if (markerArea > maxArea) {
			chosen = &markers[i];
			maxArea = markerArea;
		}
	}

	if (chosen) {
        Mat position = Mat::zeros(4,1,CV_32F);
        Mat transMat;
        position.at<float>(3, 0) = 1;
        transMat = computeTransMat(chosen->Rvec, chosen->Tvec);
        position = transMat*position;

		float x = markerCoords[chosen->id].x;
		float y = markerCoords[chosen->id].y;
        float z = position.at<float>(1, 0);
        pos3D.at<float>(0, 0) = x - position.at<float>(2, 0);
        pos3D.at<float>(1, 0) = y - position.at<float>(0, 0);
		pos3D.at<float>(2, 0) = z;
		pos3D.at<float>(3, 0) = 1;

        if (guiEnabled)
			chosen->draw(frame, Scalar(0,0,255), 1);
		
		return 1;
	}
	else
		return 0;
}

void* run_server(void *args)
{
    int sock, conn, flag = 1;
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
    server_addr.sin_port = htons(port);
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

    printf("\nTCPServer Waiting for client on port %d\n", port);
	fflush(stdout); 

	sin_size = sizeof(client_addr);
	conn = accept(sock, (struct sockaddr *) &client_addr, &sin_size);
	cout << "Connected" << endl;

	while(1) {
		sem_wait(&sem);
		pthread_mutex_lock(&mutex);
		if (found)
            send(conn, posData, strlen(posData), 0);
		pthread_mutex_unlock(&mutex);
	}
}

int main(int argc,char **argv)
{
	MarkerDetector markerDetector;
	VideoCapture capture;
	vector<Marker> markers;
	Mat inputImage;
	CameraParameters cameraParameters;
	pair<double,double> avgTime(0,0) ;//determines the average time required for detection
    float thresParam1, thresParam2;
    double tick;
    int c;
    char frameFileName[20];
    int failedFrames = 0;
    int totalFrames = 0;
    Mat pos3D = Mat::zeros(4, 1, CV_32F);

    // Parse command-line arguments
    while ((c = getopt_long(argc, argv, "w:h:i:s:c:p:l::vg", opts, NULL)) != -1) {
        switch (c) {
        case 'w':
            width = atoi(optarg);
            break;
        case 'h':
            height = atoi(optarg);
            break;
        case 'i':
            inputSource = optarg;
            break;
        case 's':
            markerSize = atof(optarg);
            break;
        case 'c':
            calibFile = optarg;
            break;
        case 'p':
            port = atoi(optarg);
            break;
        case 'l':
            logframeEnabled = 1;
            if (optarg)
                sprintf(outputPath, "%s", optarg);
            else
                sprintf(outputPath, "./logs");
            break;
        case 'v':
            verboseEnabled = 1;
            break;
        case 'g':
            guiEnabled = 1;
            break;
        default:
            printf("Invalid option %c\n", c);
            return 1;
        }
    }

	// Initialize sync 	
	pthread_t child;
	pthread_attr_t attr;
	pthread_attr_init(&attr);

	pthread_mutex_init(&mutex, NULL);
	sem_init(&sem, 0, 0);

    // Initialize marker coordinate mapping
	initMarkerCoordinates();

    // GUI window if enabled
    if (guiEnabled)
        namedWindow("in",1);

    //read from camera or from  file
    if (inputSource == "live") {
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
    if (calibFile != "") {
        capture >> inputImage;
        cameraParameters.readFromXMLFile(calibFile);
        cameraParameters.resize(inputImage.size());
    }

    if (pyrDownLevel > 0)
        markerDetector.pyrDown(pyrDownLevel);

    //markerDetector.getThresholdParams(ThresParam1, ThresParam2);
    //markerDetector.setCornerRefinementMethod(MarkerDetector::LINES);

    thresParam1 = 13;
    thresParam2 = 2;
    markerDetector.setThresholdParams(thresParam1, thresParam2);

    tick = (double)getTickCount(); // Time of detection

    // If a port number specified, then dispatch the thread
    if (port)
        pthread_create(&child, &attr, run_server, NULL);

    int res;
    while (capture.grab())
    {
        capture.retrieve(inputImage);

        avgTime.first=((double)getTickCount() - tick) / getTickFrequency();

        totalFrames++; //number of images captured
        tick = (double)getTickCount();

        res = localize(markerDetector, inputImage, markers, cameraParameters, markerSize, pos3D);

        if (res == 0) // no marker in the current frame
            failedFrames++;

        if(logframeEnabled && (res == 0)) {
            sprintf(frameFileName, "%sframe_%d.png", outputPath, totalFrames);
            imwrite(frameFileName, inputImage);
            cout << "Frame " << totalFrames << " saved to: " << frameFileName << endl;
        }

        float x = pos3D.at<float>(0, 0);
        float y = pos3D.at<float>(1, 0);
        float z = pos3D.at<float>(2, 0);

        pthread_mutex_lock(&mutex);
        found = res;
        sprintf(posData, "%.6f, %.6f, %.6f, %.6f", x, y, z, 1000*avgTime.first);
        pthread_mutex_unlock(&mutex);
        sem_post(&sem);

        if (found && verboseEnabled)
            fprintf(stderr, "[%5d/%5d] Pos = %s\n", failedFrames, totalFrames, posData);

        if (guiEnabled) {
            imshow("in", inputImage);
            waitKey(10);
        }
    }

	return 0;
}
