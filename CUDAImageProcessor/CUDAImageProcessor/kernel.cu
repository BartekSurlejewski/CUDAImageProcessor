
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<math.h>
#include <iostream>

#include "opencv2/opencv.hpp"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

using namespace cv;
using namespace std;

#define C_PI 3.141592653589793238462643383279502884197169399375

const string FILEPATH = "lena512.bmp";

void __global__ SwirlCu(int width, int height, int stride, uchar *pRawBitmapOrig, uchar *pBitmapCopy, double factor)
{
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// Test to see if we're testing a valid pixel
	if (i >= height || j >= width) return;

	double cX = (double)width / 2.0f;
	double cY = (double)height / 2.0f;
	double relY = cY - i;
	double relX = j - cX;
	// relX and relY are points in our UV space

	double originalAngle;
	if (relX != 0)
	{
		originalAngle = atan(abs(relY) / abs(relX));
		if (relX > 0 && relY < 0) originalAngle = 2.0f*C_PI - originalAngle;
		else if (relX <= 0 && relY >= 0) originalAngle = C_PI - originalAngle;
		else if (relX <= 0 && relY < 0) originalAngle += C_PI;
	}
	else
	{
		if (relY >= 0) originalAngle = 0.5f * C_PI;
		else originalAngle = 1.5f * C_PI;
	}

	double radius = sqrt(relX*relX + relY * relY);
	
	// Equation that determines how much to rotate image by
	double newAngle = originalAngle + 1 / (factor*radius + (4.0f / C_PI));

	// Transform source UV coordinates back into bitmap coordinates
	int srcX = (int)(floor(radius * cos(newAngle) + 0.5f));
	int srcY = (int)(floor(radius * sin(newAngle) + 0.5f));
	srcX += cX;
	srcY += cY;
	srcY = height - srcY;
	// Clamp the source to legal image pixel
	if (srcX < 0) srcX = 0;
	else if (srcX >= width) srcX = width - 1;
	if (srcY < 0) srcY = 0;
	else if (srcY >= height) srcY = height - 1;
	
	// Set the pixel color
	pRawBitmapOrig[i*stride / 4 + j] = pBitmapCopy[srcY*stride / 4 + srcX];
}

int main()
{
	Mat image(512, 512, CV_8UC3, cv::Scalar::all(0));
	
	image = imread(FILEPATH);
	namedWindow("Display window", WINDOW_AUTOSIZE);

	if (!image.data)                              
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	const int rows = image.rows;
	const int columns = image.cols;
	cout << "ROWS: " << rows << endl;
	cout<<" COLUMNS: " << columns;

	uchar* h_image = image.data;
	uchar* d_image = new uchar[rows * columns];
	uchar* d_imageCopy = new uchar[rows * columns];

	//Copy the data to the device
	cudaMemcpy(d_image, h_image, sizeof(uchar*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_imageCopy, d_image, sizeof(uchar*), cudaMemcpyDeviceToDevice);

	SwirlCu <<<16, 16>>> (rows, columns, rows * 4, d_image, d_imageCopy, 0.005f);
	cudaThreadSynchronize();

	// Copy the data back to the host
	cudaMemcpy(h_image, d_image, sizeof(uchar*), cudaMemcpyDeviceToHost);

	image.data = h_image;

	imshow("Display window", image);

	/*delete[] h_image;
	delete[] d_image;
	delete[] d_imageCopy;*/

	waitKey(5000);											// Wait for a keystroke in the window
	return 0;
}
