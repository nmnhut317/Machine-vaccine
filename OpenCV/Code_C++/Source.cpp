#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

struct giveValue_t {
	vector<Point> point_t;
	vector < tuple<Point, Point, Point, int>> tle_t;
};

struct determinBBValue
{
	Mat backImage;
	Mat bellyImage;
	Mat subImageCopy;
	Mat splitBinaryBellyImage;
	bool verify;
};

class ProcessImage
{
public:
	ProcessImage(int id, string paths, float rateScales);
	~ProcessImage();

	Mat processImageWithLine(Mat& img);
	Mat getSubImg(RotatedRect rect, Mat& img);
	vector<Mat> splitFin(Mat& splitBelly2Draw, Mat& splitBinaryBelly);
	RotatedRect findCNT(Mat& img);

	giveValue_t drawCenterLine(Mat& img2draw, Mat& binaryImg);
	vector < tuple<Point, Point, Point, int>> findPoints(Mat& img, int idx_start = 0, int idx_stop = 1);
	int distanceMultiplePoints(vector<Point> onsite);
	int calculatorLengthByCenterLine(int lengthPx, int lengthFrame = 340, int widthPixelFrame = 1920, bool printValue = true);
	vector<tuple<Mat, Mat, Mat, Mat, bool>> isDetermineBackAndBelly(Mat& subImageToDraw, Mat& subImageThreshold,
		vector < tuple<Point, Point, Point, int>>& onSiteValue, bool onsite = false);
	vector<Point> drawMid(Mat& img, vector < tuple<Point, Point, Point, int>>& array, int numline = 20);
	int convertLenghToInjection(int length);
	void saveImage(Mat& drawframe, Mat& img, Mat& threshold, bool takeImage = false);
	void setUpperHSV(int uh, int us, int uv);
	void setLowerHSV(int lh, int ls, int lv);
	void setZoneSaveImg(Mat& img);
	void setZones(int start, int stop);
	giveValue_t measureLengthWithTail(Mat& img, Mat& binaryImg);
	void findFin(Mat& img, Mat& binaryImg);

private:
	int id;
	string path;
	float rateScale;
	int UpperHSV[3];
	int LowerHSV[3];
	int setValueZone[2];
};

class Objects : public ProcessImage
{
public:
	Objects(int id, string path, float scale);
	~Objects();

private:

};

Objects::Objects(int id, string path, float scale) : ProcessImage(id, path, scale)
{

}

Objects::~Objects()
{
}

ProcessImage::ProcessImage(int id, string paths, float rateScales)
{
	this->id = id;
	this->path = paths;
	this->rateScale = rateScales;
}

ProcessImage::~ProcessImage()
{

}

void ProcessImage::setZones(int start, int stop)
{
	this->setValueZone[0] = start;
	this->setValueZone[1] = stop;
}

void ProcessImage::findFin(Mat& img, Mat& binaryImg)
{
	Mat tempImg[3], output, thresholds, temp, colorFin, binaryFin, output2;

	split(img, tempImg);
	vector<Mat> channels = { tempImg[0] , tempImg[1] , tempImg[2] };

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	temp = tempImg[0].clone();
	output2 = img.clone();

	equalizeHist(tempImg[0], tempImg[0]);
	//Ptr<CLAHE> cle = createCLAHE(10);
	//cle->apply(temp, temp);

	threshold(tempImg[0], thresholds, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

	int morph_size = 2;
	Mat erodes;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
	erode(thresholds, erodes, element, Point(-1, -1), 1);

	bitwise_and(erodes, erodes, binaryFin, binaryImg);
	bitwise_and(img, img, colorFin, binaryFin);

	findContours(binaryFin, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	double minArea = img.cols * 0.05;
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		Rect rect;
		rect = boundingRect(contours[i]);

		if (area < minArea || (rect.br().x > (img.cols*0.8)) || (rect.br().x < (img.cols*0.4))) {
			drawContours(colorFin, contours, i, Scalar(0, 0, 0), FILLED);
			drawContours(binaryFin, contours, i, Scalar(0, 0, 0), FILLED);
		}
	}

	findContours(binaryFin, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	Mat zeros(img.size(), CV_8UC1, Scalar(0,0,0));
	for (int i = 0; i < contours.size(); i++)
	{
		Rect rect;
		rect = boundingRect(contours[i]);
		rectangle(colorFin, rect.tl(), rect.br(), Scalar(255, 0, 0), 1);
		putText(colorFin, to_string(i + 1), rect.tl(), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1);
		drawContours(zeros, contours, i, Scalar(255, 255, 255), FILLED);
		cout << mean(temp, zeros)[0] << endl;
		drawContours(zeros, contours, i, Scalar(0, 0, 0), FILLED);
		line(colorFin, rect.br(), Point(img.cols, rect.br().y), Scalar(0, 0, 255), 1);
		putText(colorFin, to_string((int)round((img.cols - rect.br().x)*0.17)).append("mm"),
			Point(rect.br().x + 20, rect.br().y - 10), FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255), 1);
	}

	cvtColor(binaryFin, binaryFin, COLOR_GRAY2BGR);

	vconcat(colorFin, binaryFin, output);

	imshow("b", output);
	/*imshow("c", output2);*/
}

giveValue_t ProcessImage::measureLengthWithTail(Mat& img2draw, Mat& binaryImg) {
	vector < tuple<Point, Point, Point, int>> points;
	vector<float> site{ 1, 0.9f, 0.5f, 0.35f, 0.2f, 0.15f };
	vector<Point> midPoint;
	giveValue_t	valueSt;

	points = findPoints(binaryImg);
	valueSt.tle_t = points;

	int length = (int)points.size(); /*450*/
	int minDistance = binaryImg.cols;
	int tempIdx = 0;

	for (int i = 0; i < length; i++)
	{
		int t = (int)((site[i] * binaryImg.cols) - 10);

		if (t < 0) t = 0;
		midPoint.push_back(get<2>(points[t]));

		if (site[i] == 0.15f) {

			for (int j = t; j > 0; j--)
			{
				for (int z = 0; z < binaryImg.rows; z++) {
					if (binaryImg.at<uchar>(z, j) == 0)
					{
						int tempDst = (int)sqrt(pow((int)get<2>(points[t]).x - j, 2) + pow((int)get<2>(points[t]).y - z, 2));
						if (tempDst < minDistance) {
							minDistance = tempDst;
							tempIdx = j;
						}
					}
				}
			}
			midPoint.push_back(get<2>(points[tempIdx]));
			circle(img2draw, get<2>(points[tempIdx]), 4, Scalar(255, 0, 0), FILLED, 1);
			break;
		}

		circle(img2draw, get<2>(points[t]), 4, Scalar(255, 0, 0), FILLED, 1);
		line(img2draw, get<0>(points[t]), get<1>(points[t]), Scalar(0, 0, 255), 2);
	}
	polylines(img2draw, midPoint, false, Scalar(0, 255, 0), 2);
	/*cout << midPoint << endl;*/

	valueSt.point_t = midPoint;
	/*imshow("ifs", img2draw);*/

	return valueSt;
}

void ProcessImage::setZoneSaveImg(Mat& frame)
{
	line(frame, Point((int)((frame.cols / 2) - this->setValueZone[1]), 0),
		Point((int)((frame.cols / 2) - this->setValueZone[1]), frame.rows), Scalar(0, 255, 0), 2);
	line(frame, Point((int)((frame.cols / 2) - this->setValueZone[0]), 0),
		Point((int)((frame.cols / 2) - this->setValueZone[0]), frame.rows), Scalar(0, 0, 255), 2);
	putText(frame, "id= " + to_string(this->id),
		Point(50, 50), FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255));
}

void ProcessImage::setUpperHSV(int uh, int us, int uv)
{
	this->UpperHSV[0] = uh;
	this->UpperHSV[1] = us;
	this->UpperHSV[2] = uv;
}

void ProcessImage::setLowerHSV(int lh, int ls, int lv)
{
	this->LowerHSV[0] = lh;
	this->LowerHSV[1] = ls;
	this->LowerHSV[2] = lv;
}

int ProcessImage::convertLenghToInjection(int length)
{
	int positionInjection = int((length * 0.3362 + 2.6474) * 0.63);
	return positionInjection;
}

vector<Point> ProcessImage::drawMid(Mat& img, vector<tuple<Point, Point, Point, int>>& array, int numline)
{
	vector<Point> points;
	int sizeArray = (int)array.size();
	int value = (int)(sizeArray / numline);

	for (int i = 0; i < sizeArray; i++)
	{
		sizeArray = sizeArray - value;
		line(img, get<0>(array[sizeArray]), get<1>(array[sizeArray]), Scalar(0, 0, 255), 1);
		circle(img, get<2>(array[sizeArray]), 4, Scalar(250, 0, 0), FILLED, 1);

		if (sizeArray < value) {
			points.insert(points.begin(), Point(img.cols, get<2>(array[array.size() - 1]).y));
			points.push_back(Point(0, get<2>(array[sizeArray + value]).y));
			break;
		}

		points.push_back(get<2>(array[sizeArray]));
	}

	return points;
}

vector<Mat> ProcessImage::splitFin(Mat& splitBelly2Draw, Mat& splitBinaryBelly)
{
	vector < tuple<Point, Point, Point, int>> pointBelly;
	vector<Point> aNumberOfLines, * newMidBelly;
	vector<Mat> rtImg;

	Mat tempImg = splitBinaryBelly.clone();
	Mat bellyImage;

	pointBelly = findPoints(splitBinaryBelly);
	aNumberOfLines = drawMid(tempImg, pointBelly, 40);


	//#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#
			/*# SPLIT IMAGE INTO 2 PARTS*/

	polylines(splitBelly2Draw, aNumberOfLines, false, Scalar(0, 0, 0), 1);

	bellyImage = splitBelly2Draw.clone();
	newMidBelly = &aNumberOfLines;

		//# # fillpoly apart top side
	newMidBelly->push_back(Point(0, 0));
	newMidBelly->push_back(Point(splitBelly2Draw.cols, 0));

		//# # side belly
	fillPoly(bellyImage, *newMidBelly, Scalar(0, 0, 0));	/*# BELLY*/
	fillPoly(splitBinaryBelly, *newMidBelly, Scalar(0, 0, 0));	/*# BELLY*/

	rtImg.push_back(bellyImage);
	rtImg.push_back(splitBinaryBelly);

	return rtImg;
}

vector<tuple<Mat, Mat, Mat, Mat, bool>> ProcessImage::isDetermineBackAndBelly(Mat& subImageToDraw, Mat& subImageThreshold,
	vector < tuple<Point, Point, Point, int>>& onSiteValue, bool onsite)
{
	Mat subImageCopy = subImageToDraw.clone();
	Mat splitImage = subImageToDraw.clone();
	Mat splitBinaryBellyImage = subImageThreshold.clone();
	bool verify = false;
	vector < tuple<Point, Point, Point, int>> points;
	vector<Point> aNumberofLine;
	vector<Point> newMidBack, * newMidBelly;
	vector<tuple<Mat, Mat, Mat, Mat, bool>> isReturn;

	if (onsite == true) {
		points = onSiteValue;
	}
	else {
		points = findPoints(subImageThreshold);
	}
	aNumberofLine = drawMid(subImageToDraw, points);
	polylines(subImageCopy, aNumberofLine, false, Scalar(0, 0, 0), 2);

	Mat backImage = splitImage.clone();
	Mat bellyImage = splitImage.clone();

	newMidBack = aNumberofLine;
	newMidBelly = &aNumberofLine;

	/*fillpoly apart top side*/
	newMidBelly->push_back(Point(0, 0));
	newMidBelly->push_back(Point(subImageToDraw.cols, 0));

	/*fillpoly apart bottom side*/
	newMidBack.push_back(Point(0, subImageToDraw.rows));
	newMidBack.push_back(Point(subImageToDraw.cols, subImageToDraw.rows));

	/*back side and belly*/
	fillPoly(backImage, newMidBack, Scalar(0, 0, 0));
	fillPoly(bellyImage, *newMidBelly, Scalar(0, 0, 0));

	//backImage = backImage(Range(0, backImage.rows), Range((int)(0.5 * backImage.cols), (int)(0.8 * backImage.cols)));
	//bellyImage = bellyImage(Range(0, bellyImage.rows), Range((int)(0.5 * bellyImage.cols), (int)(0.8 * bellyImage.cols)));

	Mat backGray;
	Mat bellyGray;
	Mat backBinaryImageSub;
	Mat bellyBinaryImageSub;

	cvtColor(backImage, backGray, COLOR_BGR2GRAY);
	cvtColor(bellyImage, bellyGray, COLOR_BGR2GRAY);

	threshold(backGray, backBinaryImageSub, 0, 255, THRESH_BINARY);
	threshold(bellyGray, bellyBinaryImageSub, 0, 255, THRESH_BINARY);

	int meanBackGray = (int)mean(backGray, backBinaryImageSub)[0];
	int meanBellyGray = (int)mean(bellyGray, bellyBinaryImageSub)[0];

	cout << "meanBackGray= " << meanBackGray << " " << "meanBellyGray= " << meanBellyGray << endl;
	if (meanBellyGray > meanBackGray)
	{
		verify = true;
		fillPoly(splitBinaryBellyImage, *newMidBelly, Scalar(0, 0, 0));
		cout << "Continues proccessing then the next step is segment fin" << endl;
	}

	//imshow("back", backImage);
	//imshow("bellyImage", bellyImage);

	isReturn.push_back(make_tuple(backImage, bellyImage, subImageCopy, splitBinaryBellyImage, verify));

	return isReturn;
}

int ProcessImage::calculatorLengthByCenterLine(int lengthPx, int lengthFrame, int widthPixelFrame, bool printValue)
{
	float rate = ((float)lengthFrame / widthPixelFrame);
	int length = (int)(lengthPx * rate);
	if (printValue == true)	cout << "legnth fish= " << length << endl;
	return length;
}

int ProcessImage::distanceMultiplePoints(vector<Point> onsite)
{
	Point previousPoint;
	int dist = 0;

	previousPoint = onsite[0];
	for (int i = 0; i < onsite.size(); i++)
	{
		Point currentPoint = onsite[i];
		dist += (int)sqrt(pow(currentPoint.x - previousPoint.x, 2) + pow(currentPoint.y - previousPoint.y, 2));
		previousPoint = currentPoint;
	}
	cout << "length pixel= " << dist << endl;
	return dist;
}

vector < tuple<Point, Point, Point, int>> ProcessImage::findPoints(Mat& img, int idx_start, int idx_stop)
{
	int original = img.cols;
	int start = original * idx_start;
	int stop = original * idx_stop;

	vector<tuple<Point, Point, Point, int>> points;
	Point first_point;
	Point last_point;
	Point center_point;
	int distance;

	//cout << img.channels() << " "<< img.type()<<endl;

	for (int i = start; i < stop; i++)
	{
		vector<Point> pts;
		for (int j = 0; j < img.rows; j++)
		{
			if (img.at<uchar>(j, i) == 255) {
				pts.push_back(Point(i, j));
				/*pts.push_back(img.at<Point>(j, i));*/
			}
		}

		if (pts.size() > 2)
		{
			first_point = pts[0];
			last_point = pts[pts.size() - 1];
			center_point = Point(pts[0].x, (int)((pts[pts.size() - 1].y + pts[0].y) / 2));
			distance = abs(pts[pts.size() - 1].y - pts[0].y);
			points.push_back(make_tuple(first_point, last_point, center_point, distance));
		}
		pts.clear();
	}

	//for (int i = 0; i < 5; i++)
	//{
	//	cout << "first points "<< get<0>(points[i]) 
	//		<< ", last points "<< get<1>(points[i]) << ", distance " 
	//		<< get<3>(points[i]) <<", center " << get<2>(points[i]) << endl << endl;
	//}

	return points;
}

giveValue_t ProcessImage::drawCenterLine(Mat& img2draw, Mat& binaryImg)
{
	vector < tuple<Point, Point, Point, int>> points;
	vector<float> site{ 0, 0.1f, 0.2f, 0.35f, 0.5f, 0.9f, 1 };
	vector<Point> midPoint;
	giveValue_t	valueSt;

	points = findPoints(binaryImg);
	valueSt.tle_t = points;
	//cout << typeid(site[1]).name()<< '\n';
	//printf("%d\n", (int)((site[2] * img2draw.cols) - 2));

	//int tt = static_cast<int>(((site[2] * img2draw.cols) - 2));
	//cout << tt << " "<< typeid(tt).name() << endl;

	int length = (int)points.size(); /*450*/

	for (int i = 0; i < length; i++)
	{
		int t = (int)((site[i] * img2draw.cols) - 10);

		if (t < 0) t = 0;
		midPoint.push_back(get<2>(points[t]));

		if ((int)(site[i]) == 1) {
			circle(img2draw, get<2>(points[t]), 6, Scalar(255, 0, 0), FILLED, 1);
			break;
		}

		circle(img2draw, get<2>(points[t]), 6, Scalar(255, 0, 0), FILLED, 1);
		line(img2draw, get<0>(points[t]), get<1>(points[t]), Scalar(0, 0, 255), 2);
	}
	polylines(img2draw, midPoint, false, Scalar(0, 255, 0), 2);
	/*cout << midPoint << endl;*/

	valueSt.point_t = midPoint;
	/*imshow("ifs", img2draw);*/

	return valueSt;
}

Mat ProcessImage::getSubImg(RotatedRect rect, Mat& img)
{
	vector<Point2f> boxPts(4);
	rect.points(boxPts.data());

	/* convert float to int */
#if 0
	/*cout << "box" << boxPts << endl;*/
	//vector<Point2i> boxPtsi(4);

	//for (int i = 0; i < 4; i++)
	//{
	//	boxPtsi[i] = boxPts[i];
	//}
#endif

	/* draw points on the picture */
#if 0
	/*drawContours(img, boxPtsi, 1, Scalar(0, 255, 0), 1);*/
	//for (int i = 0; i < 4; i++)
	//{
	//	circle(img, boxPtsi[i], 6, Scalar(0, 255, 255), 1, -1);
	//	putText(img, to_string(i), Point(boxPtsi[i].x - 20, boxPtsi[i].y), FONT_HERSHEY_COMPLEX,1, Scalar(255, 0, 0),1);
	//}

	//imshow("fgt", img);
#endif

	Point2f src_pts[4];
	src_pts[0] = boxPts[0];
	src_pts[1] = boxPts[1];
	src_pts[2] = boxPts[2];
	src_pts[3] = boxPts[3];

	Point2f dst_pts[4];
	dst_pts[0] = Point(0, 0);
	dst_pts[1] = Point(rect.boundingRect().width - 1, 0);
	dst_pts[3] = Point(0, rect.boundingRect().height - 1);
	dst_pts[2] = Point(rect.boundingRect().width - 1, rect.boundingRect().height - 1);

	Mat rotated;
	Size size(rect.boundingRect().width, rect.boundingRect().height);
	Mat PerspectiveTransform = getPerspectiveTransform(src_pts, dst_pts);
	warpPerspective(img, rotated, PerspectiveTransform, size, INTER_LINEAR, BORDER_CONSTANT);
	/*imshow("fgt", rotated);*/
	return rotated;
}

RotatedRect ProcessImage::findCNT(Mat& img)
{
	Mat gray, thresholds;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	RotatedRect rect;

	/*cvtColor(img, gray, COLOR_BGR2GRAY);*/
	threshold(img, thresholds, 0, 255, THRESH_BINARY);
	findContours(thresholds, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);


	if (contours.size() > 0) {
		double largestContour = 0;
		int largestContour_idx = 0;
		for (int i = 0; i < contours.size(); i++)
		{
			double area = contourArea(contours[i]);
			if (area > largestContour) {
				largestContour = area;
				largestContour_idx = i;
			}
		}

		rect = minAreaRect(contours[largestContour_idx]);
	}
	return rect;
}

void ProcessImage::saveImage(Mat& drawframe, Mat& img, Mat& threshold, bool takeImage)
{
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	Rect rect;
	int midFRS = (int)((img.cols / 2) - this->setValueZone[1]);
	int midFRST = (int)((img.cols / 2) - this->setValueZone[0]);

	/*cout << "midFRS=" << midFRS << " ,midFRST=" << midFRST << endl;*/

	findContours(threshold, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	if (contours.size() > 0) {
		double largestContour = 0;
		int largestContour_idx = 0;
		for (int i = 0; i < contours.size(); i++)
		{
			double area = contourArea(contours[i]);
			if (area > largestContour) {
				largestContour = area;
				largestContour_idx = i;
			}
		}

		rect = boundingRect(contours[largestContour_idx]);
		int w = rect.width;
		int h = rect.height;

		int CX = (int)(rect.x + (w / 2));
		int CY = (int)(rect.y + (h / 2));

		if ((int)contourArea(contours[largestContour_idx]) > 5000) {
			/*cout << ", cx=" << CX << ", cy=" << CY << endl;*/
			circle(drawframe, Point(CX, CY), 6, Scalar(0, 0, 255), FILLED);
			rectangle(drawframe, rect.tl(), rect.br(), Scalar(0, 0, 255), 2);

			if ((CX > midFRS) && (CX < midFRST)) {
				processImageWithLine(img);
				cout << "save" << endl;
				this->id += 1;
			}
		}
	}
}

Mat ProcessImage::processImageWithLine(Mat& img)
{
	Mat imgHSV, subBinaryImg, result, subColor, tempImg;
	Mat newBinaryImage(img.rows, img.cols, CV_8UC1, Scalar(0, 0, 0));
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	giveValue_t pointSite, pointSite2;
	int dist;
	vector<tuple<Mat, Mat, Mat, Mat, bool>> BBDetermine;
	vector<Mat> split2Fin;

	vector<Mat> matricesOne;
	vector<Mat> matricesTwo;
	Mat commonFrame, commonFrame1, commonFrame2;

	cvtColor(img, imgHSV, COLOR_BGR2HSV);
	Scalar power(this->UpperHSV[0], this->UpperHSV[1], this->UpperHSV[2]);
	Scalar upper(this->LowerHSV[0], this->LowerHSV[1], this->LowerHSV[2]);

	Mat mask, subCenterLine, imgTail;
	inRange(imgHSV, power, upper, mask);
	mask = ~mask;

	findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	if (contours.size() > 0)
	{
		double largestContour = 0;
		int largestContour_idx = 0;
		for (int i = 0; i < contours.size(); i++)
		{
			double area = contourArea(contours[i]);
			if (area > largestContour) {
				largestContour = area;
				largestContour_idx = i;
			}
		}
		drawContours(newBinaryImage, contours, largestContour_idx, Scalar(255, 255, 255), FILLED);
		RotatedRect rect = findCNT(newBinaryImage);
		subBinaryImg = getSubImg(rect, newBinaryImage);
		/*split(newBinaryImage, newBinaryImage);*/

		bitwise_and(img, img, result, newBinaryImage);
		subColor = getSubImg(rect, result);
		subCenterLine = subColor.clone();
		imgTail = subColor.clone();
		/*findPoints(subBinaryImg);*/

		pointSite = drawCenterLine(subCenterLine, subBinaryImg);
		dist = distanceMultiplePoints(pointSite.point_t);

		/* measure fish with tail */
		pointSite2 = measureLengthWithTail(imgTail, subBinaryImg);
		int dist2 = distanceMultiplePoints(pointSite2.point_t);
		cout << "length tail=" << dist2 << endl;

		int lengthFish = calculatorLengthByCenterLine((int)((1 / rateScale) * dist));

		Mat drawInjectionImage = subColor.clone();

		BBDetermine = isDetermineBackAndBelly(subColor, subBinaryImg, pointSite.tle_t, true);
		split2Fin = splitFin(get<1>(BBDetermine[0]), get<3>(BBDetermine[0]));

		Mat borderImage = drawInjectionImage.clone();

		if (get<4>(BBDetermine[0]) == true)
		{
			cout << "true" << endl;
			int IJP = convertLenghToInjection(lengthFish);
			int injection_pixel = (int)((IJP / 0.17) / (1 / rateScale));
			/*cout << IJP << endl;*/
			int injection_OnImgX = (int)(drawInjectionImage.cols - injection_pixel);
			/*cout << injection_OnImgX << " " << get<1>(pointSite.tle_t[injection_OnImgX])
			<< " " << (int)get<1>(pointSite.tle_t[injection_OnImgX]).y << endl;*/

			copyMakeBorder(drawInjectionImage, borderImage, 50, 100, 50, 50, BORDER_CONSTANT, Scalar(0, 0, 0));
			circle(borderImage, Point(injection_OnImgX + 50,
				(int)get<1>(pointSite.tle_t[injection_OnImgX]).y + 50), 10, Scalar(0, 0, 255), 2);
			putText(borderImage, "Injection position=" + to_string(IJP),
				Point(20, borderImage.rows - 65), FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255));
			putText(borderImage, "Length fish=" + to_string(lengthFish).append("mm"),
				Point(20, borderImage.rows - 20), FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255));
		}
		else {
			cout << "not true condition" << endl;
		}

		tempImg = borderImage.clone();

		matricesOne = { get<0>(BBDetermine[0]), get<1>(BBDetermine[0]), split2Fin[0]};
		matricesTwo = { get<2>(BBDetermine[0]), subCenterLine, subColor };

		vconcat(matricesOne, commonFrame1);
		vconcat(matricesTwo, commonFrame2);
		hconcat(commonFrame1, commonFrame2, commonFrame);

		imshow("subColor", commonFrame);
		imshow("mainImg", tempImg);
		imshow("img tail", imgTail);
		/*findFin(split2Fin[0], split2Fin[1]);*/
	}
	return tempImg;
}

int main()
{
	int id = 0;
	string path_img = "Resources/calibresult1.jpg";
	string path_video = "D:/video/vd3.avi";

	ProcessImage pri(id, path_img, 0.5);
	pri.setUpperHSV(65, 110, 104);
	pri.setLowerHSV(179, 255, 255);
	pri.setZones(40, 60);

	/* image*/
#if 0
	Mat img = imread(path_img);
	cout << "channels of image:"<<img.channels();
	Mat tempImg;
	tempImg = pri.processImageWithLine(img);
	/*imshow("image", tempImg);*/
	
	waitKey(0);
#endif

//#if 0
		/* video */
	VideoCapture capture(path_video);

	// Check if camera opened successfully
	if (!capture.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		return -1;
	}

	//create Background Subtractor objects
	Ptr<BackgroundSubtractor> pBackSub;
	pBackSub = createBackgroundSubtractorMOG2();
	Mat frame, cpyFrame, binaryImg, mainImg;

	while (capture.isOpened()) {
		try {
			// Capture frame-by-frame
			capture.read(frame);
			resize(frame, frame, Size(), 0.5, 0.5, INTER_LINEAR);
			cpyFrame = frame.clone();

			// If the frame is empty, break immediately
			if (frame.empty())
				break;

			pBackSub->apply(frame, binaryImg);
			pri.setZoneSaveImg(frame);
			pri.saveImage(frame, cpyFrame, binaryImg);

			imshow("frame", frame);
			/*imshow("binary", binaryImg);*/
			int key = waitKey(1);
			if (key == 'q') break;
		}
		catch (...) {
			cout << "has a exception...." << endl;
			break;
		}
	}

	capture.release();
	destroyAllWindows();
//#endif

	return 0;
}
