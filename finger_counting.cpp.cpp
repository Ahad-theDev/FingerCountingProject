#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main()
{
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		cout << "can't Open Camera\a";
		return 0;
	}
	int roiWidth = 200, roiHeight = 200;
	Mat roiFrame, background, gbackground, binary, diff;
	Mat persistantFrame;
	Rect roi;
	for (int i = 0; i < 90; i++)
	{
		cap >> background;
		if (i == 89)
		{
			roi = Rect(30, 50, roiWidth, roiHeight);
			cvtColor(background(roi), gbackground, COLOR_BGR2GRAY);
			GaussianBlur(gbackground, gbackground, Size(3, 3), 0);
		}
	}
	Mat frame;
	float t = 0;
	while (true)
	{
		cap >> frame;
		if (frame.empty())
		{
			cout << "Can't open Camera\a";
			return 0;
		}

		roiFrame = frame(roi);
		cvtColor(roiFrame, roiFrame, COLOR_BGR2GRAY);
		GaussianBlur(roiFrame, roiFrame, Size(3, 3), 0);
		absdiff(gbackground, roiFrame, diff);
		threshold(diff, binary, 30, 255, THRESH_BINARY);
		vector<vector<Point>> contours;
		findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		bool bigContour = false;
		if (!contours.empty())
		{
			auto maxContour = *max_element(contours.begin(), contours.end(), [](const auto& a, const auto& b)
				{
					return contourArea(a) < contourArea(b);
				});

			vector<Point> hull;
			vector<int> hullIndices;

			for (size_t i = 0; i < contours.size(); i++)
			{
				convexHull(maxContour, hullIndices, false);
				vector<Point> hull;
				convexHull(maxContour, hull);

				if (!hull.empty())
					polylines(frame(roi), hull, true, Scalar(255, 0, 0), 2);

				if (contourArea(maxContour) > 3000)
				{
					drawContours(frame(roi), contours, (int)i, Scalar(0, 255, 0), 2);
					bigContour = true;

					vector<Vec4i> defects;
					try 
					{
						if (hullIndices.size() > 3 && maxContour.size() > 3)
							convexityDefects(maxContour, hullIndices, defects);
						else
							continue;
					}
					catch (const cv::Exception& e) 
					{
						cerr << "OpenCV Error: " << e.what() << endl;
						continue;
					}

					Moments m = moments(maxContour);
					Point center;
					if (m.m00 != 0)
						center = Point(m.m10 / m.m00, m.m01 / m.m00);
					else
						continue;

					if (!defects.empty() && maxContour.size() > 0)
					{
						int fingerCount = 0;
						for (const auto& d : defects)
						{
							if (d[0] >= maxContour.size()) continue;
							Point start = maxContour[d[0]];
							float depth = d[3] / 256.0f;
							if (depth > 11 && start.y < center.y)
							{
								line(frame(roi), center, start, Scalar(255, 255, 20), 1);
								circle(frame(roi), start, 5, Scalar(0, 5, 244), -1);
								fingerCount++;
							}
						}

						// Progress Ring Animation
						const int maxFingers = 5;
						float percent = min(fingerCount, maxFingers) / (float)maxFingers;
						float angle = 360.0f * percent;
						Point centerAnim(roi.x + roi.width / 2, roi.y + roi.height / 2);
						int radius = min(roi.width, roi.height) / 2;
						// Static ring (same size as ROI)
						circle(frame, centerAnim, radius, Scalar(50, 50, 50), 4);
						// Animated arc
						ellipse(frame, centerAnim, Size(radius, radius), -90, 0, angle, Scalar(0, 255, 0), 4);

						putText(frame, "Fingers: " + to_string(fingerCount), Point(roi.x, roi.y - 10), FONT_HERSHEY_DUPLEX, 0.6, Scalar(0, 255, 255), 2);
					}
				}
			}
		}
		else
		{
			t = 0;
		}
		//  Always draw static circle instead of rectangle
		Point circleCenter(roi.x + roi.width / 2, roi.y + roi.height / 2);
		int circleRadius = min(roi.width, roi.height) / 2;
		circle(frame, circleCenter, circleRadius, Scalar(50, 50, 50), 2);  // static ring
		//  Removed rectangle 
		// rectangle(frame, roi, Scalar(0, 255, 5), 2);
		imshow("frmae", frame);
		imshow("Binary", binary);
		if (waitKey(33) == 'q')
			break;
	}
}
