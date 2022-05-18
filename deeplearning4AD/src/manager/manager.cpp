#include "manager.hpp"
using std::vector;
using namespace cv;
static Logger gLogger;

Trtyolosort::Trtyolosort(char *yolo_engine_path,
						 char *sort_engine_path,
						 char *hrnet_engine_path,
						 char *ufld_engine_path){
	sort_engine_path_ = sort_engine_path;
	yolo_engine_path_ = yolo_engine_path;
	trt_engine = yolov5_trt_create(yolo_engine_path_);
	printf("create yolov5-trt , instance = %p\n", trt_engine);
	DS = new DeepSort(sort_engine_path_, 128, 256, 0, &gLogger);
	
	hrnet_engine_path_ = hrnet_engine_path;
	hrnet_trt_engine = hrnet_trt_create(hrnet_engine_path_);
	printf("create hrnet-trt , instance = %p\n", hrnet_trt_engine);

	ufld_engine_path_ = ufld_engine_path;
	ufld_trt_engine = ufld_trt_create(ufld_engine_path_);
	printf("create ufld-trt , instance = %p\n", ufld_trt_engine);

}
void Trtyolosort::showDetection(cv::Mat& img, std::vector<DetectBox>& boxes) {
    cv::Mat temp = img.clone();
    for (auto box : boxes) {
        cv::Point lt(box.x1, box.y1);
        cv::Point br(box.x2, box.y2);
        cv::rectangle(temp, lt, br, cv::Scalar(255, 0, 0), 1);
        //std::string lbl = cv::format("ID:%d_C:%d_CONF:%.2f", (int)box.trackID, (int)box.classID, box.confidence);
		//std::string lbl = cv::format("ID:%d_C:%d", (int)box.trackID, (int)box.classID);
		// std::string lbl = cv::format("ID:%d_x:%f_y:%f",(int)box.trackID,(box.x1+box.x2)/2,(box.y1+box.y2)/2);
		std::string lbl = cv::format("ID:%d",(int)box.trackID);
        cv::putText(temp, lbl, lt, cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,255,0));
    }
    cv::imshow("img_deepsort_yolo", temp);
    cv::waitKey(1);
}
int Trtyolosort::TrtDetect(cv::Mat &frame,float &conf_thresh,std::vector<DetectBox> &det){
	// yolo detect
	auto ret = yolov5_trt_detect(trt_engine, frame, conf_thresh,det);
	DS->sort(frame,det);
	// showDetection(frame,det);
	return 1 ;
	
}

void Trtyolosort::TrtSeg(cv::Mat &frame_in, cv::Mat &frame_out){
	int n = hrnet_trt_seg(hrnet_trt_engine, frame_in, frame_out);
}

void Trtyolosort::TrtUfld(cv::Mat &frame_in, cv::Mat &frame_out){
	int n = ufld_trt_det(ufld_trt_engine, frame_in, frame_out);
}
