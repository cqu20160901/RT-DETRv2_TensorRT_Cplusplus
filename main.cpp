#include "src/CNN.hpp"
#include <opencv2/opencv.hpp>
#include <thread>
#include <chrono>

int main()
{
    std::string OnnxFile = "/home/zq/Detect/rt_detr_v2/models/rtdetrv2_r18vd.onnx";
    std::string SaveTrtFilePath = "/home/zq/Detect/rt_detr_v2/models/rtdetrv2_r18vd.trt";
    cv::Mat SrcImage = cv::imread("/home/zq/Detect/rt_detr_v2/images/test.jpg");

    int img_width = SrcImage.cols;
    int img_height = SrcImage.rows;

    CNN CnnModel(OnnxFile, SaveTrtFilePath, 1, 3, 640, 640);

    auto t_start = std::chrono::high_resolution_clock::now();
    int Temp = 3000;
    for (int i = 0; i < Temp; i++)
    {
        CnnModel.Inference(SrcImage);
        // std::this_thread::sleep_for(std::chrono::milliseconds(90));
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "Info: " << Temp << " times infer and postprocess average cost: " << total_inf / float(Temp) - 0 << " ms." << std::endl;
    

    for (int i = 0; i < CnnModel.DetectiontRects_.size(); i += 6)
    {
        int classId = int(CnnModel.DetectiontRects_[i + 0]);
        float conf = CnnModel.DetectiontRects_[i + 1];
        int xmin = int(CnnModel.DetectiontRects_[i + 2] * float(img_width) + 0.5);
        int ymin = int(CnnModel.DetectiontRects_[i + 3] * float(img_height) + 0.5);
        int xmax = int(CnnModel.DetectiontRects_[i + 4] * float(img_width) + 0.5);
        int ymax = int(CnnModel.DetectiontRects_[i + 5] * float(img_height) + 0.5);

        char text1[256];
        sprintf(text1, "%d:%.2f", classId, conf);
        rectangle(SrcImage, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(255, 0, 0), 2);
        putText(SrcImage, text1, cv::Point(xmin, ymin + 15), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    }

    imwrite("/home/zq/Detect/rt_detr_v2/images/result.jpg", SrcImage);

    printf("== obj: %d \n", int(float(CnnModel.DetectiontRects_.size()) / 6.0));

    return 0;
}
