#ifndef _POSTPROCESS_H_
#define _POSTPROCESS_H_

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <vector>

typedef signed char int8_t;
typedef unsigned int uint32_t;


typedef struct
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    int classId;
} DetectRect;

// rtdetrV2
class RtDetrV2
{
public:
    RtDetrV2();

    ~RtDetrV2();


    int GetConvDetectionResult(std::vector<float *> &BlobPtr, std::vector<float> &DetectiontRects);

private:

    const int ClassNum = 80;

    float ObjectThresh = 0.5;
    int Maxnum = 300;
};

#endif