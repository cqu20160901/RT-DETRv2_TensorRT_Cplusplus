#include "postprocess.hpp"
#include <algorithm>
#include <math.h>



/****** RtDetrV2 ****/
RtDetrV2::RtDetrV2()
{
}

RtDetrV2::~RtDetrV2()
{
}


int RtDetrV2::GetConvDetectionResult(std::vector<float *> &BlobPtr, std::vector<float> &DetectiontRects)
{
    float *BoxPtr = BlobPtr[0];
    float *LabelPtr = BlobPtr[1];
    float cx = 0, cy = 0, w = 0, h = 0;
    
    float Score = 0;
    int Index = 0;
    
    for (int i = 0; i < Maxnum; i++)
    {
        Score = 0;
        Index = 0;
 
        for (int c = 0; c < ClassNum; c ++)
        {
            if (c == 0)
            {
                Score = LabelPtr[i * ClassNum + c];
                Index = c;
            }
            else
            {
                if (Score < LabelPtr[i * ClassNum + c])
                {
                    Score = LabelPtr[i * ClassNum + c];
                    Index = c; 
                }
            }
        }

        if (Score > ObjectThresh) 
        {
            DetectiontRects.push_back(Index);
            DetectiontRects.push_back(Score);
            
            cx = BoxPtr[i * 4 + 0];
            cy = BoxPtr[i * 4 + 1];
            w = BoxPtr[i * 4 + 2];
            h = BoxPtr[i * 4 + 3];
            
            DetectiontRects.push_back(cx - 0.5 * w);
            DetectiontRects.push_back(cy - 0.5 * h);
            DetectiontRects.push_back(cx + 0.5 * w);
            DetectiontRects.push_back(cy + 0.5 * h);
        } 
    }
    return 0;
}
