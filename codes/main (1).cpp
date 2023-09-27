/*
* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <opencv2/core/version.hpp>
#if CV_MAJOR_VERSION >= 3
#include <opencv2/imgcodecs.hpp>
#else
#include <opencv2/highgui/highgui.hpp>
#endif
#include <opencv2/imgproc/imgproc.hpp>
#include <vpi/OpenCVInterop.hpp>

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/MinMaxLoc.h>
#include <vpi/algo/TemplateMatching.h>
#include <vpi/algo/Convolution.h>
#include <cassert>
#include <cstring> // for memset
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <fstream>
#define CHECK_STATUS(STMT)                                    \
    do                                                        \
    {                                                         \
        VPIStatus status = (STMT);                            \
        if (status != VPI_SUCCESS)                            \
        {                                                     \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
            std::ostringstream ss;                            \
            ss << vpiStatusGetName(status) << ": " << buffer; \
            throw std::runtime_error(ss.str());               \
        }                                                     \
    } while (0);



using namespace std;

void getthershold(long long int*thr,int fps)
{
    *thr = ((float)(1/(2*(float)fps))*1000000000);
}  
int calculate(long long int t1,long long int t2,int frames,int fps)
{
    int flag = 1;
    long long int thr;
    getthershold(&thr, fps);
    if(abs(t1-t2)>thr)
        flag = 0;
    return flag;
}
cv::Mat sobelll(cv::Mat cvImage, VPIBackend backend)
{
    // OpenCV image that will be wrapped by a VPIImage.
    // Define it here so that it's destroyed *after* wrapper is destroyed
    // VPI objects that will be used
    VPIImage imagex = NULL;
    VPIImage imagey = NULL;
    VPIImage imageBGRx = NULL;
    VPIImage gradientx = NULL;
    VPIStream streamx = NULL;

    VPIImage imageBGRy = NULL;
    VPIImage gradienty = NULL;
    VPIStream streamy = NULL;
    int retval = 0;
    cv::Mat mag;
    try
    {

        // Create the stream for any backend.
        CHECK_STATUS(vpiStreamCreate(0, &streamx));
        CHECK_STATUS(vpiStreamCreate(0, &streamy));
        // We now wrap the loaded image into a VPIImage object to be used by VPI.
        CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvImage, 0, &imageBGRx));
        CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvImage, 0, &imageBGRy));

        // Now create the input image as a single unsigned 8-bit channel
        CHECK_STATUS(vpiImageCreate(cvImage.cols, cvImage.rows, VPI_IMAGE_FORMAT_U8, 0, &imagex));
        CHECK_STATUS(vpiImageCreate(cvImage.cols, cvImage.rows, VPI_IMAGE_FORMAT_U8, 0, &imagey));

        // Convert the loaded image to grayscale
        CHECK_STATUS(vpiSubmitConvertImageFormat(streamx, VPI_BACKEND_CUDA, imageBGRx, imagex, NULL));

        CHECK_STATUS(vpiSubmitConvertImageFormat(streamy, VPI_BACKEND_CUDA, imageBGRy, imagey, NULL));

        // Now create the output image, single unsigned 8-bit channel.
        CHECK_STATUS(vpiImageCreate(cvImage.cols, cvImage.rows, VPI_IMAGE_FORMAT_U8, 0, &gradientx));
        CHECK_STATUS(vpiImageCreate(cvImage.cols, cvImage.rows, VPI_IMAGE_FORMAT_U8, 0, &gradienty));

        // Define the convolution filter, a simple edge detector.
        float kernel1[3 * 3] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
        float kernel2[3 * 3] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
        // Submit it for processing passing the input image the result image that will store the gradient.
        CHECK_STATUS(vpiSubmitConvolution(streamx, backend, imagex, gradientx, kernel1, 3, 3, VPI_BORDER_ZERO));
        CHECK_STATUS(vpiSubmitConvolution(streamy, backend, imagey, gradienty, kernel2, 3, 3, VPI_BORDER_ZERO));

        // Wait until the algorithm finishes processing
        CHECK_STATUS(vpiStreamSync(streamx));
        CHECK_STATUS(vpiStreamSync(streamy));
        // Now let's retrieve the output image contents and output it to disk
        {
            // Lock output image to retrieve its data.
            VPIImageData outDatax;
            VPIImageData outDatay;

            CHECK_STATUS(vpiImageLockData(gradientx, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &outDatax));

            CHECK_STATUS(vpiImageLockData(gradienty, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &outDatay));

            // Returned data consists of host-accessible memory buffers in pitch-linear layout.
            assert(outDatax.bufferType == VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR);
            assert(outDatay.bufferType == VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR);
            const VPIImageBufferPitchLinear &outPitchx = outDatax.buffer.pitch;
            const VPIImageBufferPitchLinear &outPitchy = outDatay.buffer.pitch;

            cv::Mat cvOutx(outPitchx.planes[0].height, outPitchx.planes[0].width, CV_8UC1, outPitchx.planes[0].data,
                           outPitchx.planes[0].pitchBytes);
            cv::Mat cvOuty(outPitchy.planes[0].height, outPitchy.planes[0].width, CV_8UC1, outPitchy.planes[0].data,
                           outPitchy.planes[0].pitchBytes);

            // imwrite("edges_x" + strBackend + ".png", cvOutx);
            // imwrite("edges_y" + strBackend + ".png", cvOuty);

            cv::Mat img_l, img_r;

            CHECK_STATUS(vpiImageDataExportOpenCVMat(outDatax, &img_l));
            CHECK_STATUS(vpiImageDataExportOpenCVMat(outDatay, &img_r));

            img_l.convertTo(img_l, CV_32F);

            img_r.convertTo(img_r, CV_32F);

            cv::magnitude(img_l, img_r, mag);
            // imwrite("edges_mag" + strBackend + ".png", mag);
            //  Done handling output image, don't forget to unlock it.
            CHECK_STATUS(vpiImageUnlock(gradientx));
            CHECK_STATUS(vpiImageUnlock(gradienty));
        }
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        retval = 1;
    }
    if (retval == 1)
    {
        throw std::runtime_error("Can't open");
    }
    // Clean up

    // Make sure stream is synchronized before destroying the objects
    // that might still be in use.
    if (streamx != NULL)
    {
        vpiStreamSync(streamx);
    }
    if (streamy != NULL)
    {
        vpiStreamSync(streamy);
    }

    vpiImageDestroy(imagex);
    vpiImageDestroy(imageBGRx);
    vpiImageDestroy(gradientx);

    vpiStreamDestroy(streamx);

    vpiImageDestroy(imagey);
    vpiImageDestroy(imageBGRy);
    vpiImageDestroy(gradienty);

    vpiStreamDestroy(streamy);

    return mag;
}
int stereo_match(std::string& pay, std::string& lt, std::string& rt,const std::string& rtm1 = "none",const std::string& rtp1 = "none")
{
    // OpenCV image that will be wrapped by a VPIImage.
    // Define it here so that it's destroyed *after* wrapper is destroyed
    cv::Mat cvImagel;
    cv::Mat cvImager;
    cv::Mat cvImagerm1;
    cv::Mat cvImagerp1;

    cv::Mat cvImageU8l;
    cv::Mat cvImageU8r;
    cv::Mat cvImageU8rm1;
    cv::Mat cvImageU8rp1;

    cv::Mat cvTempl;

    VPIStream streamt = NULL;
    VPIStream streamtm1 = NULL;
    VPIStream streamtp1 = NULL;

    // VPI objects that will be used
    VPIImage inputr = NULL;
    VPIImage inputrm1 = NULL;
    VPIImage inputrp1 = NULL;

    VPIImage templ = NULL;
    int originX, originY, templWidth, templHeight;
    int flag = 0;
    int intru = 0;
    int outWidth, outHeight;
    VPIImage outputr = NULL;
    VPIImage outputrm1 = NULL;
    VPIImage outputrp1 = NULL;

    VPIImage outputScaledr = NULL;
    VPIImage outputScaledrm1 = NULL;
    VPIImage outputScaledrp1 = NULL;

    VPIImage outputU8l = NULL;
    VPIImage outputU8r = NULL;
    VPIImage outputU8rm1 = NULL;
    VPIImage outputU8rp1 = NULL;

    VPIArray minCoords = NULL;
    VPIArray maxCoords = NULL;

    VPIPayload payloadr = NULL;
    VPIPayload payloadrm1 = NULL;
    VPIPayload payloadrp1 = NULL;

    int retval = 0;

    try
    {
    	std::string strBackend = pay;
	if(rtm1 == "none" && rtp1 == "none")
	{	

		std::string strInputFileNamel = lt;
        	std::string strInputFileNamer = rt;
        	
        	
        	cvImagel = cv::imread(strInputFileNamel);
        	cvImager = cv::imread(strInputFileNamer);
        
    		cv::Mat blackImage(cvImagel.rows, cvImagel.cols, CV_8UC3, cv::Scalar(0));
    		cvImagerm1 = blackImage;
    		cvImagerp1 = blackImage;
    		if (cvImagel.empty())
        	{
            		throw std::runtime_error("Can't open '" + strInputFileNamel + "'");
        	}
        	if (cvImager.empty())
        	{
            		throw std::runtime_error("Can't open '" + strInputFileNamer + "'");
        	}
		intru = 1;
        }
        else
        {	
        
        std::string strInputFileNamel = lt;
        std::string strInputFileNamer = rt;
        std::string strInputFileNamerm1 = rtm1;
        std::string strInputFileNamerp1 = rtp1;

        // Load the input image
        cvImagel = cv::imread(strInputFileNamel);
        cvImager = cv::imread(strInputFileNamer);
        cvImagerm1 = cv::imread(strInputFileNamerm1);
        cvImagerp1 = cv::imread(strInputFileNamerp1);

        if (cvImagel.empty())
        {
            throw std::runtime_error("Can't open '" + strInputFileNamel + "'");
        }
        if (cvImager.empty())
        {
            throw std::runtime_error("Can't open '" + strInputFileNamer + "'");
        }
        if (cvImagerm1.empty())
        {
            throw std::runtime_error("Can't open '" + strInputFileNamerm1 + "'");
        }
        if (cvImagerp1.empty())
        {
            throw std::runtime_error("Can't open '" + strInputFileNamerp1 + "'");
        }        
       	

        }
        VPIBackend backend;

        if (strBackend == "cpu")
        {
            backend = VPI_BACKEND_CPU;
        }
        else if (strBackend == "cuda")
        {
            backend = VPI_BACKEND_CUDA;
        }
        else
        {
            throw std::runtime_error("Backend '" + strBackend + "' not recognized, it must be either cpu, cuda");
        }
        

        assert(cvImagel.type() == CV_8UC3);
        assert(cvImager.type() == CV_8UC3);
        assert(cvImagerm1.type() == CV_8UC3);
        assert(cvImagerp1.type() == CV_8UC3);	

        // convert image to gray scale

        cvtColor(cvImagel, cvImageU8l, cv::COLOR_BGR2GRAY);
        cvtColor(cvImager, cvImageU8r, cv::COLOR_BGR2GRAY);
        cvtColor(cvImagerm1, cvImageU8rm1, cv::COLOR_BGR2GRAY);
        cvtColor(cvImagerp1, cvImageU8rp1, cv::COLOR_BGR2GRAY);

        cvImageU8l = sobelll(cvImageU8l, backend);
        cvImageU8r = sobelll(cvImageU8r, backend);
        cvImageU8rm1 = sobelll(cvImageU8rm1, backend);
        cvImageU8rp1 = sobelll(cvImageU8rp1, backend);

        cvImageU8l.convertTo(cvImageU8l, CV_8UC3);
        cvImageU8r.convertTo(cvImageU8r, CV_8UC3);
        cvImageU8rm1.convertTo(cvImageU8rm1, CV_8UC3);
        cvImageU8rp1.convertTo(cvImageU8rp1, CV_8UC3);

        /*if (originX + templWidth > cvImage.cols || originY + templHeight > cvImage.rows)
        {
            throw std::runtime_error("Bounding box is out of range of input image size");
        }*/
        originX = cvImagel.cols / 4;
        originY = cvImagel.rows / 4;
        templWidth = cvImagel.cols / 2;
        templHeight = cvImagel.rows / 2;
        cv::Rect templROI(originX, originY, templWidth, templHeight);
        cv::Mat croppedRef(cvImageU8l, templROI);
        croppedRef.copyTo(cvTempl);

        // Now parse the backend

        // 1. Initialization phase ---------------------------------------

        CHECK_STATUS(vpiStreamCreate(backend, &streamt));
        CHECK_STATUS(vpiStreamCreate(backend, &streamtm1));
        CHECK_STATUS(vpiStreamCreate(backend, &streamtp1));

        // We now wrap the loaded image into a VPIImage object to be used by VPI.
        // VPI won't make a copy of it, so the original
        // image must be in scope at all times.
        CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvImageU8r, 0, &inputr));
        CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvImageU8rm1, 0, &inputrm1));
        CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvImageU8rp1, 0, &inputrp1));

        // Create template iamge
        CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvTempl, 0, &templ));

        // Now create the output image.
        outWidth = cvImagel.cols - templWidth + 1;
        outHeight = cvImagel.rows - templHeight + 1;

        CHECK_STATUS(vpiImageCreate(outWidth, outHeight, VPI_IMAGE_FORMAT_F32, 0, &outputr));
        CHECK_STATUS(vpiImageCreate(outWidth, outHeight, VPI_IMAGE_FORMAT_F32, 0, &outputrm1));
        CHECK_STATUS(vpiImageCreate(outWidth, outHeight, VPI_IMAGE_FORMAT_F32, 0, &outputrp1));

        CHECK_STATUS(vpiImageCreate(outWidth, outHeight, VPI_IMAGE_FORMAT_F32, 0, &outputScaledr));
        CHECK_STATUS(vpiImageCreate(outWidth, outHeight, VPI_IMAGE_FORMAT_F32, 0, &outputScaledrm1));
        CHECK_STATUS(vpiImageCreate(outWidth, outHeight, VPI_IMAGE_FORMAT_F32, 0, &outputScaledrp1));

        CHECK_STATUS(vpiImageCreate(outWidth, outHeight, VPI_IMAGE_FORMAT_U8, 0, &outputU8r));
        CHECK_STATUS(vpiImageCreate(outWidth, outHeight, VPI_IMAGE_FORMAT_U8, 0, &outputU8rm1));
        CHECK_STATUS(vpiImageCreate(outWidth, outHeight, VPI_IMAGE_FORMAT_U8, 0, &outputU8rp1));

        // Create payload
        CHECK_STATUS(vpiCreateTemplateMatching(backend, cvImager.cols, cvImager.rows, &payloadr));
        CHECK_STATUS(vpiCreateTemplateMatching(backend, cvImagerm1.cols, cvImagerm1.rows, &payloadrm1));
        CHECK_STATUS(vpiCreateTemplateMatching(backend, cvImagerp1.cols, cvImagerp1.rows, &payloadrp1));

        // CHECK_STATUS(vpiCreateMinMaxLoc(backend, cvImage.cols, cvImage.rows, VPI_IMAGE_FORMAT_F32, &payloadMinMax));

        // 2. Computation phase ---------------------------------------

        // Set source image
        CHECK_STATUS(vpiTemplateMatchingSetSourceImage(streamt, backend, payloadr, inputr));
        CHECK_STATUS(vpiTemplateMatchingSetSourceImage(streamtm1, backend, payloadrm1, inputrm1));
        CHECK_STATUS(vpiTemplateMatchingSetSourceImage(streamtp1, backend, payloadrp1, inputrp1));

        // Set source image
        CHECK_STATUS(vpiTemplateMatchingSetTemplateImage(streamt, backend, payloadr, templ, NULL));
        CHECK_STATUS(vpiTemplateMatchingSetTemplateImage(streamtm1, backend, payloadrm1, templ, NULL));
        CHECK_STATUS(vpiTemplateMatchingSetTemplateImage(streamtp1, backend, payloadrp1, templ, NULL));

        // Submit
        CHECK_STATUS(vpiSubmitTemplateMatching(streamt, backend, payloadr, outputr, VPI_TEMPLATE_MATCHING_NCC));
        CHECK_STATUS(vpiSubmitTemplateMatching(streamtm1, backend, payloadrm1, outputrm1, VPI_TEMPLATE_MATCHING_NCC));
        CHECK_STATUS(vpiSubmitTemplateMatching(streamtp1, backend, payloadrp1, outputrp1, VPI_TEMPLATE_MATCHING_NCC));

        // CHECK_STATUS(vpiSubmitMinMaxLoc(stream, backend, payloadMinMax, output, minCoords, maxCoords));

        // Convert output from F32 to U8
        VPIConvertImageFormatParams params;
        CHECK_STATUS(vpiInitConvertImageFormatParams(&params));
        params.scale = 255;

        CHECK_STATUS(vpiSubmitConvertImageFormat(streamt, backend, outputr, outputScaledr, &params));
        CHECK_STATUS(vpiSubmitConvertImageFormat(streamtm1, backend, outputrm1, outputScaledrm1, &params));
        CHECK_STATUS(vpiSubmitConvertImageFormat(streamtp1, backend, outputrp1, outputScaledrp1, &params));

        CHECK_STATUS(vpiSubmitConvertImageFormat(streamt, backend, outputScaledr, outputU8r, NULL));
        CHECK_STATUS(vpiSubmitConvertImageFormat(streamtm1, backend, outputScaledrm1, outputU8rm1, NULL));
        CHECK_STATUS(vpiSubmitConvertImageFormat(streamtp1, backend, outputScaledrp1, outputU8rp1, NULL));

        // Wait until the algorithm finishes processing
        CHECK_STATUS(vpiStreamSync(streamt));
        CHECK_STATUS(vpiStreamSync(streamtm1));
        CHECK_STATUS(vpiStreamSync(streamtp1));

        // Now let's retrieve the output image contents and output it to disk
        {
            VPIImageData outDatar;
            CHECK_STATUS(vpiImageLockData(outputU8r, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &outDatar));

            VPIImageData outDatarm1;
            CHECK_STATUS(vpiImageLockData(outputU8rm1, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &outDatarm1));

            VPIImageData outDatarp1;
            CHECK_STATUS(vpiImageLockData(outputU8rp1, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &outDatarp1));

            // Returned data consists of host-accessible memory buffers in pitch-linear layout.
            assert(outDatar.bufferType == VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR);
            assert(outDatarm1.bufferType == VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR);
            assert(outDatarp1.bufferType == VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR);
            cv::Mat cvOutr;
            CHECK_STATUS(vpiImageDataExportOpenCVMat(outDatar, &cvOutr));

            cv::Mat cvOutrm1;
            CHECK_STATUS(vpiImageDataExportOpenCVMat(outDatarm1, &cvOutrm1));

            cv::Mat cvOutrp1;
            CHECK_STATUS(vpiImageDataExportOpenCVMat(outDatarp1, &cvOutrp1));

            double min_val, max_val;
            float c1, c2, c3;
            cv::Point min_loc, max_loc;
            cv::minMaxLoc(cvOutr, &min_val, &max_val, &min_loc, &max_loc);

            // std::cout<<min_val<<" "<<max_val<<" "<<min_loc<<" "<<max_loc<<std::endl;

            c1 =(max_val / 255);
            //printf("\nc1: %f\n", c1);
            std::cout << "Correlation co-eff for left_t to right_t: " << c1 << std::endl;

            cv::minMaxLoc(cvOutrm1, &min_val, &max_val, &min_loc, &max_loc);

            // std::cout<<min_val<<" "<<max_val<<" "<<min_loc<<"Corr value"<<c1<<std::endl;

            c2 =(max_val / 255);
            std::cout << "Correlation co-eff for left_t to right_t-1: " << c2 << std::endl;

            cv::minMaxLoc(cvOutrp1, &min_val, &max_val, &min_loc, &max_loc);

            // std::cout<<min_val<<" "<<max_val<<" "<<min_loc<<" "<<max_loc<<"Corr value"<<c2<<std::endl;

            c3 =(max_val / 255);

            std::cout << "Correlation co-eff for left_t to right_t+1: " << c3 << std::endl;
            // std::cout<<min_val<<" "<<max_val<<" "<<min_loc<<" "<<max_loc<<"Corr value"<<c3<<std::endl;

            float c = std::max(c1, c2);
            c = std::max(c, c3);
	    
	    if(intru == 1)
	    {
	    	 if (c >= 0.40000)
            {
                //printf("\nSync\n");
                flag = 1;
            }
            else
            {
                //printf("\nAsync\n");
                flag = 0;
            }
	    }
	    else
	    {
            if (c == c1)
            {
                //printf("\nSync\n");
                flag = 1;
            }
            else
            {
                //printf("\nAsync\n");
                flag = 0;
            }
            }
        }
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        retval = 1;
    }

    // Clean up

    // Make sure stream is synchronized before destroying the objects
    // that might still be in use.
    vpiStreamSync(streamt);
    vpiStreamSync(streamtm1);
    vpiStreamSync(streamtp1);

    vpiPayloadDestroy(payloadr);
    vpiPayloadDestroy(payloadrm1);
    vpiPayloadDestroy(payloadrp1);

    vpiImageDestroy(inputr);
    vpiImageDestroy(templ);
    vpiImageDestroy(outputr);
    vpiImageDestroy(outputU8r);
    vpiStreamDestroy(streamt);

    vpiImageDestroy(inputrm1);
    vpiImageDestroy(outputrm1);
    vpiImageDestroy(outputU8rm1);
    vpiStreamDestroy(streamtm1);

    vpiImageDestroy(inputrp1);
    vpiImageDestroy(outputrp1);
    vpiImageDestroy(outputU8rp1);
    vpiStreamDestroy(streamtp1);

    return flag;
}
int main()
{   
    ifstream inFile("tsc_log.txt");
    if (!inFile) {
        cerr << "Unable to open file!" << std::endl;
        return 1;
    }
    long long int interval,t1=1,t2=2,x=-1;
    string line1,line2,f1,f2;
    int flag1 =0 ,flag2= 0,temp = 0 ;
    int frames = 1;
    string exen = ".png";
    string pay = "cuda";
   while(std::getline(inFile, line1)&&std::getline(inFile, line2))
   {
        //cout<<line1<<endl;
        //cout<<line2<<endl;
        
        string lt,rt,rtm1,rtp1,temp;
        stringstream ss1,ss2;
        for (char c : line1) {
            if (isdigit(c)) {
                ss1 << c;
            }
        }
        line1 = ss1.str();
        for (char c : line2) {
            if (isdigit(c)) {
                ss2 << c;
            }
        }
        line2 = ss2.str();

        f1 = line1.substr(0,(line1.length() - 14));
        f2 = line2.substr(0,(line2.length() - 14));
        //cout<<f1<<" "<<f2<<endl;
        if(stol(f1) == stol(f2))
        {
            //cout<<f1<<f2;
            frames = stoi(f1);
            
            line1 = line1.substr((line1.length() - 14),14);
            line2 = line2.substr((line2.length() - 14),14);
            t1 = stol(line1);
            t2 = stol(line2);
            //cout<<t1<<endl<<t2<<endl;
            flag1 = calculate(t1,t2,frames,30);
            lt = "/media/nvidia/JAMES/new/left/img_left";
            lt = lt + f1 + exen;
            rt = "/media/nvidia/JAMES/new/right/img_right";
            rt = rt + f2 + exen;
            x = stol(f1);
            x--;
            temp = to_string(x);
            rtm1 = "/media/nvidia/JAMES/new/right/img_right";
            rtm1 = rtm1 + temp + exen;
            x = stol(f1);
            x++;
            temp = to_string(x);
            rtp1 = "/media/nvidia/JAMES/new/right/img_right";
            rtp1 = rtp1 + temp + exen;
            if(frames == 1)
            	flag2 = stereo_match(pay, lt,rt);
            else
            	flag2 = stereo_match(pay,lt,rt,rtm1,rtp1);
        //cout<<"interval"<<endl;
        //cout<<"frames"<<endl;
        //cout<<"Timestamps - "<<"Left: "<<line1<<"  "<<"Right: "<<line2<<endl;
        //cout<<flag1<<"  "<<flag2<<endl;
        //cout<<frames<<endl;
        if(flag1 == 1 && flag2 == 1)
            cout<<"Frame: "<<frames<<"  -->  "<<" Sync "<<endl;
        else 
            cout<<"Frame: "<<frames<<"  -->  "<<" Async "<<endl;
         }
	frames++;
	x = 0;
   }
   
    inFile.close();

    return 0;
}
