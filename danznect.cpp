/* 
 * See http://youtu.be/1W11wrZPE2E
 *
 * This is a Microsoft Kinect-using program for colorfully rendering
 * depth maps in near-real time. Excellent for dance floors - e.g. connect
 * a Kinect to a laptop, point the Kinect at a dance floor, and connect the 
 * laptop to a projector or large TV. This is a heavily, crappily modified 
 * version of an old version of the 'glview' demo included with the 
 * libfreenect library.
 * 
 * Major changes from the original demo:
 * A motion trail effect was created by adding multiple depth buffers. A
 * shifting color gradient system gives the output an 80's rave feeling. A 
 * simple in-painting algorithm fills in gaps or shadows in the depth map.
 * Also included is a median filter to smooth out noise. Added an aspect 
 * ratio correction to simplify widescreen setups. 
 * 
 * Currently only tested on Linux, but should be adaptable to other 
 * platforms.
 *
 * - Steve Busan, 2014
 */

/*
 * The original license header follows.
 */

/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2010 individual OpenKinect contributors. See the CONTRIB file
 * for details.
 *
 * This code is licensed to you under the terms of the Apache License, version
 * 2.0, or, at your option, the terms of the GNU General Public License,
 * version 2.0. See the APACHE20 and GPL2 files for the text of the licenses,
 * or the following URLs:
 * http://www.apache.org/licenses/LICENSE-2.0
 * http://www.gnu.org/licenses/gpl-2.0.txt
 *
 * If you redistribute this file in source form, modified or unmodified, you
 * may:
 *   1) Leave this header intact and distribute it under the same terms,
 *      accompanying it with the APACHE20 and GPL20 files, or
 *   2) Delete the Apache 2.0 clause and accompany it with the GPL2 file, or
 *   3) Delete the GPL v2 clause and accompany it with the APACHE20 file
 * In all cases you must keep the copyright notice intact and include a copy
 * of the CONTRIB file.
 *
 * Binary distributions must follow the binary distribution requirements of
 * either License.
 */

#include <libfreenect.hpp>

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <pthread.h>
#include <unistd.h>
#include "glWindowPos.h"


#if defined(__APPLE__)
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#ifndef MIN
#define MIN(a,b) ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#define MAX(a,b) ((a) < (b) ? (b) : (a))
#endif

using namespace std;

// global options
bool medianFilterSet = true;
bool inPaintSet = true;
bool gradientMotionSet = true;
unsigned int bufferWidth = 320;
unsigned int bufferHeight = 240;
#define maxBuffers 45
unsigned int currentBuffers = 45;
float brightnessFactor = 1;

// global output string
char outputCharBuf[1024] = {0};
char* outputString = 
"\n"
"        OO         O      O     O  OOO  O     O    OO    OO  OOO\n"
"        O   O    O O    OO  O       O   OO  O  O       O          O\n"
"        O    O  OOO   O O O     O     O O O  OO    O          O\n"
"        O   O  O     O  O  OO   O       O  OO  O       O          O\n"
"        OO    O       O O     O  OOO  O     O    OO    OO     O\n"
"\n\n                              Welcome to DANZNECT"
"\n\n                   Press H at any time for a list of hotkeys";
int hackyTimer = 500;
int lineSpacing = 15;

void setOutputString(char* s)
{
    outputString = s;
    hackyTimer = 100;
}


//define OpenGL variables
GLuint gl_depth_tex;
int g_argc;
char **g_argv;
int got_frames(0);
int window(0);
bool fullscreen = false;
int windowWidth = 640;
int windowHeight = 480;
int windowPad = 0;

//void *font = GLUT_BITMAP_TIMES_ROMAN_24;
void* font = GLUT_BITMAP_HELVETICA_18;
void* monoFont = GLUT_BITMAP_9_BY_15;

//define MyFreenectDevice and Mutex class
class Mutex {
    public:
        Mutex() {
            pthread_mutex_init( &m_mutex, NULL );
        }
        void lock() {
            pthread_mutex_lock( &m_mutex );
        }
        void unlock() {
            pthread_mutex_unlock( &m_mutex );
        }
    private:
        pthread_mutex_t m_mutex;
};

// Optimized median search on 9 values
#define PIX_SORT(a,b) { if ((a)>(b)) PIX_SWAP((a),(b)); }
#define PIX_SWAP(a,b) { uint16_t temp=(a);(a)=(b);(b)=temp; }
/*----------------------------------------------------------------------------
   Function :   opt_med9()
   In       :   pointer to an array of 9 pixelvalues
   Out      :   a pixelvalue
   Job      :   optimized search of the median of 9 pixelvalues
   Notice   :   in theory, cannot go faster without assumptions on the
                signal.
                Formula from:
                XILINX XCELL magazine, vol. 23 by John L. Smith

                The input array is modified in the process
                The result array is guaranteed to contain the median
                value
                in middle position, but other elements are NOT sorted.
 ---------------------------------------------------------------------------*/
uint16_t opt_med9(uint16_t* p){
    PIX_SORT(p[1], p[2]) ; PIX_SORT(p[4], p[5]) ; PIX_SORT(p[7], p[8]) ; 
    PIX_SORT(p[0], p[1]) ; PIX_SORT(p[3], p[4]) ; PIX_SORT(p[6], p[7]) ; 
    PIX_SORT(p[1], p[2]) ; PIX_SORT(p[4], p[5]) ; PIX_SORT(p[7], p[8]) ; 
    PIX_SORT(p[0], p[3]) ; PIX_SORT(p[5], p[8]) ; PIX_SORT(p[4], p[7]) ; 
    PIX_SORT(p[3], p[6]) ; PIX_SORT(p[1], p[4]) ; PIX_SORT(p[2], p[5]) ; 
    PIX_SORT(p[4], p[7]) ; PIX_SORT(p[4], p[2]) ; PIX_SORT(p[6], p[4]) ; 
    PIX_SORT(p[4], p[2]) ; return(p[4]) ;
}
#undef PIX_SWAP
#undef PIX_SORT

// simple median filter
void medianFilter(uint16_t* array, unsigned int bufferHeight, unsigned int bufferWidth){
    unsigned int i;
    uint16_t* depthList = (uint16_t*) malloc(9*sizeof(uint16_t));
    for(unsigned int y=1; y<bufferHeight-1; y++){
        for(unsigned int x=1; x<bufferWidth-1; x++){
            i = y*bufferWidth+x;
            // 3x3 kernel
            depthList[0] = array[i];          // center
            depthList[1] = array[i-1];        // left
            depthList[2] = array[i+1];        // right
            depthList[3] = array[i-1-bufferWidth];    // top-left
            depthList[4] = array[i-bufferWidth];      // top
            depthList[5] = array[i+1-bufferWidth];    // top-right
            depthList[6] = array[i-1+bufferWidth];    // bottom-left
            depthList[7] = array[i+bufferWidth];      // bottom
            depthList[8] = array[i+1+bufferWidth];    // bottom-right

            array[i] = opt_med9(depthList);
        }
    }
    free(depthList);
}

void inPaintHoriz(uint16_t* array, unsigned int bufferHeight, unsigned int bufferWidth){
    srand((unsigned)time(0));
    int ceiling = 10;
    int rnd;

    unsigned int index;
    unsigned int leftIndex;
    unsigned int leftDepth;
    unsigned int rightDepth;
    unsigned int rightIndex;
    unsigned int x;     
    unsigned int newDepth;
    unsigned int fillIndex;     
    for(unsigned int y=0; y<bufferHeight; y++){
        x = 1;
        while(x<bufferWidth-1){
            // locate start of hole
            index = y*bufferWidth+x;
            if(array[index]==2047){
                leftIndex = index;
                // find right side of hole
                while(array[index]==2047 and x<bufferWidth-1){
                    x++;
                    index = y*bufferWidth+x;
                }
                rightIndex = index - 1;
                leftDepth = array[leftIndex-1];
                rightDepth = array[rightIndex+1];
                if(leftDepth == 2047){ leftDepth = rightDepth;}
                if(rightDepth == 2047){ rightDepth = leftDepth;}
                newDepth = MAX(leftDepth,rightDepth);
                //newDepth = backBufferA[rightIndex+1];
                // fill hole with farthest bounding depth value
                int tempDepth;
                for(fillIndex=leftIndex; fillIndex<=rightIndex; fillIndex++){
                    tempDepth = newDepth;
                    rnd = rand() % ceiling - ceiling/2;
                    if(int(tempDepth)+rnd > 2047){ 
                        tempDepth = 2047; 
                    }else if(int(tempDepth)+rnd < 0){ 
                        tempDepth = 0; 
                    }else{
                        tempDepth = int(tempDepth) + rnd;
                    }
                    array[fillIndex] = uint16_t(tempDepth);
                }              
            }
            x++;
        }
    }
}

void inPaintVert(uint16_t* array, unsigned int bufferHeight, unsigned int bufferWidth){
    srand((unsigned)time(0));
    int ceiling = 10;
    int rnd;

    unsigned int index;
    unsigned int topIndex;
    unsigned int topDepth;
    unsigned int bottomDepth;
    unsigned int bottomIndex;
    unsigned int y;     
    unsigned int newDepth;
    unsigned int fillIndex;     
    for(unsigned int x=0; x<bufferWidth-4; x++){
        y = 1;
        while(y<bufferHeight-1){
            // locate start of hole
            index = y*bufferWidth+x;
            if(array[index]==2047){
                topIndex = index;
                // find right side of hole
                while(array[index]==2047 and y<bufferHeight-1){
                    y++;
                    index = y*bufferWidth+x;
                }
                bottomIndex = index - bufferWidth;
                topDepth = array[topIndex-bufferWidth];
                bottomDepth = array[bottomIndex+bufferWidth];
                if(topDepth == 2047){ topDepth = bottomDepth;}
                if(bottomDepth == 2047){ bottomDepth = topDepth;}
                newDepth = MAX(topDepth,bottomDepth);
                //newDepth = backBufferA[rightIndex+1];
                // fill hole with farthest bounding depth value
                int tempDepth;
                for(fillIndex=topIndex; fillIndex<=bottomIndex; fillIndex+=bufferWidth){
                    tempDepth = newDepth;                    
                    rnd = rand() % ceiling - ceiling/2;
                    if(int(tempDepth)+rnd > 2047){ 
                        tempDepth = 2047; 
                    }else if(int(tempDepth)+rnd < 0){ 
                        tempDepth = 0; 
                    }else{
                        tempDepth = int(tempDepth) + rnd;
                    }
                    array[fillIndex] = uint16_t(tempDepth);
                }             
            }
            y++;
        }
    }
}

void makeGradient(uint8_t gradient[], int numColors, int rArray[], int gArray[], int bArray[], int startDepth, int depthIncrement){ 
    for(int i=0; i<2048*3; i++){
        gradient[i] = (uint8_t)0;
    }
        
    int r1, g1, b1, r2, g2, b2;
    int depthStart, depthEnd;       
    depthEnd = startDepth;
    for(int arrayIndex=0; arrayIndex<numColors-1; arrayIndex++){
        depthStart = depthEnd;
        depthEnd = depthStart+depthIncrement;
        if(depthEnd>2047){
            printf("\r\n depthEnd out of range\n");
            fflush(stdout);
            break;
        }
        //printf("\r\n depthStart = %i, depthEnd=%i",depthStart,depthEnd);
        int depthRange = depthEnd-depthStart; 
        
        r1 = rArray[arrayIndex];
        g1 = gArray[arrayIndex];
        b1 = bArray[arrayIndex];

        r2 = rArray[arrayIndex+1];
        g2 = gArray[arrayIndex+1];
        b2 = bArray[arrayIndex+1];
        //printf("\r\n r1,g1,b1 = %i,%i,%i, r2,g2,b2 = %i,%i,%i\r\n",r1,g1,b1,r2,g2,b2);
        //fflush(stdout);
        double rSlope = (double)(r2-r1)/depthRange;
        double gSlope = (double)(g2-g1)/depthRange;
        double bSlope = (double)(b2-b1)/depthRange;
        for(int i=depthStart; i<depthEnd; i++){
            gradient[3*i  ] = (uint8_t) ((i-depthStart)*rSlope + r1);
            gradient[3*i+1] = (uint8_t) ((i-depthStart)*gSlope + g1);
            gradient[3*i+2] = (uint8_t) ((i-depthStart)*bSlope + b1);
            //printf("\r\n gradient[3*%i] = %i",i,gradient[3*i]);
        }
    }
}


class MyFreenectDevice : public Freenect::FreenectDevice {
    public:
        uint16_t* bufferPt[maxBuffers];
        uint8_t gradient[2048*3];
        uint8_t gradientB[2048*3];
        uint8_t gradientMod[2048*3];
        int contourInterval, contourSlope;
        int contourMin, contourMax;
        int contourOffset, contourOffsetMax, contourOffsetMin;
        int gradientOffset, gradientOffsetB;
        uint16_t* tempBufferA;
        uint16_t* tempBufferB;
        uint16_t* tempPointer;
        uint16_t* procDepth;

        MyFreenectDevice(freenect_context *_ctx, int _index) : Freenect::FreenectDevice(_ctx, _index),
        m_buffer_depth(bufferWidth*bufferHeight*3), 
        m_gamma(2048), 
        m_new_depth_frame(false) {
            srand((unsigned)time(0));
            for(unsigned int i=0; i<maxBuffers; i++){
                bufferPt[i] = (uint16_t*) malloc(bufferWidth*bufferHeight*sizeof(uint16_t));
            }

            int numColors = 17;
            int rArray[17] = {  0,  255,    0,    0,    0,  255,    0,    0,    0,  255,   0,  128,   0, 255,  0,    0,  0};
            int gArray[17] = {  0,    0,    0,  255,    0,  255,    0,  255,    0,  128,   0,  255,   0,   0,  0,  128,  0};
            int bArray[17] = {  0,  255,    0,  255,    0,    0,    0,  128,    0,    0,   0,    0,   0, 128,  0,  255,  0};                
            int startDepth = 0;
            int depthIncrement = 120;            
            makeGradient(gradient, numColors, rArray, gArray, bArray, startDepth, depthIncrement);

            numColors = 17;
            int rArrayB[17] = {  0,  128,    0,  128,    0,    0,    0,    0,    0,  128,   0,    0,   0,   0,  0,  255,  0};
            int gArrayB[17] = {  0,    0,    0,  128,    0,    0,    0,  128,    0,    0,   0,  128,   0,   0,  0,    0,  0};
            int bArrayB[17] = {  0,    0,    0,    0,    0,  128,    0,    0,    0,  128,   0,  128,   0, 255,  0,    0,  0};                
            startDepth = 0;
            depthIncrement = 77;
            makeGradient(gradientB, numColors, rArrayB, gArrayB, bArrayB, startDepth, depthIncrement);          

            gradientOffset = 0;
            gradientOffsetB = 0;
            
            tempBufferA = (uint16_t*) malloc(bufferWidth*bufferHeight*sizeof(uint16_t));
            tempBufferB = (uint16_t*) malloc(bufferWidth*bufferHeight*sizeof(uint16_t));
            tempPointer = NULL;
            procDepth = (uint16_t*) malloc(bufferWidth*bufferHeight*sizeof(uint16_t));

            for( unsigned int i = 0 ; i < 2048 ; i++) {
                float v = i/2048.0;
                v = pow(v, 3)* 6;
                m_gamma[i] = v*6*256;
            }
        }

        void VideoCallback(void* _rgb, uint32_t timestamp) {
            ;            
        };

        void DepthCallback(void* _depth, uint32_t timestamp) {
            m_depth_mutex.lock();
            uint16_t* depth = static_cast<uint16_t*>(_depth);
            
            // rotate buffers:
            tempPointer = bufferPt[maxBuffers-1];
            //printf("\r\n pointer = %p",tempPointer);
            for(int i = maxBuffers-1; i>=1; i--){
                bufferPt[i] = bufferPt[i-1];
            }
            bufferPt[0]=tempPointer;

            //downsample depth map into buffer
            unsigned int index;
            for(unsigned int x = 0; x<320; x++){
                for(unsigned int y=0; y<240; y++){
                    index = y*320+x;
                    bufferPt[0][index] = MIN(MIN(MIN(depth[y*2*640+x*2],depth[y*2*640+x*2+1]),depth[(y*2+1)*640+x*2+1]),depth[(y*2+1)*640+x*2]);                
                }
            }

            if(inPaintSet){ 
                //copy buffer
                for(unsigned int i = 0; i<bufferWidth*bufferHeight; i++){
                    tempBufferA[i]= bufferPt[0][i];
                    tempBufferB[i]= bufferPt[0][i];
                }
                                
                // fill in holes in depth map            
                inPaintVert(tempBufferA,bufferHeight,bufferWidth); 
                inPaintHoriz(tempBufferB,bufferHeight,bufferWidth);
                
                // put farthest value from each filling method into buffer A
                for(unsigned int i = 0; i<bufferWidth*bufferHeight; i++){
                    bufferPt[0][i] = MAX(tempBufferA[i],tempBufferB[i]);
                    //bufferPt[0][i] = tempBufferB[i];
                }              
            }              
                
            uint16_t tempDepth;
            for(unsigned int i = 0; i<bufferWidth*bufferHeight; i++){
                tempDepth = (uint16_t)2047;
                for(unsigned int j=0; j<currentBuffers; j+=6){
                    tempDepth = MIN(bufferPt[j][i],tempDepth);
                }
                procDepth[i] = tempDepth;
            }  

            // median filter
            if(medianFilterSet){ medianFilter(procDepth,bufferHeight,bufferWidth); }  

            // move color gradients (should add option to adjust speed)
            if(gradientMotionSet){
                gradientOffset += 5;//13;
                if(gradientOffset>2047){ gradientOffset = 0; }
                gradientOffsetB -= 3;//7;
                if(gradientOffsetB<0){ gradientOffsetB = 2047; }
            }

            // create combined gradient using both gradients at current offsets
            for(int i=0; i<2048; i++){
                int k = i+gradientOffset;
                int j = i+gradientOffsetB;
                // wraparound
                if(k>2047){ k=k-2048; }
                if(j>2047){ j=j-2048; }   
                gradientMod[3*i  ] = MAX( 0, gradient[3*j  ]-gradientB[3*k  ]);
                gradientMod[3*i+1] = MAX( 0, gradient[3*j+1]-gradientB[3*k+1] );
                gradientMod[3*i+2] = MAX( 0, gradient[3*j+2]-gradientB[3*k+2] );
            }
            
            // dim the colors by given factor
            for(int i=0; i<2048*3; i++){
                gradientMod[i] = (uint8_t)(gradientMod[i]/brightnessFactor);
            }

            // convert depth map values to gradient colors
            for( unsigned int x=2 ; x<bufferWidth-5; x++){
                for( unsigned int y=1; y<bufferHeight-1; y++) {
                    unsigned int i = y*bufferWidth + x;
                    unsigned int pval = (unsigned int)m_gamma[procDepth[i]];
                    m_buffer_depth[3*i+0] = gradientMod[3*pval+0];
                    m_buffer_depth[3*i+1] = gradientMod[3*pval+1];
                    m_buffer_depth[3*i+2] = gradientMod[3*pval+2];
                }
            }
            m_new_depth_frame = true;
            m_depth_mutex.unlock();
        }

        bool getDepth(vector<uint8_t> &buffer) {
            m_depth_mutex.lock();
            if(m_new_depth_frame) {
                buffer.swap(m_buffer_depth);
                m_new_depth_frame = false;
                m_depth_mutex.unlock();
                return true;
            } else {
                m_depth_mutex.unlock();
                return false;
            }
        }
        
    private:
        vector<uint8_t> m_buffer_depth;
        vector<uint16_t> m_gamma;
        Mutex m_depth_mutex;
        bool m_new_depth_frame;
};

//define libfreenect variables
Freenect::Freenect freenect;
MyFreenectDevice* device;
double freenect_angle(0);
freenect_video_format requested_format(FREENECT_VIDEO_RGB);

//define Kinect Device control elements
//glutKeyboardFunc Handler
void keyPressed(unsigned char key, int x, int y)
{
    switch(key)
    {
    case (char)27:
    case 'q':
    case 'Q':
        device->setLed(LED_RED);
        freenect_angle = 0;
        //glutReshapeWindow(640, 480);
        //fullscreen = false;
        //freenect_stop_depth(device);
	//freenect_stop_video(device);
	//freenect_close_device(device);
	//freenect_shutdown(f_ctx);
        glutDestroyWindow(window);
        break;
    case 'h':
    case 'H':
        outputString = "ESC,Q :   Quit\n"
                       "        H :   Display this message\n"
                       "        F :   Fullscreen ON/OFF\n"
                       "     + - :   Adjust number of motion-trail buffers\n"
                       "       [ ] :   Adjust screen brightness\n"
                       "     < > :   Adjust window width padding\n"
                       "       M :   Median filter ON/OFF\n"
                       "         I :   In-painting ON/OFF\n"
                       "       G :   Color gradient movement ON/OFF\n"
                       "\n space :   Hide text\n"
                       ;
        lineSpacing = 25;
        hackyTimer = 1000;
        break;
    case ' ':
        hackyTimer = 0;
        break;
    case 'm':
    case 'M':
        medianFilterSet = !medianFilterSet;
        if (medianFilterSet){
            setOutputString("Median filter is ON");
        }else{
            setOutputString("Median filter is OFF");
        }
        break;
    case 'i':
    case 'I':
        inPaintSet = !inPaintSet;
        if (inPaintSet){
            setOutputString("In-painting is ON");
        }else{
            setOutputString("In-painting is OFF");
        }
        break;
    case 'g':
    case 'G':
        gradientMotionSet = !gradientMotionSet;
        if (gradientMotionSet){
            setOutputString("Color gradient movement is ON");
        }else{
            setOutputString("Color gradient movement is OFF");
        }
        break;
    case 'f':
    case 'F':
        if(fullscreen){
            glutReshapeWindow(640, 480);
            fullscreen = false;
            windowWidth = glutGet(GLUT_WINDOW_WIDTH);
            windowHeight = glutGet(GLUT_WINDOW_HEIGHT);
            windowPad = 0;
            setOutputString("Fullscreen OFF");
        }else{
            glutFullScreen();
            fullscreen = true;
            windowWidth = glutGet(GLUT_WINDOW_WIDTH);
            windowHeight = glutGet(GLUT_WINDOW_HEIGHT);
            setOutputString("Fullscreen ON");
        }
        break;
    case '-':
    case '_':
        currentBuffers--;
        if (currentBuffers < 2 ){ currentBuffers = 2; }
        sprintf(outputCharBuf,"Number of buffers is now %i", currentBuffers);
        setOutputString(outputCharBuf);
        break;
    case '=':
    case '+':
        currentBuffers++;
        if (currentBuffers > maxBuffers ){ currentBuffers = maxBuffers; }
        sprintf(outputCharBuf,"Number of buffers is now %i", currentBuffers);
        setOutputString(outputCharBuf);
        break;
    case '[':
    case '{':
        brightnessFactor -= 0.2;
        if (brightnessFactor < 1 ){ brightnessFactor = 1; }
        sprintf(outputCharBuf,"Dimming factor is now %.2f", brightnessFactor);
        setOutputString(outputCharBuf);
        break;
    case ']':
    case '}':
        brightnessFactor += 0.2;
        if (brightnessFactor > 100 ){ brightnessFactor = 100; }
        sprintf(outputCharBuf,"Dimming factor is now %.2f", brightnessFactor);
        setOutputString(outputCharBuf);
        break;
    case '.':
    case '<':
        windowPad -= 10;
        if (windowPad < 0 ){ windowPad = 0; }
        sprintf(outputCharBuf,"Window width padding is now %i", windowPad);
        setOutputString(outputCharBuf);
        break;
    case ',':
    case '>':
        windowPad += 10;
        if (windowPad > windowWidth/2 ){ windowPad = windowWidth/2; }
        sprintf(outputCharBuf,"Window width padding is now %i", windowPad);
        setOutputString(outputCharBuf);
        break;
    default:
        break;
    }

}

// define OpenGL functions
void renderBitmapString(
		float x,
		float y,
		float z,
		void* font,
		char* string) {
    char* c;
    int yOffset = 0;
    int maxY = glutGet(GLUT_WINDOW_HEIGHT);
    int maxX = glutGet(GLUT_WINDOW_WIDTH);
    glWindowPos3f(x*maxX/640, maxY-y, z);
    for (c=string; *c != '\0'; c++) {
        if (*c=='\n'){ 
            yOffset -= lineSpacing; 
            glWindowPos3f(x*maxX/640, maxY-y+yOffset, z); 
        }
        glutBitmapCharacter(font, *c);
    }
}

void DrawGLScene()
{
    static std::vector<uint8_t> depth(bufferWidth*bufferHeight*4);

    device->updateState();

    device->getDepth(depth);

    got_frames = 0;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, gl_depth_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, 4, bufferWidth, bufferHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, &depth[0]);

    glBegin(GL_TRIANGLE_FAN);
    glColor4f(255.0f, 255.0f, 255.0f, 255.0f);
    glTexCoord2f(0, 0); glVertex3f(windowPad,0,-1);
    glTexCoord2f(1, 0); glVertex3f(640-windowPad,0,-1);
    glTexCoord2f(1, 1); glVertex3f(640-windowPad,480,-1);
    glTexCoord2f(0, 1); glVertex3f(windowPad,480,-1);
    glEnd();

    // countdown user notification timer, blank if 0
    hackyTimer--;
    if (hackyTimer < 0){ hackyTimer = 0; outputString = ""; lineSpacing=25;}    

    // render user notifications
    // disable lighting and texture so text color comes through
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    // first draw a jittered version of the string to create a dark outline around the text
    glColor3d(0.2, 0.0, 0.0);
    renderBitmapString( windowPad+40.0f,80.0f,-0.5f, font, outputString);
    renderBitmapString( windowPad+41.0f,81.0f,-0.5f, font, outputString);
    renderBitmapString( windowPad+41.0f,80.0f,-0.5f, font, outputString);
    renderBitmapString( windowPad+39.0f,79.0f,-0.5f, font, outputString);
    renderBitmapString( windowPad+39.0f,80.0f,-0.5f, font, outputString);
    renderBitmapString( windowPad+39.0f,81.0f,-0.5f, font, outputString);
    renderBitmapString( windowPad+40.0f,81.0f,-0.5f, font, outputString);
    renderBitmapString( windowPad+41.0f,79.0f,-0.5f, font, outputString);
    renderBitmapString( windowPad+40.0f,79.0f,-0.5f, font, outputString);
    // now draw main text on top
    glColor3d(0.4, 1.0, 0.7);
    renderBitmapString( windowPad+40.0f,80.0f,-0.5f, font, outputString);

    glutSwapBuffers();
}

void InitGL()
{
    //glutFullScreen();
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(1.0);
    glDepthFunc(GL_LESS);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glShadeModel(GL_SMOOTH);
    glGenTextures(1, &gl_depth_tex);
    glBindTexture(GL_TEXTURE_2D, gl_depth_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho (0, 640, 480, 0, 0.0f, 1.0f);
    glMatrixMode(GL_MODELVIEW);
}

void displayKinectData(MyFreenectDevice* device){
    //putenv( (char *) "__GL_SYNC_TO_VBLANK=1" );
    glutInit(&g_argc, g_argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_STENCIL | GLUT_DEPTH);
    glutInitWindowSize(640, 480);
    glutInitWindowPosition(30, 30);
    window = glutCreateWindow("DANZNECT");
    glutDisplayFunc(&DrawGLScene);
    glutIdleFunc(&DrawGLScene);
    glutKeyboardFunc(&keyPressed);
    InitGL();
    glutMainLoop();
}


//define main function
int main(int argc, char **argv) {
    device = &freenect.createDevice<MyFreenectDevice>(0);
    // do this a few times - for some reason fails on the first try sometimes
    device = &freenect.createDevice<MyFreenectDevice>(0);
    device = &freenect.createDevice<MyFreenectDevice>(0);

    // Start Kinect Device
    device->setTiltDegrees(0);
    device->startDepth();
    device->setLed(LED_GREEN);

    // start GL window
    displayKinectData(device);

    // Stop Kinect Device
    device->setLed(LED_RED);
    device->stopDepth();

    glutDestroyWindow(window);

    return 0;
}

