/* 
 * See http://youtu.be/zerMfuB1WrQ
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

// TODO: store default presets in an external file, and let user define new ones with a hotkey
// TODO: add a mirror option (flip the screen horizontally)
// TODO: add fog that works with bright outlines (i.e. hide back wall if wanted)
// FIXME: move all buffer allocations to main() (although compiler optimization
//        might already take care of this)
// TODO: build with debug symbols, perform random stack stops to find slow parts of code
// TODO: add event layer and sequencer, and feed keyboard events into event sequencer
//		 - instead of changing global params directly, put state instance in queue with 
//         current time, then "activate" that state when past that time
//		 - need to figure out how to set up cumulative timer, not just frame-to-frame
//         elapsed time delta
//       - fills could then be a list of state instances and offset times, added to event
//         queue
//       - figure out what to do if states put in event queue out of order of their intended
//         times of activation
// TODO: try using Mat_<_Tp> instead of Mat to simplify code
// FIXME: replace one-dimensional vector buffers holding 3-channels with explicit 
//        array of 3-element arrays (although this might be slower?)
// TODO: option to to save video from this program
// TODO: other gradient interpolation options (e.g. thresholded rather than interpolated)

// TODO: use shift-numbers held down to trigger temporary "rolls" - will require class to convert between temp and current options 

// TODO: option to hide notifications

#include "libfreenect_cpp11.hpp"
#include "opencv/cv.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/photo/photo.hpp"

#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <cmath>
#include <pthread.h>
#include <unistd.h>
#include <chrono>
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


// FIXME: implement an event "queue". not really a queue.
//        - key by time submitted
//        - vals: preset params object, time to activate, time to deactivate (0 to indicate never deactivate)


class Timer // from https://gist.github.com/gongzhitaao/7062087
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    // FIXME: not sure of the units here
    double elapsed() const { 
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

Timer fps_timer = Timer();
Timer gradient_timer = Timer();
Timer event_timer = Timer();



// global options
// FIXME: move to a params object so presets can be encapsulated
bool notificationsOn = true;
bool mirrorSet = true;
bool medianFilterSet = true;
bool inPaintSet = true;
bool gradientMotionSet = true;
int colorScheme = 0;
bool posterizeSet = false;
bool circularSet = false;
bool showFPSset = false;
unsigned int bufferWidth = 320;
unsigned int bufferHeight = 240;
#define maxBuffers 45
unsigned int currentBuffers = 45;
float brightnessFactor = 1;
//float speedFactor = 1.0;
//std::vector<float> speedFactors = {0.0, 0.01, 0.06, 0.1, 0.5, 1.0, 2.0, 3.0};
std::vector<float> speedFactors = {0.0, 0.03, 0.08, 0.1, 0.5, 2.0, 3.0, 10.0};
int speedFactorIndex = 4;

bool fogSet = false;
std::vector<float> fogStarts = {0.  ,  0.02,  0.04,  0.06,  0.08,  0.1 ,  0.12,  0.14,  0.16,
        0.18,  0.2 ,  0.22,  0.24,  0.26,  0.28,  0.3 ,  0.32,  0.34,
        0.36,  0.38,  0.4 ,  0.42,  0.44,  0.46,  0.48,  0.5 ,  0.52,
        0.54,  0.56,  0.58,  0.6 ,  0.62,  0.64,  0.66,  0.68,  0.7 ,
        0.72,  0.74,  0.76,  0.78,  0.8 ,  0.82,  0.84,  0.86,  0.88,
        0.9 ,  0.92,  0.94,  0.96,  0.98,  1.  };
float fogDepth = 0.05;
int fogIndex = 20;
float fogStart = fogStarts[fogIndex];
float fogEnd = fogStart + fogDepth;

std::vector<std::string> outlines = {"none", "rainbow", "bright", "dark"};
int outline_index = 0;

vector<vector<uint8_t>> gradient_a;
vector<vector<uint8_t>> gradient_b;
int gradient_index = 0;

uint8_t gradientMod[2048*3] = {};
std::vector<float> gradient_periods = {0.01, 0.011, 0.025, 0.05, 0.1, 0.2, 0.333333, 0.6, 0.8, 1.0, 2.0, 5.0};
int gradient_period_index = 6;
float gradientOffsetA = 0.0;
float gradientOffsetB = 0.0;

// global output string
char outputCharBuf[1024] = {0};
std::string outputString = 
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

void setOutputString(string s)
{
    outputString = s.c_str();
    hackyTimer = 100;
}

class Params
{
public:
    int gradient_index;
    int speedFactorIndex;
    int gradient_period_index;
    bool posterizeSet;
    int outline_index;
    int currentBuffers;
    Params() {}
    Params(int gi, int sfi, int gpi, bool ps, int oi, int cb) {
        gradient_index = gi;
        speedFactorIndex = sfi;
        gradient_period_index = gpi;
        posterizeSet = ps;
        outline_index = oi;
        currentBuffers = cb;  
    }
    void set(int gi, int sfi, int gpi, bool ps, int oi, int cb) {
        gradient_index = gi;
        speedFactorIndex = sfi;
        gradient_period_index = gpi;
        posterizeSet = ps;
        outline_index = oi;
        currentBuffers = cb;  
    }
    void set_global() {
        ::gradient_index = this->gradient_index;
        ::speedFactorIndex = this->speedFactorIndex;
        ::gradient_period_index = this->gradient_period_index;
        ::posterizeSet = this->posterizeSet;
        ::outline_index = this->outline_index;
        ::currentBuffers = this->currentBuffers;  
    }
    void get_global() {
        this->gradient_index = ::gradient_index;
        this->speedFactorIndex = ::speedFactorIndex;
        this->gradient_period_index = ::gradient_period_index;
        this->posterizeSet = ::posterizeSet;
        this->outline_index = ::outline_index;
        this->currentBuffers = ::currentBuffers;          
    }
};


map<string,string> shift_nums = {
{"!","1"},
{"@","2"},
{"#","3"},
{"$","4"},
{"%","5"},
{"^","6"},
{"&","7"},
{"*","8"},
};

// define presets
map<string,Params> presets;
void init_presets() {
    presets["1"] = Params(1,3, 4, true, 3, 35);
    presets["2"] = Params(2,2,0,false,2,13);
    presets["3"] = Params(4,5,4,false,2,20);
    presets["4"] = Params(0,1,5,false,3,45);
    presets["5"] = Params(1,2,2,false,2,35);
    presets["6"] = Params(0,1,0,false,0,35);
    presets["7"] = Params(4,1,2,false,1,20);
    presets["8"] = Params(0,3,4,false,2,20);
}

struct Event {
    string preset;
    bool running;
    double duration; // in seconds
};

Params stored_params;
queue<Event> events;

// set stored params on addition to event queue if len 0
// set global params to stored params if len 0

void poll_event_queue() {
    if (events.size() > 0) {
        if (not events.front().running) { 
            event_timer.reset();
            events.front().running = true;
        } else {
            if (event_timer.elapsed() > events.front().duration) {
                events.pop();
                // restore previous params
                stored_params.set_global();
                setOutputString("end of event");
            }
        }
    }
}

void add_event(string preset, double duration) {
    // store current params for restore
    if (events.size() == 0) {
        stored_params.get_global();
    }
    events.push(Event{preset, duration});
    setOutputString("Pushed event for preset "+preset);
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

vector<uint8_t> makeGradient(vector<int> r, 
                             vector<int> g,
                             vector<int> b,
                             float period,
                             vector<int> bg_color={0,0,0}){
    bool debug = false; 
    vector<uint8_t> grad(2048*3);
    int r1, g1, b1, r2, g2, b2;
    for(int i=0; i<2048; i++){
        // locate the nearest left and right colors, then interpolate
        double grad_index = (i/2048.0f)/period*r.size();
        //float in_range_grad_index = r.size()+int(grad_index)%r.size();
        int in_range_grad_index = round(grad_index);
        if (i>1986){
            if(debug){
                cout << "i = " << i << endl;
                cout << "grad_index = " << grad_index << endl;
                cout << "in_range_grad_index = " << in_range_grad_index << endl;                
            }
        }
        // fill indices outside range with bg color to match previous behavior (may change in future)
        if (in_range_grad_index>=r.size()){
            grad[3*i  ] = (uint8_t) bg_color[0];
            grad[3*i+1] = (uint8_t) bg_color[1];
            grad[3*i+2] = (uint8_t) bg_color[2];
            continue;
        }
        int left = floor(grad_index);
        int in_range_left = (r.size()+left)%r.size();
        int right = ceil(grad_index);
        int in_range_right = (r.size()+right)%r.size();

        double x = grad_index - in_range_left;

        r1 = r[in_range_left];
        g1 = g[in_range_left];
        b1 = b[in_range_left];

        r2 = r[in_range_right];
        g2 = g[in_range_right];
        b2 = b[in_range_right];

        double r_slope = (double)(r2-r1)/(in_range_right-in_range_left);
        double g_slope = (double)(g2-g1)/(in_range_right-in_range_left);
        double b_slope = (double)(b2-b1)/(in_range_right-in_range_left);
        if (in_range_left==in_range_right){
            grad[3*i  ] = (uint8_t) r1;
            grad[3*i+1] = (uint8_t) g1;
            grad[3*i+2] = (uint8_t) b1;
        } else {
            grad[3*i  ] = (uint8_t) round(x*r_slope + r1);
            grad[3*i+1] = (uint8_t) round(x*g_slope + g1);
            grad[3*i+2] = (uint8_t) round(x*b_slope + b1);
        }
    }

    return grad;
}


void print_gradient(vector<uint8_t> grad){
    int r,g,b;
    for(int i=0; i<grad.size(); i+=3){
        r = grad[i];
        g = grad[i+1];
        b = grad[i+2];
        cout << i/3 << ": ( " << r << ", " << g << ", " << b << " )" << endl;
    }
}

void initGradients(){
    int start_depth, depth_increment;
    vector<int> r, g, b;

    // rainbow

    r = { 0,  255,    0,    0,    0,  255,    0,    0,    0,  255,   0,  128,   0, 255,  0,    0,  0};
    g = { 0,    0,    0,  255,    0,  255,    0,  255,    0,  128,   0,  255,   0,   0,  0,  128,  0};
    b = { 0,  255,    0,  255,    0,    0,    0,  128,    0,    0,   0,    0,   0, 128,  0,  255,  0}; 
    gradient_a.push_back(makeGradient(r, g, b, 1.0));

    r = {  0,  128,    0,  128,    0,    0,    0,    0,    0,  128,   0,    0,   0,   0,  0,  255,  0};
    g = {  0,    0,    0,  128,    0,    0,    0,  128,    0,    0,   0,  128,   0,   0,  0,    0,  0};
    b = {  0,    0,    0,    0,    0,  128,    0,    0,    0,  128,   0,  128,   0, 255,  0,    0,  0};                
    gradient_b.push_back(makeGradient(r, g, b, 0.641));          

    // bright rainbow

    r = { 255,  255,   255,    0,  255,  255,  255,    0,  255,  255, 255,  128, 255, 255, 255,    0,  255};
    g = { 255,    0,   255,  255,  255,  255,  255,  255,  255,  128, 255,  255, 255,   0, 255,  128,  255};
    b = { 255,  255,   255,  255,  255,    0,  255,  128,  255,    0, 255,    0, 255, 128, 255,  255,  255}; 
    gradient_a.push_back(makeGradient(r, g, b, 1.0, {255,255,255}));
    //print_gradient(gradient_a[1]);

    r = {  0,  128,    0,  128,    0,    0,    0,    0,    0,  128,   0,    0,   0,   0,  0,  255,  0};
    g = {  0,    0,    0,  128,    0,    0,    0,  128,    0,    0,   0,  128,   0,   0,  0,    0,  0};
    b = {  0,    0,    0,    0,    0,  128,    0,    0,    0,  128,   0,  128,   0, 255,  0,    0,  0};                
    gradient_b.push_back(makeGradient(r, g, b, 0.641)); 

    // red-orange green (predator-esque w/outlines)

    r = {  0,  255,    0,  0};
    g = {  0,   75,  120,  0};
    b = {  0,   50,    0,  0};                  
    gradient_a.push_back(makeGradient(r, g, b, 1.0));

    r = {  0,  128,    0};
    g = {  0,    0,    0};
    b = {  0,  255,    0};                
    gradient_b.push_back(makeGradient(r, g, b, 0.93));         

    // red-orange green alt

    r = {  0,  255,  255,   0,  0};
    g = {  0,   75,   75, 120,  0};
    b = {  0,   50,   50,   0,  0};                  
    gradient_a.push_back(makeGradient(r, g, b, 1.0));

    r = {  0,  128, 128,  0};
    g = {  0,    0,   0,  0};
    b = {  0,  255, 255,  0};                
    gradient_b.push_back(makeGradient(r, g, b, 0.93));  

    // black and white

    r = {  0,  255,  255,  255, 0, 0, 0, 0};
    g = {  0,  255,  255,  255, 0, 0, 0, 0};
    b = {  0,  255,  255,  255, 0, 0, 0, 0};                  
    gradient_a.push_back(makeGradient(r, g, b, 1.0));

    r = {  0,    0,    0};
    g = {  0,    0,    0};
    b = {  0,    0,    0};                
    gradient_b.push_back(makeGradient(r, g, b, 1.0));         


}


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



class MyFreenectDevice : public Freenect::FreenectDevice {
    public:
        uint16_t* bufferPt[maxBuffers];
        int contourInterval, contourSlope;
        int contourMin, contourMax;
        int contourOffset, contourOffsetMax, contourOffsetMin;
        uint16_t* tempBufferA;
        uint16_t* tempBufferB;
        uint16_t* tempPointer;
        uint16_t* procDepth;
        //std::vector<uint16_t        

        // FIXME: update these var names, m_buffer_depth holds RGB vals
        MyFreenectDevice(freenect_context *_ctx, int _index) : Freenect::FreenectDevice(_ctx, _index),
              m_buffer_depth(bufferWidth*bufferHeight*3), 
              m_gamma(2048), 
              m_new_depth_frame(false) {

            srand((unsigned)time(0));
            for(unsigned int i=0; i<maxBuffers; i++){
                bufferPt[i] = (uint16_t*) malloc(bufferWidth*bufferHeight*sizeof(uint16_t));
            }
            
            tempBufferA = (uint16_t*) malloc(bufferWidth*bufferHeight*sizeof(uint16_t));
            tempBufferB = (uint16_t*) malloc(bufferWidth*bufferHeight*sizeof(uint16_t));
            tempPointer = NULL;
            procDepth = (uint16_t*) malloc(bufferWidth*bufferHeight*sizeof(uint16_t));

            for( unsigned int i = 0 ; i < 2048 ; i++) {
                float v = i/2048.0;
                //v = pow(v, 3)* 6; // This was probably a bug?
                v = pow(v,3)*8; // pow(v,3)*6 was original - exaggerates depths in foreground and limits upper end to ~1500?
                m_gamma[i] = v*256;
                if(m_gamma[i]>2047){
                    std::cout << "m_gamma[i] > 2047" << std::endl;
                }
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
                    bufferPt[0][index] = MIN(MIN(MIN(depth[y*2*640+x*2],
                                                     depth[y*2*640+x*2+1]),
                                                     depth[(y*2+1)*640+x*2+1]),
                                                     depth[(y*2+1)*640+x*2]);                
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

            if(outlines[outline_index] == "rainbow"){
                cv::Mat cv_buffer(bufferHeight, bufferWidth, CV_16UC1 );
                memcpy( cv_buffer.data, bufferPt[0], bufferWidth*bufferHeight*sizeof(uint16_t) );
                cv_buffer.convertTo(cv_buffer, CV_8UC1, 256/2048.0);

                // glitchy cool outline effect
                cv::GaussianBlur( cv_buffer, cv_buffer, cv::Size(5,5), 0, 0, cv::BORDER_REFLECT );
                //cv::GaussianBlur( cv_buffer, cv_buffer, cv::Size(5,5), 0, 0, cv::BORDER_REFLECT );

                cv_buffer.convertTo(cv_buffer, CV_16UC1, 2048/256.0);
                memcpy( bufferPt[0], cv_buffer.data, bufferWidth*bufferHeight*sizeof(uint16_t) );
            }

            // collapse depth buffers to nearest points
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

            // TODO: erode/dilate options

            if(posterizeSet){
                cv::Mat cv_buffer(bufferHeight, bufferWidth, CV_16UC1 );
                memcpy( cv_buffer.data, procDepth, bufferWidth*bufferHeight*sizeof(uint16_t) );
                cv_buffer.convertTo(cv_buffer, CV_8UC1, 256/2048.0);
                cv_buffer.convertTo(cv_buffer, CV_16UC1, 2048/256.0);
                memcpy( procDepth, cv_buffer.data, bufferWidth*bufferHeight*sizeof(uint16_t) );
            }

            if(circularSet){
                memcpy( bufferPt[0], procDepth, bufferWidth*bufferHeight*sizeof(uint16_t) );
            }

            cv::Mat grad;
            if(outlines[outline_index] == "dark"){
                cv::Mat cv_buffer(bufferHeight, bufferWidth, CV_16UC1 );
                memcpy( cv_buffer.data, procDepth, bufferWidth*bufferHeight*sizeof(uint16_t) );
                cv_buffer.convertTo(cv_buffer, CV_8UC1, 256/2048.0);

                int scale = 1;
                int delta = 0;              
                int ddepth = CV_16S;                
                cv::Mat blurred;
                cv::GaussianBlur( cv_buffer, blurred, cv::Size(3,3), 0, 0, cv::BORDER_REFLECT );
                cv::Mat grad_x, grad_y;
                cv::Mat abs_grad_x, abs_grad_y;
                cv::Sobel( blurred, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
                cv::Sobel( blurred, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
                cv::convertScaleAbs( grad_x, abs_grad_x );
                cv::convertScaleAbs( grad_y, abs_grad_y );
                cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

                // add grad to itself a few times to brighten
                for(int i=0; i<5; i++){    
                    cv::add(grad, grad, grad);
                }
                //cv::GaussianBlur(grad, grad, cv::Size(3,3), 0, 0, cv::BORDER_REFLECT);                

                // scale to max grad
                double minVal; 
                double maxVal; 
                cv::Point minLoc; 
                cv::Point maxLoc;
                cv::minMaxLoc( grad, &minVal, &maxVal, &minLoc, &maxLoc );
                float scale_factor = 255.0/maxVal;
                cv::convertScaleAbs( grad, grad, scale_factor);

                cv::cvtColor(grad, grad, CV_GRAY2RGB);
            }

            

            // move color gradients 
            // TODO: associate speed factor factors (5,3) with each gradient set
            // TODO: allow negative speeds
            // FIXME: remove magic numbers in offset_offset calc (got it by trial and error)
            if(gradientMotionSet){
                double d = gradient_timer.elapsed();
                gradient_timer.reset();
                double offset_offset = 5*speedFactors[speedFactorIndex]*(1.0/d/160.0);
                gradientOffsetA += 5*offset_offset; // 5, 13;
                if(gradientOffsetA>2047.0){ gradientOffsetA = 0.0; }
                gradientOffsetB -= 3*offset_offset; // 3, 7;
                if(gradientOffsetB<0.0){ gradientOffsetB = 2047.0; }
            }

            if(outlines[outline_index] == "bright"){
                cv::Mat cv_buffer(bufferHeight, bufferWidth, CV_16UC1 );
                memcpy( cv_buffer.data, procDepth, bufferWidth*bufferHeight*sizeof(uint16_t) );
                cv_buffer.convertTo(cv_buffer, CV_8UC1, 256/2048.0);

                int scale = 1;
                int delta = 0;              
                int ddepth = CV_16S;                
                cv::Mat blurred;
                cv::GaussianBlur( cv_buffer, blurred, cv::Size(3,3), 0, 0, cv::BORDER_REFLECT );
                cv::Mat grad_x, grad_y;
                cv::Mat abs_grad_x, abs_grad_y;
                cv::Sobel( blurred, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
                cv::Sobel( blurred, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
                cv::convertScaleAbs( grad_x, abs_grad_x );
                cv::convertScaleAbs( grad_y, abs_grad_y );
                cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

                // add grad to itself a few times to brighten
                for(int i=0; i<6; i++){    
                    cv::add(grad, grad, grad);
                }
                //cv::GaussianBlur(grad, grad, cv::Size(3,3), 0, 0, cv::BORDER_REFLECT);                

                // scale to max grad
                double minVal; 
                double maxVal; 
                cv::Point minLoc; 
                cv::Point maxLoc;
                cv::minMaxLoc( grad, &minVal, &maxVal, &minLoc, &maxLoc );
                float scale_factor = 255.0/maxVal;
                cv::convertScaleAbs( grad, grad, scale_factor);

                cv::cvtColor(grad, grad, CV_GRAY2RGB);
            }

            // create combined gradient using both gradients at current offsets
            float gradient_period = gradient_periods[gradient_period_index];
            for(int i=0; i<2048; i++){
                // calc offset and scaled index taking into account wraparound
                int k = (2048+(int)round((i+gradientOffsetA)/gradient_period))%2048;
                int j = (2048+(int)round((i+gradientOffsetB)/gradient_period))%2048; 
                gradientMod[3*i  ] = (uint8_t) MIN(255, MAX( 0, gradient_a[gradient_index][3*j  ]-gradient_b[gradient_index][3*k  ] ));
                gradientMod[3*i+1] = (uint8_t) MIN(255, MAX( 0, gradient_a[gradient_index][3*j+1]-gradient_b[gradient_index][3*k+1] ));
                gradientMod[3*i+2] = (uint8_t) MIN(255, MAX( 0, gradient_a[gradient_index][3*j+2]-gradient_b[gradient_index][3*k+2] ));
            }

            // dim the colors by given factor
            for(int i=0; i<2048*3; i++){
                gradientMod[i] = (uint8_t)(gradientMod[i]/brightnessFactor);
            }

            // convert depth map values to gradient colors
            // skip a few border pixels that seem to flicker
            for( unsigned int x=2 ; x<bufferWidth-4; x++){
                for( unsigned int y=1; y<bufferHeight-2; y++) {
                    unsigned int i = y*bufferWidth + x;
                    unsigned int pval = (unsigned int)m_gamma[procDepth[i]];
                    if(3*int(pval)+2 >= 2048*3){
                        std::cout << "gradientMod out of bounds" << std::endl;
                        std::cout << "pval: " << pval << std::endl;
                        std::cout << "m_gamma[procDepth[i]]: " << m_gamma[procDepth[i]] << std::endl;
                        std::cout << "procDepth[i]: " << procDepth[i] << std::endl;
                        std::cout << "i: " << i << std::endl;
                        std::cout << "x: " << x << std::endl;
                        std::cout << "y: " << y << std::endl;
                    }

                    m_buffer_depth[3*i+0] = gradientMod[3*pval+0];
                    m_buffer_depth[3*i+1] = gradientMod[3*pval+1];
                    m_buffer_depth[3*i+2] = gradientMod[3*pval+2];
                }
            }

            // darken outlines
            if(outlines[outline_index] == "dark"){
                cv::Mat colored_buffer(bufferHeight, bufferWidth, CV_8UC3);
                for (uint y=0; y<bufferHeight; y++){
                    for (uint x=0; x<bufferWidth; x++){
                        unsigned int i = y*bufferWidth + x;
                        colored_buffer.at<cv::Vec3b>(y,x)[0] = m_buffer_depth[3*i+0];
                        colored_buffer.at<cv::Vec3b>(y,x)[1] = m_buffer_depth[3*i+1];
                        colored_buffer.at<cv::Vec3b>(y,x)[2] = m_buffer_depth[3*i+2];                    
                    }
                }              

                cv::subtract(cv::Scalar::all(255), grad, grad);
                cv::multiply(grad, colored_buffer, colored_buffer, 1/255.0);
                
                // convert mat back to vector format
                // FIXME: keep consistent multi-dim array format
                for (uint y=0; y<bufferHeight; y++){
                    for (uint x=0; x<bufferWidth; x++){
                        unsigned int i = y*bufferWidth + x;
                        m_buffer_depth[3*i+0] = colored_buffer.at<cv::Vec3b>(y,x)[0];
                        m_buffer_depth[3*i+1] = colored_buffer.at<cv::Vec3b>(y,x)[1];
                        m_buffer_depth[3*i+2] = colored_buffer.at<cv::Vec3b>(y,x)[2];                    
                    }
                }   
            }

            // brighten outlines
            if(outlines[outline_index] == "bright"){
                cv::Mat colored_buffer(bufferHeight, bufferWidth, CV_8UC3);
                for (uint y=0; y<bufferHeight; y++){
                    for (uint x=0; x<bufferWidth; x++){
                        unsigned int i = y*bufferWidth + x;
                        colored_buffer.at<cv::Vec3b>(y,x)[0] = m_buffer_depth[3*i+0];
                        colored_buffer.at<cv::Vec3b>(y,x)[1] = m_buffer_depth[3*i+1];
                        colored_buffer.at<cv::Vec3b>(y,x)[2] = m_buffer_depth[3*i+2];                    
                    }
                }              

                cv::multiply(grad, colored_buffer, colored_buffer, 1/255.0);
                
                // convert mat back to vector format
                // FIXME: keep consistent multi-dim array format
                for (uint y=0; y<bufferHeight; y++){
                    for (uint x=0; x<bufferWidth; x++){
                        unsigned int i = y*bufferWidth + x;
                        m_buffer_depth[3*i+0] = colored_buffer.at<cv::Vec3b>(y,x)[0];
                        m_buffer_depth[3*i+1] = colored_buffer.at<cv::Vec3b>(y,x)[1];
                        m_buffer_depth[3*i+2] = colored_buffer.at<cv::Vec3b>(y,x)[2];                    
                    }
                }   
            }

            if(fogSet){
                for( unsigned int x=0 ; x<bufferWidth; x++){
                    for( unsigned int y=0; y<bufferHeight; y++) {
                        unsigned int i = y*bufferWidth + x;
                        double depth = procDepth[i];
                        //if (x==0){
                        //    std::cout << "procDepth[" << i << "]: " << depth << std::endl;
                        //}
                        double norm_depth = depth/2048.0;
                        if (norm_depth < fogStart){
                            continue;
                        }
                        double factor = 0.0;
                        if (norm_depth < fogEnd){
                            factor = 1.0-(norm_depth-fogStart)/(fogEnd-fogStart);
                            //if (x==0){
                            //    std::cout << "factor: " << factor << std::endl;
                            //}
                        }
                        // just darkening all channels toward 0 for now
                        // TODO: support first color in current gradient and blend
                        for( unsigned int n=0; n<3; n++) {
                            m_buffer_depth[3*i+n] = (uint8_t)(m_buffer_depth[3*i+n] * factor);
                        }
                    }
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
        device->setLed(LED_RED);
        freenect_angle = 0;
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
    /*case ' ':
        hackyTimer = 0;
        break;*/
    /*case 'm':
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
        break;*/
    case 'c':
    case 'C':
        gradient_index++;
        if (gradient_index>=gradient_a.size()){
            gradient_index = 0;
        }
        sprintf(outputCharBuf,"Color index is now %i", gradient_index);
        setOutputString(outputCharBuf);
        break;
    case 'w':
    case 'W':
        fogSet = !fogSet;
        if (fogSet){
            setOutputString("Fog is ON");
        }else{
            setOutputString("Fog is OFF");
        }
        break;
    case 'e':
        fogIndex += 1;
        if (fogIndex >= fogStarts.size()){ fogIndex = fogStarts.size()-1; }
        fogStart = fogStarts[fogIndex];
		fogEnd = fogStart + fogDepth;
        sprintf(outputCharBuf, "FogStart is now %.3f", fogStarts[fogIndex]);
        setOutputString(outputCharBuf);
        break;
    case 'q':
        fogIndex -= 1;
        if (fogIndex < 0){ fogIndex = 0; }
        fogStart = fogStarts[fogIndex];
		fogEnd = fogStart + fogDepth;
        sprintf(outputCharBuf, "FogStart is now %.3f", fogStarts[fogIndex]);
        setOutputString(outputCharBuf);
        break;
    case 'p':
    case 'P':
        posterizeSet = !posterizeSet;
        if (posterizeSet){
            setOutputString("Posterize is ON");
        }else{
            setOutputString("Posterize is OFF");
        }
        break;
    case 'n':
    case 'N':
        notificationsOn = !notificationsOn;
        if (notificationsOn){
            setOutputString("Notifications are ON");
        }else{
            setOutputString("Notifications are OFF");
        }
        break;
    case 'o':
    case 'O':
        outline_index++;
        if (outline_index>=outlines.size()){outline_index=0;}
        sprintf(outputCharBuf,"Outline style is now %s", outlines[outline_index].c_str());
        setOutputString(outputCharBuf);
        break;
    case 'l':
    case 'L':
        circularSet = !circularSet;
        if (circularSet){
            setOutputString("Loop is ON");
        }else{
            setOutputString("Loop is OFF");
        }
        break;
    case 's':
    case 'S':
        showFPSset = !showFPSset;
        if (showFPSset){
            setOutputString("Show FPS is ON");
        }else{
            setOutputString("Show FPS is OFF");
        }
        break;
    case 'm':
    case 'M':
        mirrorSet = !mirrorSet;
        if (mirrorSet){
            setOutputString("Display is flipped");
        }else{
            setOutputString("Display is not flipped");
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
    case '1':
        /*gradient_index = 1;
        speedFactorIndex = 3;
        gradient_period_index = 4;
        posterizeSet = true;
        outline_index = 3;
        currentBuffers = 35;*/
        presets["1"].set_global();
        break;
    case '2':
        gradient_index = 2;
        speedFactorIndex = 2;
        gradient_period_index = 0;
        posterizeSet = false;
        outline_index = 2;
        currentBuffers = 13;
        break;
    case '3':
        gradient_index = 4;
        speedFactorIndex = 5;
        gradient_period_index = 4;
        posterizeSet = false;
        outline_index = 2;
        currentBuffers = 20;
        break;
    case '4':
        gradient_index = 0;
        speedFactorIndex = 1;
        gradient_period_index = 5;
        posterizeSet = false;
        outline_index = 3;
        currentBuffers = 45;
        break;
    // FIXME: rapid switching between preset 5 and any other preset 
    //        results in openCV error: sizes of input arguments do not match
    // - disabling for now, glitchy outlines weren't that cool anyway
    /*case '5':
        gradient_index = 0;
        speedFactorIndex = 5;
        gradient_period_index = 7;
        posterizeSet = false;
        outline_index = 1;
        currentBuffers = 7;
        break;*/
    case '5':
        gradient_index = 1;
        speedFactorIndex = 2;
        gradient_period_index = 2;
        posterizeSet = false;
        outline_index = 2;
        currentBuffers = 35;
        break;
    case '6':
        gradient_index = 0;
        speedFactorIndex = 1;
        gradient_period_index = 0;
        posterizeSet = false;
        outline_index = 0;
        currentBuffers = 35;
        break;
    case '7':
        gradient_index = 4;
        speedFactorIndex = 1;
        gradient_period_index = 2;
        posterizeSet = false;
        outline_index = 1;
        currentBuffers = 20;
        break;
    case '8':
        {
            Params p = Params();
            p.gradient_index = 0;
            p.speedFactorIndex = 3;
            p.gradient_period_index = 4;
            p.posterizeSet = false;
            p.outline_index = 2;
            p.currentBuffers = 20;
            p.set_global();
        }
        break;
    case '!':
    case '@':
    case '#':
    case '$':
    case '%':
    case '^':
    case '&':
    case '*':
        add_event(shift_nums[to_string((int)key)], 200.);
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
    case '0':
    case ')':
        speedFactorIndex += 1;
        if (speedFactorIndex >= speedFactors.size() ){ speedFactorIndex = speedFactors.size()-1; }
        sprintf(outputCharBuf,"Speed factor is now %.2f", speedFactors[speedFactorIndex]);
        setOutputString(outputCharBuf);
        break;
    case '9':
    case '(':
        speedFactorIndex -= 1;
        if (speedFactorIndex < 0 ){ speedFactorIndex = 0; }
        sprintf(outputCharBuf,"Speed factor is now %.2f", speedFactors[speedFactorIndex]);
        setOutputString(outputCharBuf);
        break;
    case '\'':
    case '"':
        gradient_period_index += 1;
        if (gradient_period_index >= gradient_periods.size() ){ gradient_period_index = gradient_periods.size()-1; }
        sprintf(outputCharBuf,"Gradient period is now %.3f", gradient_periods[gradient_period_index]);
        setOutputString(outputCharBuf);
        break;
    case ':':
    case ';':
        gradient_period_index -= 1;
        if (gradient_period_index < 0 ){ gradient_period_index = 0; }
        sprintf(outputCharBuf,"Gradient period is now %.3f", gradient_periods[gradient_period_index]);
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
		std::string s,
        int decimals = 100000) {
    char* c;
    int yOffset = 0;
    int maxY = glutGet(GLUT_WINDOW_HEIGHT);
    int maxX = glutGet(GLUT_WINDOW_WIDTH);
    glWindowPos3f(x*maxX/640, maxY-y, z);
    int d = -1;
    bool past_dot = false;
    for (char& c : s) {      
        if (c=='.'){
            if ( decimals==0 ){
                return;
            } else {
                past_dot = true;
            }
        }
        if ( past_dot ){
            d++;
        }
        if ( d > decimals ){
            return;
        }
        if (c=='\n'){ 
            yOffset -= lineSpacing; 
            glWindowPos3f(x*maxX/640, maxY-y+yOffset, z); 
        }
        glutBitmapCharacter(font, c);
    }
}

void renderOutlinedString(float x,
                          float y,
                          std::string s,
                          int decimals = 100000) {
    if (notificationsOn) {
        // x = 40.0f
        // y = 80.0f
        // disable lighting and texture so text color comes through
        glDisable(GL_LIGHTING);
        glDisable(GL_TEXTURE_2D);
        // first draw a jittered version of the string to create a dark outline around the text
        glColor3d(0.2, 0.0, 0.0);
        renderBitmapString( windowPad+x,y,-0.5f, font, s, decimals);
        renderBitmapString( windowPad+x+1.0f,y+1.0f,-0.5f, font, s, decimals);
        renderBitmapString( windowPad+x+1.0f,y,-0.5f, font, s, decimals);
        renderBitmapString( windowPad+x-1.0f,y-1.0f,-0.5f, font, s, decimals);
        renderBitmapString( windowPad+x-1.0f,y,-0.5f, font, s, decimals);
        renderBitmapString( windowPad+x-1.0f,y+1.0f,-0.5f, font, s, decimals);
        renderBitmapString( windowPad+x,y+1.0f,-0.5f, font, s, decimals);
        renderBitmapString( windowPad+x+1.0f,y-1.0f,-0.5f, font, s, decimals);
        renderBitmapString( windowPad+x,y-1.0f,-0.5f, font, s, decimals);
        // now draw main text on top
        glColor3d(0.4, 1.0, 0.7);
        renderBitmapString( windowPad+x,y,-0.5f, font, s, decimals);
    }
}

void DrawGLScene()
{
    poll_event_queue();

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
    glTexCoord2f(mirrorSet, 0); glVertex3f(windowPad,0,-1);
    glTexCoord2f(!mirrorSet, 0); glVertex3f(640-windowPad,0,-1);
    glTexCoord2f(!mirrorSet, 1); glVertex3f(640-windowPad,480,-1);
    glTexCoord2f(mirrorSet, 1); glVertex3f(windowPad,480,-1);
    glEnd();

    // countdown user notification timer, blank if 0
    hackyTimer--;
    if (hackyTimer < 0){ hackyTimer = 0; outputString = ""; lineSpacing=25;}    

    // render user notifications
    renderOutlinedString(40.0f, 80.0f, outputString);
    
    /*
    if(showFPSset){
        double d = fps_timer.elapsed();
        fps_timer.reset(); 
        std::string s = std::to_string(1.0f/d);
        renderOutlinedString(40.0f, 60.0f, s, 0);            
    }*/

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

void init(){
    init_presets();
    initGradients();
}

//define main function
int main(int argc, char **argv) {
    init();

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

