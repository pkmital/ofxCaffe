/*
 
 ofxCaffe - testApp.h
 
 The MIT License (MIT)
 
 Copyright (c) 2015 Parag K. Mital, http://pkmital.com
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 
 */

#pragma once

#include "ofMain.h"
#include "ofxOpenCv.h"
#include "ofxCaffe.h"
#include "ofxOscSender.h"
#include "ofxOscReceiver.h"
#include <stdlib.h>
#include <thread>
//#include <boost/interprocess/shared_memory_object.hpp>

void threadedSay(string sentence);

//--------------------------------------------------------------
class pkmImageToText {
public:
    
    pkmImageToText()
    {
        b_mutex = false;
        b_0 = b_1 = true;
        b_2 = b_3 = b_4 = false;
        
        layer_num = 1;
        color_img.setUseTexture(false);
        color_img.allocate(1, 1);
        
        sender.setup("localhost", 1239);
        receiver.setup(1234);
    }
    
    void allocate(size_t w, size_t h)
    {
        width = w; height = h;
        color_img.allocate(w, h);
        
        current_model = 0;
        
        caffe = std::shared_ptr<ofxCaffe>(new ofxCaffe());
        caffe->initModel(ofxCaffe::getModelTypes()[current_model]);
    }
    
    void update(ofPixels &pixels)
    {
        if(pixels.getPixels() != NULL && pixels.getWidth() > 0 && pixels.getHeight() > 0)
        {
            if(color_img.getWidth() != pixels.getWidth() || color_img.getHeight() != pixels.getHeight())
                color_img.allocate(pixels.getWidth(), pixels.getHeight());
            
            color_img.setFromPixels(pixels.getPixels(), pixels.getWidth(), pixels.getHeight());
            cv::Mat img(color_img.getCvImage()), img2;
            img.copyTo(img2);
            caffe->forward(img);
        }
    }
    
    string getPredictedLabel()
    {
        caffe->getPredictedLabel();
    }
    
    void receiveOSCFeatures()
    {
        if(current_model != 0)
            return;
        
        while (receiver.hasWaitingMessages()) {
            ofxOscMessage m;
            receiver.getNextMessage(&m);
            string address = m.getAddress();
            last_sentence = m.getArgAsString(0);
//            threadedSay(last_sentence);
            threadedSay(caffe->getPredictedLabel());
        }
    }
    
    void sendOSCFeatures(int layer_num = 19)
    {
        if(current_model != 0)
            return;
        
        if(ofGetElapsedTimef() > 1.0f)
        {
            ofxOscMessage m;
            m.setAddress("/features");
            const float *ptr = caffe->getCPUDataForOutputLayer(layer_num);
            for (int i = 0; i < 4096; i++) {
                m.addFloatArg(*(ptr+i));
            }
            sender.sendMessage(m, true);
        }
    }
    
    void sendOSCLabel()
    {
        ofxOscMessage m;
        m.setAddress("/label");
        m.addStringArg(caffe->getPredictedLabel());
        sender.sendMessage(m, false);
    }
    
    void draw()
    {
        ofSetColor(255);
//        color_img.draw(0, 0, width, height);
        
        if (caffe->getModelType() == ofxCaffe::OFXCAFFE_MODEL_BVLC_CAFFENET_34x17 ||
            caffe->getModelType() == ofxCaffe::OFXCAFFE_MODEL_BVLC_CAFFENET_8x8)
        {
            ofEnableAlphaBlending();
            ofSetColor(255, 255, 255, 200);
            caffe->drawDetectionGrid(width, height);
            ofDisableAlphaBlending();
        }
        
        ofDrawBitmapStringHighlight("model (-/+):  " + caffe->getModelTypeNames()[current_model], 20, 30);
        
        ofDisableAlphaBlending();
        if (b_1)
            caffe->drawLabelAt(20, 50);
        if (b_2)
            caffe->drawLayerXParams(0, 80, width, 32, layer_num, ofGetElapsedTimef() * 10.0);
        if (b_3)
            caffe->drawLayerXOutput(0, 420, width, 32, layer_num);
        if (b_4)
            caffe->drawProbabilities(0, 500, width, 200);
        
        
        ofDrawBitmapStringHighlight("prediction:  " + last_sentence, 20, 90);
    }
    
    void keyPressed(int key)
    {
        cout << key << endl;
        if (key == '0')
            b_0 = !b_0;
        else if (key == '1')
            b_1 = !b_1;
        else if (key == '2')
            b_2 = !b_2;
        else if (key == '3')
            b_3 = !b_3;
        else if (key == '4')
            b_4 = !b_4;
        else if (key == '[')
        {
            layer_num = std::max<int>(0, layer_num - 1);
            cout << "layer_num '[' or ']': " << layer_num << endl;
        }
        else if (key == ']')
        {
            while (b_mutex) {} b_mutex = true;
            layer_num = std::min<int>(caffe->getTotalNumBlobs(), layer_num + 1);
            b_mutex = false;
            cout << "layer_num '[' or ']': " << layer_num << endl;
        }
        else if (key == '-' || key == '_')
        {
            current_model = (current_model == 0) ? (ofxCaffe::getTotalModelNums() - 1)  : (current_model - 1);
            while (b_mutex) {} b_mutex = true;
            caffe.reset();
            caffe = std::shared_ptr<ofxCaffe>(new ofxCaffe());
            caffe->initModel(ofxCaffe::getModelTypes()[current_model]);
            b_mutex = false;
        }
        else if (key == '+' || key == '=')
        {
            current_model = (current_model + 1) % ofxCaffe::getTotalModelNums();
            while (b_mutex) {} b_mutex = true;
            caffe.reset();
            caffe = std::shared_ptr<ofxCaffe>(new ofxCaffe());
            caffe->initModel(ofxCaffe::getModelTypes()[current_model]);
            b_mutex = false;
        }
    }
    
private:
    
    //--------------------------------------------------------------
    size_t width, height;
    
    //--------------------------------------------------------------
    
    
    //--------------------------------------------------------------
    // which model have we loaded
    int current_model;
    
    //--------------------------------------------------------------
    // osc
    ofxOscSender sender;
    ofxOscReceiver receiver;
    string last_sentence;
    
    //    boost::interprocess::shared_memory_object shm;
    
    //--------------------------------------------------------------
    // ptr to caffe obj
    std::shared_ptr<ofxCaffe> caffe;
    ofxCvColorImage color_img;
    
    
    //--------------------------------------------------------------
    // which layer are we visualizing
    int layer_num;
    
    
    //--------------------------------------------------------------
    // simple flags for switching on drawing options of camera image/layers/parameters/probabilities
    bool b_0, b_1, b_2, b_3, b_4;
    bool b_mutex;
};

//--------------------------------------------------------------
class ImgAnalysisThread: public ofThread {
public:
    ImgAnalysisThread()
    :newFrame(true) {
        // start the thread as soon as the
        // class is created, it won't use any CPU
        // until we send a new frame to be analyzed
        startThread();
    }
    ~ImgAnalysisThread() {
        // when the class is destroyed
        // close both channels and wait for
        // the thread to finish
        toAnalize.close();
        analized.close();
        waitForThread(true);
    }
    
    void allocate(int w, int h) {
        image_to_text.allocate(w, h);
    }

    void analyze(ofPixels & pixels) {
        // send the frame to the thread for analyzing
        // this makes a copy but we can't avoid it anyway if
        // we want to update the grabber while analyzing
        // previous frames
        this->pixels = pixels;
        texture.loadData(pixels);
    }
    
    void update() {
        // check if there's a new analyzed frame and upload
        // it to the texture. we use a while loop to drop any
        // extra frame in case the main thread is slower than
        // the analysis
        // tryReceive doesn't reallocate or make any copies
        newFrame = false;
//        while(analized.tryReceive(pixels)){
//            newFrame = true;
//        }
        if(newFrame){
            texture.loadData(pixels);
        }
    }
    bool isFrameNew() {
        return newFrame;
    }
    ofPixels & getPixels() {
        return pixels;
    }
    ofTexture & getTexture() {
        return texture;
    }
    void draw() {
        texture.draw(0, 0);
        image_to_text.draw();
    }
    
    void keyPressed(int key) {
        image_to_text.keyPressed(key);
    }
    
private:
    void threadedFunction() {
        while(isThreadRunning()){
            // wait until there's a new frame
            // this blocks the thread, so it doesn't use
            // the CPU at all, until a frame arrives.
            // also receive doesn't allocate or make any copies
//            ofPixels pixels;
            if(pixels.isAllocated()){
                // we have a new frame, process it, the analysis

                cout << "updating...";
                image_to_text.update(pixels);
                image_to_text.sendOSCLabel();
//                cout << ".";
//                image_to_text.sendOSCFeatures();
//                cout << ".";
//                image_to_text.receiveOSCFeatures();
//                threadedSay(image_to_text.getPredictedLabel());
                newFrame = true;
                cout << ".done" << endl;
                
                // once processed send the result back to the
                // main thread. in c++11 we can move it to
                // avoid a copy
//#if __cplusplus>=201103
//                analized.send(std::move(pixels));
//#else
//                analized.send(pixels);
//#endif
            }
//            else{
//                // if receive returns false the channel
//                // has been closed, go out of the while loop
//                // to end the thread
//                break;
//            }
        }
    }
    
    pkmImageToText image_to_text;
    ofThreadChannel<ofPixels> toAnalize;
    ofThreadChannel<ofPixels> analized;
    ofPixels pixels;
    ofTexture texture;
    bool newFrame;
};


class testApp : public ofBaseApp{

public:
    //--------------------------------------------------------------
    void setup();
    void update();
    void draw();
    
//    ~testApp() {
//        bool removed = shared_memory_object::remove("features");
//    }
    
    //--------------------------------------------------------------
    void keyPressed(int key);
    void keyReleased(int key);
    void mouseMoved(int x, int y );
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    void windowResized(int w, int h);
    void dragEvent(ofDragInfo dragInfo);
    void gotMessage(ofMessage msg);
	
    //--------------------------------------------------------------
    // camera and opencv image objects
    ofVideoGrabber camera;

    ImgAnalysisThread image_to_text;
    
    //--------------------------------------------------------------
    // image and window dimensions
    int width, height;
    
    //--------------------------------------------------------------
    // hacky mutex for when changing caffe model
    bool b_mutex, b_setup;
    
};