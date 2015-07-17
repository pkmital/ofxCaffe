/*
 
 ofxCaffe - testApp.cpp
 
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

#include "testApp.h"
#include <memory>
//#include <boost/interprocess/shared_memory_object.hpp>
//#include <boost/interprocess/mapped_region.hpp>

//--------------------------------------------------------------
void testApp::setup(){
    b_setup = false;
    
    width = 1280; height = 720;
    
//    shm = boost::interprocess::shared_memory_object(boost::interprocess::create_only, "features", boost::interprocess::read_write);
//    shm.truncate(sizeof(float)*4096);

    ofSetWindowShape(width, height);
    camera.initGrabber(width, height);
    camera.listDevices();
    camera.setDeviceID(1);
    
    image_to_text.allocate(width, height);
    
    b_setup = true;
}

//--------------------------------------------------------------
void testApp::update(){
    if(!b_setup) return;
    
    ofSetWindowTitle(ofToString(ofGetFrameRate()));
    
    camera.update();
    
//    image_to_text.analyze(camera.getPixels());
//    image_to_text.update();
    
    
    image_to_text.update(camera.getPixels());
    image_to_text.sendOSCFeatures();
    image_to_text.receiveOSCFeatures();
}

//--------------------------------------------------------------
void testApp::draw(){
    if(!b_setup) return;
    
    ofBackground(255);
    ofSetColor(255);
    camera.draw(0, 0);
    image_to_text.draw();
}

//--------------------------------------------------------------
void testApp::keyPressed(int key){
    ofResetElapsedTimeCounter();
    
    image_to_text.keyPressed(key);
}

//--------------------------------------------------------------
void testApp::keyReleased(int key){

}

//--------------------------------------------------------------
void testApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::windowResized(int w, int h){
    width = w;
    height = h;

}

//--------------------------------------------------------------
void testApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void testApp::dragEvent(ofDragInfo dragInfo){ 

}
