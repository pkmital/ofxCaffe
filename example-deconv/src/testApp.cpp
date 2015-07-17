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

//--------------------------------------------------------------
void testApp::setup(){
    b_setup = false;
    b_filter = false;
    
#ifdef DO_LOAD_VIDEO
    video.loadMovie("target.mov");
    width_og = video.getWidth();
    height_og = video.getHeight();
    total_frames = video.getTotalNumFrames();
    current_frame = 0;
    
    cout << "Loaded target.mov (" << width << "x" << height << ") with " << total_frames << " frames" << endl;
#else
#ifdef DO_LOAD_SYPHON
    syphon.setup();
    syphon.set("synthesis", "Corpus Based Visual Synthesis");
    width_og = 1280; height_og = 720;
    fbo.allocate(width_og, height_og, GL_RGB);
    pixels.allocate(width_og, height_og, OF_PIXELS_RGB);
#else
    width_og = 1280; height_og = 720;
    camera.setDeviceID(0);
    camera.initGrabber(width, height);
#endif
#endif
    
    ofSetWindowShape(width_og, height_og);
    
    width = width_og * .6; height = height_og * .6;
    
    synthesis_img.allocate(width, height);
    synthesis_h.allocate(width, height);
    synthesis_s.allocate(width, height);
    synthesis_v.allocate(width, height);
    camera_img.allocate(width_og, height_og);
    camera_img_rsz.allocate(width, height);
//    cv::Mat m(synthesis_img.getCvImage());
//
//    for(int i=0; i<m.rows; i++)
//    {
//        for(int j=0; j<m.cols; j++)
//        {
//            m.at<cv::Vec3b>(i, j)[0] = rand() % 255;
//            m.at<cv::Vec3b>(i, j)[1] = rand() % 255;
//            m.at<cv::Vec3b>(i, j)[2] = rand() % 255;
//        }
//    }
//    synthesis_img.flagImageChanged();
    cout << "allocating " << width << " x " << height << endl;
    caffe.allocate(width, height);
    
    filter = cv::AdaptiveManifoldFilter::create();
    filter->set("sigma_s", 4.0);
    filter->set("sigma_r", 0.15);
    filter->set("tree_height", -1);
    filter->set("num_pca_iterations", 1);
}

//--------------------------------------------------------------
void testApp::update()
{
    ofSetWindowTitle(ofToString(ofGetFrameRate()));
    
#ifdef DO_LOAD_VIDEO
    if(current_frame > total_frames)
    {
        OF_EXIT_APP(0);
    }
    video.setFrame(current_frame++);
    camera_img.setFromPixels(video.getPixels());
#else
#ifdef DO_LOAD_SYPHON
    fbo.begin();
    syphon.draw(0, 0, width_og, height_og);
    fbo.end();
    fbo.readToPixels(pixels);
    camera_img.setFromPixels(pixels);
#else
    camera.update();
    camera_img.setFromPixels(camera.getPixels());
#endif
#endif
    camera_img_rsz.scaleIntoMe(camera_img);
    
    
    if(b_filter)
    {
        cv::Mat dst, tilde_dst, joint_img;
        filter->apply(cv::Mat(camera_img_rsz.getCvImage()), cv::Mat(camera_img_rsz.getCvImage()), tilde_dst, joint_img);
        camera_img_rsz.flagImageChanged();
    }
    
    if(b_setup)
        caffe.update(cv::Mat(camera_img_rsz.getCvImage()));

}

//--------------------------------------------------------------
void testApp::draw(){
    
    ofBackground(255);
    ofSetColor(255);
    if(!b_setup)
    {
        camera_img.draw(0, 0, width_og, height_og);
    }
    
    caffe.draw(0, 0, width_og, height_og);
}

//--------------------------------------------------------------
void testApp::keyPressed(int key){
    ofResetElapsedTimeCounter();
    
    cout << key << endl;
    if (key == ' ')
    {
        caffe.update(cv::Mat(camera_img_rsz.getCvImage()));
        b_setup = true;
    }
    else
        caffe.keyPressed(key);
    
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
