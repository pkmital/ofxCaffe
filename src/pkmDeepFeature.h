#pragma once

#include "ofxCaffe.hpp"

class pkmDeepFeature {
    
public:
    
    pkmDeepFeature();
    void allocate();
    
    void setImage(cv::Mat &img);
    void updateFeatures();
    
    pkm::Mat getEdgeFeaturesFor(size_t x, size_t y);
    
    void draw(int w = 1240, int h = 300);
    
    pkm::Mat& getFeatureFC5();
    pkm::Mat& getFeatureProb();
    float getMaxFeatureProb();

private:
    //--------------------------------------------------------------
    // ptr to caffe obj
    std::shared_ptr<ofxCaffe> caffe;
    
    pkm::Mat feature_fc5;
    pkm::Mat feature_prob;
    
    size_t width, height;
    
};