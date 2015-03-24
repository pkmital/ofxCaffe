#pragma once

#include "ofxCaffe.h"

class pkmDeepFeature {
    
public:
    
    pkmDeepFeature()
    {
        
    }
    
    void allocate()
    {
        caffe = std::shared_ptr<ofxCaffe>(new ofxCaffe());
        caffe->initModel(ofxCaffe::getModelTypes()[ofxCaffe::OFXCAFFE_MODEL_BVLC_CAFFENET]);
    }
    
    void setImage(cv::Mat &img)
    {
        caffe->forward(img);
        feature_fc5 = caffe->getLayerByName("conv5", true);
        feature_fc5.setTranspose();
        
        feature_prob = caffe->getLayerByName("prob", false);
        feature_prob.setTranspose();
    }
    
    void draw(int w = 1240, int h = 300)
    {
        caffe->drawGraph(feature_fc5, "conv5", 20, 0, w, h, 255.0f);
        caffe->drawGraph(feature_prob, "prob", 20, 40, w, h, 1.0f);
    }
    
    pkm::Mat& getFeatureFC5()
    {
        return feature_fc5;
    }
    
    pkm::Mat& getFeatureProb()
    {
        return feature_prob;
    }
    

private:
    //--------------------------------------------------------------
    // ptr to caffe obj
    std::shared_ptr<ofxCaffe> caffe;
    
    pkm::Mat feature_fc5;
    pkm::Mat feature_prob;
    
};