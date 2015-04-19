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
        width = img.cols;
        height = img.rows;
        caffe->forward(img);
    }
    
    void updateFeatures()
    {
//        feature_fc5 = caffe->getLayerByName("conv5", false);
//        feature_fc5.setTranspose();
//     
//        feature_prob = caffe->getLayerByName("prob", false);
//        feature_prob.setTranspose();
    }
    
    pkm::Mat getEdgeFeaturesFor(size_t x, size_t y)
    {
        // 1st layer is "conv1"
        // 2nd layer is "norm1"
        // 3rd layer is "pool1"
        // 10th layer is "pool5"
        
        size_t layer_num = 1;
        
        const float* fp_from = caffe->getCPUDataForOutputLayer(layer_num);
        
        size_t filter_channels = caffe->getChannelsForLayer(layer_num);
        size_t filter_width = caffe->getWidthForLayer(layer_num);
        size_t filter_height = caffe->getHeightForLayer(layer_num);
        
        // 5th num is center crop
        size_t n = 4;
        
        // scale to filter size
//        cout << "ch:: " << filter_channels << " w:: " << width << " h:: " << height << " fw:: " << filter_width << " fh:: " << filter_height << " x: " << x << " y: " << y;
        x = std::min<size_t>(filter_width, floor(x / (float)(width - 1) * (float)(filter_width - 1)));
        y = std::min<size_t>(filter_height, floor(y / (float)(height - 1) * (float)(filter_height - 1)));
//        cout << " nx: " << x << " ny: " << y << endl;
        
        pkm::Mat features(filter_channels, 1);
        for (size_t channel_i = 0; channel_i < filter_channels; channel_i++)
        {
            features[channel_i] = (fp_from[ ((n * filter_channels + channel_i) * filter_height + y) * filter_width + x ]);
        }
        
        return features;
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
    
    size_t width, height;
    
};