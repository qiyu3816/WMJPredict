#pragma once
#include <iostream>
#include <thread>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace predict
{
    /**
     * @brief KF预测器
     */
    class KF
    {
    public:
        enum FILTER_MODE{IDLE, POINT, POSE, SINGLE};
        FILTER_MODE m_filter_mod;
        int         m_measure_num;
        int         m_state_num;

        int         m_init_count_threshold;
        float       m_process_noise_cov;
        float       m_measure_noise_cov;
        float       m_control_freq;
        float       m_predict_coe;
        cv::Mat     m_measurement;
        std::shared_ptr<cv::KalmanFilter> m_KF;

        cv::Point3f m_last_position;
        float       m_last_armor_dis;
        float       m_cur_armor_dis;
        float       m_target_change_dist_threshold;
        
        cv::Point2f      m_last_pose;
        float            m_target_change_pose_threshold;
        
        float       m_last_value;
        float       m_value_diff;

        KF(KF::FILTER_MODE);
        ~KF() {}

        void setParam();
        cv::Point3f     predict(cv::Point3f); // 三维点的预测
        cv::Point2f     predict(cv::Point2f); // 二维点的预测
        float           predict(float); // 一维点的预测
        void initFilter(cv::Point3f); // 三维点滤波器初始化
        void initFilter(cv::Point2f); // 二维点滤波器初始化
        void initFilter(float); // 一维点滤波器初始化

    private:
        bool        m_initialized{false};
        int         m_debug;
    };
}
