#include "../include/KF.hpp"

namespace predict
{
    /**
     * @brief 预测器构造函数
     *
     * 构造函数传入测量量和状态量维度 现有 3×9 2×6 1×3
     * 需要其他维度的 自行扩展initFilter和predict
     *
     * @param a_filter_mod 预测模式
     */
    KF::KF(KF::FILTER_MODE a_filter_mode)
    {
        if((m_filter_mod = a_filter_mode) == POINT)
        {
            m_filter_mod = POINT;
            setParam();

            m_measure_num = 3;
            m_state_num = 9;

            m_KF = std::make_shared<cv::KalmanFilter>(m_state_num, m_measure_num, 0);
            m_measurement = cv::Mat::zeros(m_measure_num, 1, CV_32F);
            cv::setIdentity(m_KF->measurementMatrix);
            cv::setIdentity(m_KF->processNoiseCov, cv::Scalar::all(m_process_noise_cov));
            cv::setIdentity(m_KF->measurementNoiseCov, cv::Scalar::all(m_measure_noise_cov));
            cv::setIdentity(m_KF->errorCovPost, cv::Scalar::all(1));
            float dt = 1.f / m_control_freq;
            m_KF->transitionMatrix = (cv::Mat_<float>(m_state_num, m_state_num) <<
                                      1, 0, 0, dt, 0, 0, 0.5 * dt * dt, 0, 0,
                                      0, 1, 0, 0, dt, 0, 0, 0.5 * dt * dt, 0,
                                      0, 0, 1, 0, 0, dt, 0, 0, 0.5 * dt * dt,
                                      0, 0, 0, 1, 0, 0, dt, 0, 0,
                                      0, 0, 0, 0, 1, 0, 0, dt, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, dt,
                                      0, 0, 0, 0, 0, 0, 1, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 1, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 1);
        }
        else if((m_filter_mod = a_filter_mode) == POSE)
        {
            setParam();

            m_measure_num = 3;
            m_state_num = 6;

            m_KF = std::make_shared<cv::KalmanFilter>(m_state_num, m_measure_num, 0);
            m_measurement = cv::Mat::zeros(m_measure_num, 1, CV_32F);
            cv::setIdentity(m_KF->measurementMatrix);
            cv::setIdentity(m_KF->processNoiseCov, cv::Scalar::all(m_process_noise_cov));
            cv::setIdentity(m_KF->measurementNoiseCov, cv::Scalar::all(m_measure_noise_cov));
            cv::setIdentity(m_KF->errorCovPost, cv::Scalar::all(1));
            float dt = 1.f / m_control_freq;
            m_KF->transitionMatrix = (cv::Mat_<float>(m_state_num, m_state_num) <<
                                      1, 0, dt, 0, 0.5 * dt * dt, 0,
                                      0, 1, 0, dt, 0, 0.5 * dt * dt,
                                      0, 0, 1, 0, dt, 0,
                                      0, 0, 0, 1, 0, dt,
                                      0, 0, 0, 0, 1, 0,
                                      0, 0, 0, 0, 0, 1);
        }
        else if((m_filter_mod = a_filter_mode) == SINGLE)
        {
            setParam();

            m_measure_num = 1;
            m_state_num = 3;

            m_KF = std::make_shared<cv::KalmanFilter>(m_state_num, m_measure_num, 0);
            m_measurement = cv::Mat::zeros(m_measure_num, 1, CV_32F);
            cv::setIdentity(m_KF->measurementMatrix);
            cv::setIdentity(m_KF->processNoiseCov, cv::Scalar::all(m_process_noise_cov));
            cv::setIdentity(m_KF->measurementNoiseCov, cv::Scalar::all(m_measure_noise_cov));
            cv::setIdentity(m_KF->errorCovPost, cv::Scalar::all(1));
            float dt = 1.f / m_control_freq;
            m_KF->transitionMatrix = (cv::Mat_<float>(m_state_num, m_state_num) <<
                                      1, dt, 0.5 * dt * dt,
                                      0, 1, dt,
                                      0, 0, 1);
        }
    }

    /**
     * @brief 根据模式初始化参数
     */
    void KF::setParam()
    {
        cv::FileStorage fs(KF_CFG, cv::FileStorage::READ);
        m_initialized = false;
        if(m_filter_mod == POINT)
        {
            fs["point"]["debug"]                        >> m_debug;
            fs["point"]["init_count_threshold"]         >> m_init_count_threshold;
            fs["point"]["predict_coe"]                  >> m_predict_coe;
            fs["point"]["control_freq"]                 >> m_control_freq;
            fs["point"]["process_noise_cov"]            >> m_process_noise_cov;
            fs["point"]["measure_noise_cov"]            >> m_measure_noise_cov;
            fs["point"]["target_change_dist_threshold"] >> m_target_change_dist_threshold;
        } else if(m_filter_mod == POSE) {
            fs["pose"]["debug"]                        >> m_debug;
            fs["pose"]["init_count_threshold"]         >> m_init_count_threshold;
            fs["pose"]["predict_coe"]                  >> m_predict_coe;
            fs["pose"]["control_freq"]                 >> m_control_freq;
            fs["pose"]["process_noise_cov"]            >> m_process_noise_cov;
            fs["pose"]["measure_noise_cov"]            >> m_measure_noise_cov;
            fs["pose"]["target_change_dist_threshold"] >> m_target_change_dist_threshold;
            fs["pose"]["target_change_pose_threshold"] >> m_target_change_pose_threshold;
        } else if(m_filter_mod == SINGLE) {
            fs["single"]["debug"]                >> m_debug;
            fs["single"]["init_count_threshold"] >> m_init_count_threshold;
            fs["single"]["predict_coe"]          >> m_predict_coe;
            fs["single"]["control_freq"]         >> m_control_freq;
            fs["single"]["process_noise_cov"]    >> m_process_noise_cov;
            fs["single"]["measure_noise_cov"]    >> m_measure_noise_cov;
            fs["single"]["value_diff"]           >> m_value_diff;
        }
        fs.release();
    }

    /////////////// POINT ///////////////

    /**
     * @brief 滤波器初始化
     *
     * @param a_position 第一帧的三维坐标
     */
    void KF::initFilter(cv::Point3f a_position)
    {
        m_KF->statePost = (cv::Mat_<float>(m_state_num, 1) <<
                           a_position.x, a_position.y, a_position.z, 0, 0, 0, 0, 0, 0);
        m_KF->predict();
        m_measurement.at<float>(0) = a_position.x;
        m_measurement.at<float>(1) = a_position.y;
        m_measurement.at<float>(2) = a_position.z;

        m_cur_armor_dis = std::sqrt(std::pow(a_position.x, 2) + std::pow(a_position.y, 2) + std::pow(a_position.z, 2));
        m_last_armor_dis = std::sqrt(std::pow(a_position.x, 2) + std::pow(a_position.y, 2) + std::pow(a_position.z, 2));

        for(int i = 0; i < m_init_count_threshold; i++)
        {
            m_KF->correct(m_measurement);
            m_KF->predict();
        }
    }

    /**
     * @brief 根据新的绝对坐标测量量预测
     *
     * @param a_position 当前装甲绝对坐标
     *
     * @return 返回预测后的绝对坐标
     */
    cv::Point3f KF::predict(cv::Point3f a_position)
    {
        cv::Mat prediction;
        m_cur_armor_dis = std::sqrt(std::pow(a_position.x, 2) + std::pow(a_position.y, 2) + std::pow(a_position.z, 2));

        if(!m_initialized)
        {
            std::cout << "Init" << std::endl;
            initFilter(a_position);
            m_last_position = a_position;
            m_initialized = true;
        }

        m_measurement.at<float>(0) = a_position.x;
        m_measurement.at<float>(1) = a_position.y;
        m_measurement.at<float>(2) = a_position.z;
        m_KF->correct(m_measurement);
        prediction = m_KF->predict();
        if(std::sqrt(std::pow(std::fabs(a_position.x - m_last_position.x), 2) + std::pow(std::fabs(a_position.y - m_last_position.y), 2) + std::pow(std::fabs(a_position.z - m_last_position.z), 2)) > m_target_change_dist_threshold )
        {
            initFilter(a_position);
            if(m_debug)
                std::cout << "Target has changed" << std::endl;
        }
        m_last_position = a_position;
        m_last_armor_dis = m_cur_armor_dis;

        return cv::Point3f(prediction.at<float>(0) + m_predict_coe * prediction.at<float>(3),
                           prediction.at<float>(1) + m_predict_coe * prediction.at<float>(4),
                           prediction.at<float>(2) + m_predict_coe * prediction.at<float>(5));
    }


    /////////////// PosePredict///////////////

    /**
     * @brief 滤波器初始化
     *
     * @param a_pose 第一帧的二维坐标, a_position 第一帧的三维坐标
     */
    void KF::initFilter(cv::Point2f a_pose)
    {
        m_KF->statePost = (cv::Mat_<float>(m_state_num, 1) <<
                           a_pose.x, a_pose.y, 0, 0, 0, 0);
        m_KF->predict();
        m_measurement.at<float>(0) = a_pose.x;
        m_measurement.at<float>(1) = a_pose.y;

        for(int i = 0; i < m_init_count_threshold; i++)
        {
            m_KF->correct(m_measurement);
            m_KF->predict();
        }
    }

    /**
     * @brief 根据新的二维坐标预测
     *
     * @param a_pose 当前二维坐标
     *
     * @return 返回预测后的二维坐标
     */
    cv::Point2f KF::predict(cv::Point2f a_pose)
    {
        cv::Mat prediction;

        if(!m_initialized)
        {
            std::cout << "Init" << std::endl;
            initFilter(a_pose);
            m_last_pose = a_pose;
            m_initialized = true;
        }

        m_measurement.at<float>(0) = a_pose.x;
        m_measurement.at<float>(1) = a_pose.y;
        m_KF->correct(m_measurement);
        prediction = m_KF->predict();
        if((std::fabs(m_last_pose.x - a_pose.x) > m_target_change_pose_threshold) || (std::fabs(m_last_pose.y - a_pose.y) > m_target_change_pose_threshold))
        {
            initFilter(a_pose);
            if(m_debug)
                std::cout << "Target has changed" << std::endl;
        }
        m_last_pose = a_pose;

        return cv::Point2f(prediction.at<float>(0) + m_predict_coe * prediction.at<float>(2),
                            prediction.at<float>(1) + m_predict_coe * prediction.at<float>(3));
    }

    /////////////// SINGLE ///////////////

    /**
     * @brief 单量滤波器初始化
     *
     * @param a_value 第一帧的值
     */
    void KF::initFilter(float a_value)
    {
        m_KF->statePost = (cv::Mat_<float>(m_state_num, 1) << a_value, 0, 0);
        m_KF->predict();
        m_measurement.at<float>(0) = a_value;

        for(int i = 0; i < m_init_count_threshold; i++)
        {
            m_KF->correct(m_measurement);
            m_KF->predict();
        }
    }

    /**
     * @brief 根据新的测量量滤波
     *
     * @param a_value 当前测量量
     *
     * @return 返回滤波后的量
     */
    float KF::predict(float a_value)
    {
        cv::Mat prediction;

        if(!m_initialized)
        {
            initFilter(a_value);
            m_last_value = a_value;
            m_initialized = true;
        }

        m_measurement.at<float>(0) = a_value;
        m_KF->correct(m_measurement);
        prediction = m_KF->predict();
        if(std::fabs(a_value - m_last_value) > m_value_diff)
        {
            initFilter(a_value);
            if(m_debug)
                std::cout << "Target has changed" << std::endl;
        }
        m_last_value = a_value;

        return prediction.at<float>(0) + m_predict_coe * prediction.at<float>(1);
    }

}
