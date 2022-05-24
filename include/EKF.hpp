#pragma once
#include <iostream>
#include <thread>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <ceres/jet.h>
#include <Eigen/Dense>

#define EKF_CFG "../EKF.yaml"

namespace predict
{
    struct PredictTool {
        // 距离上帧时长
        double dt;
        /*
         * 定义匀速直线运动模型
         */
        template <class T>
        void operator()(const T x0[6], T x1[6]) {
            x1[0] = x0[0] + dt * x0[1];
            x1[1] = x0[1];
            x1[2] = x0[2] + dt * x0[3];
            x1[3] = x0[3];
            x1[4] = x0[4] + dt * x0[5];
            x1[5] = x0[5];
        }
    };

    /*
     * 三维坐标转球坐标 xyz -> pyd
     */
    template<class T>
    void xyz2pyd(T xyz[3], T pyd[3]) {
        pyd[0] = ceres::atan2(xyz[2], ceres::sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1]));
        pyd[1] = ceres::atan2(xyz[1], xyz[0]);
        pyd[2] = ceres::sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]);
    }

    struct CoorTransfromTool {
        /*
         * 三维坐标转球坐标 xyz -> pyd
         */
        template<class T>
        void operator()(const T x[6], T z[3]) {
            T x_[3] = {x[0], x[2], x[4]};
            xyz2pyd(x_, z);
        }
    };

    template<int N_X, int N_Z> // 6 3 x vx y vy z vz  -> p y d
    class EKF{
        using MatrixXX = Eigen::Matrix<double, N_X, N_X>;
        using MatrixZX = Eigen::Matrix<double, N_Z, N_X>;
        using MatrixXZ = Eigen::Matrix<double, N_X, N_Z>;
        using MatrixZZ = Eigen::Matrix<double, N_Z, N_Z>;
        using VectorX  = Eigen::Matrix<double, N_X, 1>;
        using VectorZ  = Eigen::Matrix<double, N_Z, 1>;
        
        public:
            EKF() : P(MatrixXX::Identity()), Q(MatrixXX::Identity()), R(MatrixZZ::Identity())
            {
                setParam();
            }

            void setParam()
            {
                m_filer_initialized = false;
                cv::FileStorage fs(EKF_CFG, cv::FileStorage::READ);
                fs["debug"]                   >> m_debug;

                fs["init_threshold"]          >> m_init_threshold;
                fs["target_change_threshold"] >> m_target_change_threshold;
                fs["predict_time_coe"]        >> m_predict_time_coe;

                fs["Q00"]                     >> m_Q00;
                fs["Q11"]                     >> m_Q11;
                fs["Q22"]                     >> m_Q22;
                fs["Q33"]                     >> m_Q33;
                fs["Q44"]                     >> m_Q44;
                fs["Q55"]                     >> m_Q55;
                fs["R00"]                     >> m_R00;
                fs["R11"]                     >> m_R11;
                fs["R22"]                     >> m_R22;
                fs.release();

                Q << m_Q00, 0.f, 0.f, 0.f, 0.f, 0.f,
                     0.f, m_Q11, 0.f, 0.f, 0.f, 0.f,
                     0.f, 0.f, m_Q22, 0.f, 0.f, 0.f,
                     0.f, 0.f, 0.f, m_Q33, 0.f, 0.f,
                     0.f, 0.f, 0.f, 0.f, m_Q44, 0.f,
                     0.f, 0.f, 0.f, 0.f, 0.f, m_Q55;
                R << m_R00, 0.f, 0.f,
                     0.f, m_R11, 0.f,
                     0.f, 0.f, m_R22;
            }

            ~EKF() {}

            /**
             * @brief: 获取当前时间
             *
             * @param: null
             *
             * @return: double 时间戳
             */
            double now()
            {
                timeval tv;
                gettimeofday(&tv, NULL);
                return (double)tv.tv_sec + (double)tv.tv_usec / 1000000;
            }

            /**
             * @brief 预测函数 输入对应的工具和观测量返回预测量
             * 
             * @param tool1 预测工具
             * @param tool2 坐标转换工具
             * @param a_Z 观测量
             */
            template <class Tool1, class Tool2>
            VectorZ predict(Tool1 tool1, Tool2 tool2, VectorZ a_Z)
            {
                m_predict_tool = tool1;
                m_coor_tool = tool2;
                Xr << a_Z(0, 0), 0.0, a_Z(1, 0), 0.0, a_Z(2, 0), 0.0;
		        m_cur_distance = std::sqrt(a_Z(0, 0) * a_Z(0, 0) + a_Z(1, 0) * a_Z(1, 0) + a_Z(2, 0) * a_Z(2, 0));

                VectorZ prediction;

                if(!m_filer_initialized)
                {
                    m_filer_initialized = true;
                    if(m_debug)
                        std::cout << "Init" << std::endl;

                    initFilter();
                }

                if(std::fabs(m_cur_distance - Z_(2, 0)) > m_target_change_threshold)
                {
                    if(m_debug)
                        std::cout << "Target changed!!!" << std::endl;
                    initFilter();
                    prediction(0, 0) = Xe(0, 0);
                    prediction(1, 0) = Xe(2, 0);
                    prediction(2, 0) = Xe(4, 0);
                    return prediction;
                }

                Filter();
                prediction(0, 0) = Xe(0, 0) + m_predict_time_coe * Xe(1, 0);
                prediction(1, 0) = Xe(2, 0) + m_predict_time_coe * Xe(3, 0);
                prediction(2, 0) = Xe(4, 0) + m_predict_time_coe * Xe(5, 0);
                return prediction;
            }

            /**
             * @brief 预测函数 输入对应的工具和观测量返回预测量
             * 
             * @param a_abs_point 三维坐标
             * 
             */
            cv::Point3f predict(cv::Point3f a_abs_point)
            {
                // m_predict_tool更新
                m_predict_begin = now();
                cv::Point3f proc_point;
                m_predict_tool.dt = m_predict_begin - m_predict_end;
                if(m_predict_tool.dt > 1.0) m_predict_tool.dt = 0.0;
                m_predict_end = m_predict_begin;
                
                // 更新当前观测量
                Xr << a_abs_point.x, 0.0, a_abs_point.y, 0.0, a_abs_point.z, 0.0;
		        m_cur_distance = std::sqrt(a_abs_point.x * a_abs_point.x + a_abs_point.y * a_abs_point.y + a_abs_point.z * a_abs_point.z);

                cv::Point3f prediction;

                if(!m_filer_initialized)
                {
                    m_filer_initialized = true;
                    if(m_debug)
                        std::cout << "Init" << std::endl;

                    initFilter();
                }

                if(std::fabs(m_cur_distance - Z_(2, 0)) > m_target_change_threshold)
                {
                    if(m_debug)
                        std::cout << "Target changed!!!" << std::endl;
                    initFilter();
                    prediction.x = Xe(0, 0);
                    prediction.y = Xe(2, 0);
                    prediction.z = Xe(4, 0);
                    return prediction;
                }

                Filter();
                prediction.x = Xe(0, 0) + m_predict_time_coe * Xe(1, 0);
                prediction.y = Xe(2, 0) + m_predict_time_coe * Xe(3, 0);
                prediction.z = Xe(4, 0) + m_predict_time_coe * Xe(5, 0);

                return prediction;
            }

            /**
             * @brief 自定义预测过程协方差矩阵
             */
            void SetCovQ(double a_Q00, double a_Q11, double a_Q22, double a_Q33, double a_Q44, double a_Q55)
            {
                m_Q00 = a_Q00;
                m_Q11 = a_Q11;
                m_Q22 = a_Q22;
                m_Q33 = a_Q33;
                m_Q44 = a_Q44;
                m_Q55 = a_Q55;
            }

            /**
             * @brief 自定义观测过程协方差矩阵
             */
            void SetCovR(double a_R00, double a_R11, double a_R22)
            {
                m_R00 = a_R00;
                m_R11 = a_R11;
                m_R22 = a_R22;
            }

        private:
            int m_debug = 0;

            int m_init_threshold;
            double m_target_change_threshold;
            double m_predict_time_coe;

            bool m_filer_initialized = false;
            float m_cur_distance;

            double m_Q00;
            double m_Q11;
            double m_Q22;
            double m_Q33;
            double m_Q44;
            double m_Q55;
            double m_R00;
            double m_R11;
            double m_R22;
            MatrixXX Q;          // 预测过程协方差
            MatrixZZ R;          // 观测过程协方差
            VectorX Xp;          // 预测状态变量
            VectorX Xe;          // 估计状态变量
            VectorX Xr;          // 当前观测量
            VectorZ Zp;          // 预测观测量
            MatrixXX P;          // 状态协方差
            MatrixXX P_;         // 先验状态协方差
            MatrixXX F;          // 预测雅可比
            MatrixZX H;          // 观测雅可比
            MatrixXZ K;          // 卡尔曼增益

            VectorZ Z_;          // 观测量转非线性量

            PredictTool m_predict_tool;
            CoorTransfromTool m_coor_tool;
            double m_predict_begin = 0.0, m_predict_end = 0.0;

            void initFilter()
            {
                if(m_debug)
                {
                    std::cout << "init position : [" << Xr(0, 0) << "," << Xr(2, 0) << "," << Xr(4, 0) << "]\n";
                }
                Xe = Xr;

                for(int i = 0; i < m_init_threshold; i++)
                {
                    Filter();
                }
            }

            /**
             * @brief EKF核心算法
             */
            void Filter()
            {
                // predict
                ceres::Jet<double, N_X> Xe_auto_jet[N_X];
                for(int i = 0; i < N_X; i++) {
                    Xe_auto_jet[i].a = Xe[i];
                    Xe_auto_jet[i].v[i] = 1;
                }
                ceres::Jet<double, N_X> Xp_auto_jet_p[N_X];
                m_predict_tool(Xe_auto_jet, Xp_auto_jet_p);
                for(int i = 0; i < N_X; i++) {
                    Xp[i] = Xp_auto_jet_p[i].a;
                    F.block(i, 0, 1, N_X) = Xp_auto_jet_p[i].v.transpose();
                }
                P_ = F * P * F.transpose() + Q;

                // update
                m_coor_tool(Xr.data(), Z_.data());
                ceres::Jet<double, N_X> Xp_auto_jet_u[N_X];
                for(int i = 0; i < N_X; i++) {
                    Xp_auto_jet_u[i].a = Xp[i];
                    Xp_auto_jet_u[i].v[i] = 1;
                }
                ceres::Jet<double, N_X> Zp_auto_jet[N_Z];
                m_coor_tool(Xp_auto_jet_u, Zp_auto_jet);
                for(int i = 0; i < N_Z; i++) {
                    Zp[i] = Zp_auto_jet[i].a;
                    H.block(i, 0, 1, N_X) = Zp_auto_jet[i].v.transpose();
                }
                K = P_ * H.transpose() * (H * P_ * H.transpose() + R).inverse();
                Xe = Xp + K * (Z_ - Zp);
                P = (MatrixXX::Identity() - K * H) * P_;

                if(m_debug)
                {
                    std::cout << "measure position : [" << Xr(0, 0) << "," << Xr(2, 0) << "," << Xr(4, 0) << "]\n";
                    std::cout << "Xe : \n" << Xe << "\n------------------------------------\n";
                }
            }
    };
}
