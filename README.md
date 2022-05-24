# KF

### 类属性

```c++
    public:
        enum FILTER_MODE{IDLE, POINT, POSE, SINGLE}; // 滤波器模式
        FILTER_MODE m_filter_mod;					 // 当前滤波器模式
        int         m_measure_num;					 // 测量量维度
        int         m_state_num;					 // 状态量维度

        int         m_init_count_threshold;			 // 初始化阈值
        double       m_process_noise_cov;			 // 噪声协方差
        double       m_measure_noise_cov;			 // 测量协方差
        double        m_control_freq;					 // 控制频率 用于计算dt
        double      m_predict_coe;				 	 // 预测量增益时间 用于增大预测量
        cv::Mat     m_measurement;					 // 观测量矩阵
        std::shared_ptr<cv::KalmanFilter> m_KF; 	 // 滤波器

		// 三维
        cv::Point3f m_last_position;  				 // 上一帧坐标
        double        m_last_armor_dis;				 // 上一帧距离
        double        m_cur_armor_dis;				 // 当前帧距离
        double       m_cur_max_shoot_speed = 15.0;	 // 当前射速上限 不用实时射速避免预测量抖动
        double       m_target_change_dist_threshold;  // 目标切换阈值 距离
        
		// 二维
        wmj::GimbalPose  m_last_pose;					 // 上一帧位姿
        double            m_target_change_pose_threshold; // 目标切换阈值 位姿
        
		// 一维
        double        m_last_value; // 上一帧值
        double     m_value_diff; // 目标切换阈值 变量

    private:
        bool        m_initialized{false}; // 滤波器初始化标志
        int         m_debug;			  // 1为调试模式,非bool
```



### 主要接口

```c++
    Predict::Predict(wmj::Predict::FILTER_MODE a_filter_mode)
```

作用：

- 构造函数

参数：

- 滤波器模式

返回：

- a_filter_mode=POINT: 创建三维点预测器
- a_filter_mode=POSE: 预测二维点预测器
- a_filter_mode=SINGLE: 预测一维点预测器



```c++
    void Predict::setParam()
```

作用：

- 预测器的参数初始化，默认读取Predict.yaml中的参数
- 将滤波器状态恢复为未初始化
- 可以在创建滤波器后直接访问类变量进行参数修改（有修改需要的变量已经全部公有了）



```c++
    void Predict::initFilter(cv::Point3f a_position)
    void Predict::initFilter(wmj::GimbalPose a_pose)
    void Predict::initFilter(double a_value)
```

作用：

- 三种预测器的滤波器初始化，创建时或目标切换时重新初始化使滤波器第一帧收敛

参数：

- 对应滤波器的测量量



```c++
    cv::Point3f Predict::predict(cv::Point3f a_position)  
	wmj::GimbalPose Predict::predict(wmj::GimbalPose a_pose)
	float Predict::predict(double a_value)
```

作用：

- 预测主函数

参数：

- 对应滤波器的测量量

返回：

- 预测结果（三维点坐标/云台位姿/一维量）



### 主要流程

1. 初始化参数，根据测距精度(7m误差<10cm)，设置过程噪声`m_processNoiseCov`较小(1e-5)，测量噪声`m_measerementNoiseCov`适中(1e-3)；从参数文件`Predict.yaml`读取`m_init_count_threshold`第一帧预测次数，`m_predict_coe`预测数度增益量，`m_control_freq`控制频率；外部调用`setShootSpeed(float)`设置当前射速上限
2. 外部直接调用`predict()`获得预测后的装甲坐标，内部判断调用`initFilter()`初始化卡尔曼滤波器，并用第一帧输入的装甲绝对坐标进行多次预测使其收敛
3. 判断当前帧和上一帧装甲相对距离，根据设置的阈值(target_change_threshold=0.25m)判断装甲板是否切换，装甲板切换需重新初始化滤波器
4. 得到预测结果，击打远距离装甲板子弹飞行时间更长，不同射速等级子弹飞行时间不同，所以加入装甲距离和射速调节预测量大小

```c++
                return cv::Point3f(prediction.at<double>(0) + m_predict_coe * m_cur_armor_dis / m_cur_max_shoot_speed * prediction.at<double >(3),
                           prediction.at<double >(1) + m_predict_coe * m_cur_armor_dis / m_cur_max_shoot_speed * prediction.at<double>(4),
                           prediction.at<double >(2) + m_predict_coe * m_cur_armor_dis / m_cur_max_shoot_speed * prediction.at<double>(5));
```

### 调参方法

1. 测距精度达到要求，先调试PID参数，完成后开始调试预测
2. 提前测算自瞄循环的帧率，主要调试predict_coe预测量和control_freq控制频率
3. 对于较快速的线性运动，有时需要增大target_change_threshold避免滤波器重复初始化造成跟随过程中的抖动，但该阈值过大容易在装甲板切换时预测出错





# EKF

​	整个模块主要部分==套用自上交EKF代码，如果开源，提交之前注意以下==

### 基础学习

[bilibili-KF&EKF详细视频](https://space.bilibili.com/230105574/channel/detail?cid=139198&ctype=0)

### 类型定义

```c++
using MatrixXX = Eigen::Matrix<double , N_X, N_X>;
using MatrixZX = Eigen::Matrix<double , N_Z, N_X>;
using MatrixXZ = Eigen::Matrix<double , N_X, N_Z>;
using MatrixZZ = Eigen::Matrix<double , N_Z, N_Z>;
using VectorX  = Eigen::Matrix<double , N_X, 1>;
using VectorZ  = Eigen::Matrix<double , N_Z, 1>;
```

### 工具函数定义

```c++
// 定义预测其器的线性模型，并传递需要外部每帧更新的数据，如射速
struct PredictTool {
    // 距离上帧时长
    double dt;
    // 射速 采取当前射速上限 用真实射速会因为射速抖动而使预测量抖动
    double cur_shoot_speed;
    /*
     * 定义匀速直线运动模型
     * 状态x0经线性模型转换后得到新状态x1
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

// 便于进行三维坐标到球坐标的转换
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
```

### 类属性

```c++
int m_debug = 0;             // 开启调试模式 预测器的状态和一些中间量将输出

int m_init_threshold;			  // 初始化次数 让第一次预测收敛到正确的位置 100就够
double  m_target_change_threshold;  // 目标变化的阈值 两帧之间的距离差大于阈值即认为目标切换 重新初始化
double m_predict_time_coe;		  // 预测时间增益量 原始Xe一般比实际延迟一些 故使预测量在其速度上适当超前
double m_shoot_delay;              // 发射延迟 固定量

bool m_filer_initialized = false; // 滤波器是否初始化
double m_cur_distance;			  // 当前目标距离 用于与上一帧距离作差比较

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
VectorZ Z;           // 观测状态变量
VectorZ Zp;          // 预测观测量
MatrixXX P;          // 状态协方差
MatrixXX P_;         // 先验状态协方差
MatrixXX F;          // 预测雅可比
MatrixZX H;          // 观测雅可比
MatrixXZ K;          // 卡尔曼增益

VectorX Xr;          // 当前观测量
VectorZ Z_;          // 观测量转非线性量

PredictTool m_predict_tool;    // 预测传参、建模工具
CoorTransfromTool m_coor_tool; // 状态量与观测量转换工具
```

### 参数文件

```yaml
%YAML:1.0
---
debug: 1
init_threshold: 100            # 初始化滤波次数
target_change_threshold: 0.25  # 目标切换判定阈值
predict_time_coe: 0.001         # 预测时间增益 返回预测值较原始滤波值的延伸时长
shoot_delay: 0.001              # 发射延迟 秒为单位

Q00: 0.1
Q11: 10
Q22: 0.1
Q33: 10
Q44: 0.01
Q55: 1

R00: 1
R11: 1
R22: 10

# Q00 -> 预测量 x   ：用于调整对直线模型中 x 轴方向运动属于直线运动的的置信度
# Q11 -> 预测量 x_v ：用于调整 x 轴速度的波动量，减小效果：击打匀速目标不会一会超前一会落后；增大效果：更新速度快，例如变向
# Q22 -> 预测量 y   ：用于调整对直线模型中 y 轴方向运动属于直线运动的的置信度
# Q33 -> 预测量 y_v ：用于调整 y 轴速度的波动量，减小效果：击打匀速目标不会一会超前一会落后；增大效果：更新速度快，例如变向
# Q44 -> 预测量 z   ：用于调整对直线模型中 z 轴不变的置信度
# Q55 -> 预测量 y_v ：用于调整 y 轴速度的波动量，减小效果：击打匀速目标不会一会超前一会落后；增大效果：更新速度快，例如变向；y 轴（pitch）一般不用调整，很容易打中

# R00 -> 观测量 yaw     ：固定不动
# R11 -> 观测量 pitch   ：固定不动
# R22 -> 观测量 distance：用于调整距离的波动大小，一般距离的波动很大；调大，距离更新快
```

### 类函数

```c++
EKF() : P(MatrixXX::Identity()), Q(MatrixXX::Identity()), R(MatrixZZ::Identity());
```

作用：

- 构造函数，默认调用setParam()初始化参数

```c++
void setParam();
```

作用：

- 从`EKF.yaml`初始化参数Q、R矩阵和预测所用预测量增益、目标切换阈值之类
- 设为公有函数，在test文件里按键p就可以重新从文件中初始化参数，达到动态调参效果

```c++
void initFilter();
```

作用：

- 初始化滤波器
- 根据m_init_threshold进行重复滤波，使首帧预测值收敛

```c++
void Filter();
```

作用：

- 滤波器核心函数，根据传入的但前观测量Xr和上一帧估计量Xe进行滤波
- 使用ceres自动求导，设定状态转换方程，将变量用`ceres::Jet`类型表示并输入状态方程，得到自动求导结果，更新F预测雅可比和H观测雅可比

```c++
template <class Tool1, class Tool2>VectorZ predict(Tool1 tool1, Tool2 tool2, VectorZ a_Z);
```

作用：

- 预测主函数，调用Filter()进行滤波并对原始滤波结果加以处理，得到用于自瞄的预测量
- 为了便于扩展，这里传入的工具量形参均按序号命名，在函数中这些工具被赋给对应的类属性

返回：

- 预测量

### 工作流程

**自瞄状态机**：

```flow
st=>start: 自瞄状态
op1=>operation: 装甲板识别
cond1=>condition: 识别到？
op2=>operation: 创建PredictTool和CoorTool
op3=>operation: 记录两帧预测间的时间差dt
op4=>operation: 记录当前射速上限
op5=>operation: 将预测工具和转换工具以及当前装甲绝对坐标传入predict()
op6=>operation: 获取预测值转换成云台系坐标传给PID解算云台速度
op7=>operation: 底层控制类发送速度包

st->op1->cond1
cond1(yes)->op2->op3->op4->op5->op6->op7->op1
cond1(no)->op1
```

**预测器内部运行**：

```flow
st=>start: 获取Tool1、Tool2、VectorZ输入
op1=>operation: 更新工具变量、观测量Xr、目标距离
cond1=>condition: 滤波器是否初始化？

op2=>operation: 将Xe初始化为当前观测量，m_filter_initialized设为true
cond2=>condition: i++ < m_init_threshold?

op3=>operation: 上一帧估计值Xe赋给雅可比阵Xe_auto_jet
op4=>operation: 预测，调用工具函数通过线性模型预测先验结果存入Xp_auto_jet_p
op5=>operation: ceres自动求导更新预测雅可比F，得到先验预测量Xp
op6=>operation: 计算先验噪声协方差P_
op7=>operation: 调用工具函数将观测值Xr转换到Z_
op8=>operation: 将先验预测量Xp存入雅可比阵Xp_auto_jet_u
op9=>operation: 调用工具函数将Xp_auto_jet_u转换为Zp_auto_jet
op10=>operation: ceres自动求导更新观测雅可比H，得到预测观测量Zp
op11=>operation: 更新卡尔曼增益K、估计量Xe、噪声协方差P

cond3=>condition: fabs(m_cur_distance - Z_(2, 0)) > m_target_change_threshold

op12=>operation: 调用Filter()滤波
e=>end: 调整原始滤波结果返回预测值

st->op1->cond1
cond1(yes)->cond3
cond1(no)->op2->cond2
cond2(yes)->op3->op4->op5->op6->op7->op8->op9->op10->op11->cond2
cond2(no)->cond3
cond3(yes)->op2
cond3(no)->op12->e
```

