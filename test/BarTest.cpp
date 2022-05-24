// KF 一维输入数据滤波与可视化测试 支持命令行指定数据文件输入和无文件随机数展示两种运行方式
// 滤波过程中会在图中以红线为时间轴 蓝色点为输入值 绿色点为滤波后的值
// 可以使用滑动条动态调整相应参数 观察参数变化对滤波结果的影响
#include "../includeKF.hpp"
#include <fstream>
#include <unistd.h>

void changePNC(int, void*);
void changeMNC(int, void*);
void changePC(int, void*);
void changeCF(int, void*);
void changeVD(int, void*);

std::shared_ptr<predict::KF> kf = std::make_shared<predict::KF>(predict::KF::SINGLE);
// 滑动条只支持整数 按如下规则转换参数
int process_noise_cov;
int measure_noise_cov;
int predict_coe         = int(10.0 * kf->m_predict_coe);
int control_freq        = int(kf->m_control_freq);
int value_diff          = int(100.0 * kf->m_value_diff);

int x_lim = 10, y_lim = 10;
int width = 800, height = 800;
std::vector<float> src_x(width / x_lim), proc_x(width / x_lim);
float src_point, proc_point;

int main(int argc, char* argv[])
{
    std::ifstream ifs;
    if(argc != 2)
        std::cout << "Usage: ./BarTest [input file]\n";
    else
        ifs.open(argv[1]);
    
    srand((unsigned)time(NULL));

    cv::Mat img(width, height, CV_8UC3, cv::Scalar(255, 255, 255));

    // 用幂级数表示协方差矩阵元素值
    float t = kf->m_process_noise_cov;
    int count = 0;
    while(std::fabs(t - 1.0) > 0.01)
    {
        count ++;
        t *= 10.f;
    }
    process_noise_cov = count;
    t = kf->m_measure_noise_cov;
    count = 0;
    while(std::fabs(t - 1.0) > 0.01)
    {
        count ++;
        t *= 10.f;
    }
    measure_noise_cov = count;

    // 滑动条数值上限
    int process_noise_cov_v = 7;
    int measure_noise_cov_v = 7;
    int predict_coe_v       = (int(10.0 * kf->m_predict_coe) * 10 ? int(10.0 * kf->m_predict_coe) * 10 : 10); // 预测增益可能为0
    int control_freq_v      = int(kf->m_control_freq) * 100;
    int value_diff_v        = int(100.0 * kf->m_value_diff) * 10;
    std::cout << predict_coe << " " << predict_coe_v << "\n";

    int n = 1000;
    while((ifs.is_open() ? 1 : (n--)))
    {
        if(ifs.is_open())
            if(ifs >> src_point) ;
            else break;
        else
            src_point += std::pow(-1, n) * float(rand() % 100) / 100.0;
        
        std::cout << "src_point : " << src_point << std::endl;
        proc_point = kf->predict(src_point);
        std::cout << "proc_point: " << proc_point << std::endl;
        std::cout << "--------------------\n";

        src_x.erase(src_x.begin());
        proc_x.erase(proc_x.begin());
        src_x.push_back(src_point);
        proc_x.push_back(proc_point);

        img = cv::Mat(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::line(img, cv::Point(1, height / 2), cv::Point(width, height / 2), cv::Scalar(0, 0, 255), 2);
        for(int i = 0; i * x_lim < width; i++)
        {
            cv::circle(img, cv::Point(i * x_lim, height / 2 - 10 * src_x[i]), 2, cv::Scalar(255, 0, 0), 2);
            cv::circle(img, cv::Point(i * x_lim, height / 2 - 10 * proc_x[i]), 5, cv::Scalar(0, 255, 0), 2);
        }

        cv::namedWindow("KF-SINGLE", cv::WINDOW_FREERATIO);
        cv::createTrackbar("process_noise_cov", "KF-SINGLE", &process_noise_cov, process_noise_cov_v, changePNC);
        cv::createTrackbar("measure_noise_cov", "KF-SINGLE", &measure_noise_cov, measure_noise_cov_v, changeMNC);
        cv::createTrackbar("predict_coe 10*", "KF-SINGLE", &predict_coe, predict_coe_v, changePC);
        cv::createTrackbar("control_freq", "KF-SINGLE", &control_freq, control_freq_v, changeCF);
        cv::createTrackbar("value_diff 100*", "KF-SINGLE", &value_diff, value_diff_v, changeVD);
        cv::imshow("KF-SINGLE", img);
        char c = cv::waitKey(100);
        if(c == 'q' || c == 27)
            break;
    }

    if(argc == 2) ifs.close();
    return 0;
}

void changePNC(int, void*)
{
    kf->m_process_noise_cov = 1.f / float(std::pow(10, process_noise_cov));
    kf->initFilter(src_point);
}

void changeMNC(int, void*)
{
    kf->m_measure_noise_cov = 1.f / float(std::pow(10, measure_noise_cov));
    kf->initFilter(src_point);
}

void changePC(int, void*)
{
    kf->m_predict_coe = float(predict_coe) / 10.f;
    kf->initFilter(src_point);
}

void changeCF(int, void*)
{
    kf->m_control_freq = float(control_freq);
    kf->initFilter(src_point);
}

void changeVD(int, void*)
{
    kf->m_value_diff = float(value_diff) / 100.f;
    kf->initFilter(src_point);
}