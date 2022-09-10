#include <iostream>
#include <vector>
#include <utility>

#include <algorithm>
#include <mutex>
#include <atomic>
#include <queue>
#include <chrono>
#include <sstream>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>
#include <ngraph/ngraph.hpp>

#include <monitors/presenter.h>
//#include <utils/slog.hpp>

#include "input.hpp"
//#include "multichannel_params.hpp"
//#include "multichannel_object_detection_demo_yolov3_params.hpp"
#include "output.hpp"
#include "threading.hpp"
#include "graph.hpp"

// static const char thresh_output_message[] = "Optional. Probability threshold for detections";

// DEFINE_double(t, 0.5, thresh_output_message);

std::string modelPath = "/home/slava/Source/YoloV3_OVN_Inference/mo_converted/yolo-v3-tf.xml";
std::string videoPath = "input.avi";
//float treshhold = 0.7;

int FLAGS_bs = 1;
int FLAGS_nireq = 1;
int FLAGS_show_stats = 0;
int FLAGS_pc = 0;
std::string FLAGS_d = "GPU";

std::string FLAGS_i = videoPath;
int FLAGS_loop = 0;
int FLAGS_n_iqs = 1;
//FLAGS_show_stats = 0;
int FLAGS_real_input_fps = 10;

int FLAGS_duplicate_num = 1;
float FLAGS_t = 0.7;
bool FLAGS_no_show = true;
int FLAGS_u = 1;
int FLAGS_fps_sp = 1;
int FLAGS_n_sp = 1;


static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry) {
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

class YoloParams {
    template <typename T>
    void computeAnchors(const std::vector<float> & initialAnchors, const std::vector<T> & mask) {
        anchors.resize(num * 2);
        for (int i = 0; i < num; ++i) {
            anchors[i * 2] = initialAnchors[mask[i] * 2];
            anchors[i * 2 + 1] = initialAnchors[mask[i] * 2 + 1];
        }
    }

	public:
	    int num = 0, classes = 0, coords = 0;
	    std::vector<float> anchors;

	    YoloParams() {}

	    YoloParams(const std::shared_ptr<ngraph::op::RegionYolo> regionYolo) {
	        coords = regionYolo->get_num_coords();
	        classes = regionYolo->get_num_classes();
	        auto initialAnchors = regionYolo->get_anchors();
 	    	auto mask = regionYolo->get_mask();
 	    	num = mask.size();

	        computeAnchors(initialAnchors, mask);
	    }
};

struct DetectionObject {
    int xmin, ymin, xmax, ymax, class_id;
    float confidence;

    DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale) :
        xmin{static_cast<int>((x - w / 2) * w_scale)},
        ymin{static_cast<int>((y - h / 2) * h_scale)},
        xmax{static_cast<int>(this->xmin + w * w_scale)},
        ymax{static_cast<int>(this->ymin + h * h_scale)},
        class_id{class_id},
        confidence{confidence} {}

    bool operator <(const DetectionObject &s2) const {
        return this->confidence < s2.confidence;
    }
    bool operator >(const DetectionObject &s2) const {
        return this->confidence > s2.confidence;
    }
};

double IntersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2) {
    double width_of_overlap_area = fmin(box_1.xmax, box_2.xmax) - fmax(box_1.xmin, box_2.xmin);
    double height_of_overlap_area = fmin(box_1.ymax, box_2.ymax) - fmax(box_1.ymin, box_2.ymin);
    double area_of_overlap;
    if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
        area_of_overlap = 0;
    else
        area_of_overlap = width_of_overlap_area * height_of_overlap_area;
    double box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin);
    double box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin);
    double area_of_union = box_1_area + box_2_area - area_of_overlap;
    return area_of_overlap / area_of_union;
}

void parseYOLOOutput(InferenceEngine::InferRequest::Ptr req,
                     const std::string &outputName,
                     const YoloParams &yoloParams, const unsigned long resized_im_h,
                     const unsigned long resized_im_w, const unsigned long original_im_h,
                     const unsigned long original_im_w,
                     const double threshold, std::vector<DetectionObject> &objects) {
    InferenceEngine::Blob::Ptr blob = req->GetBlob(outputName);

    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
    if (out_blob_h != out_blob_w)
        throw std::runtime_error("Invalid size of output. It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
        ", current W = " + std::to_string(out_blob_h));

    auto num = yoloParams.num;
    auto coords = yoloParams.coords;
    auto classes = yoloParams.classes;

    auto anchors = yoloParams.anchors;

    auto side = out_blob_h;
    auto side_square = side * side;
    InferenceEngine::LockedMemory<const void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap();
    const float *output_blob  = blobMapped.as<float *>();
    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < side_square; ++i) {
        int row = i / side;
        int col = i % side;
        for (int n = 0; n < num; ++n) {
            int obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords);
            int box_index = EntryIndex(side, coords, classes, n * side * side + i, 0);
            float scale = output_blob[obj_index];
            if (scale < threshold)
                continue;
            double x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w;
            double y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h;
            double height = std::exp(output_blob[box_index + 3 * side_square]) * anchors[2 * n + 1];
            double width = std::exp(output_blob[box_index + 2 * side_square]) * anchors[2 * n];
            for (int j = 0; j < classes; ++j) {
                int class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j);
                float prob = scale * output_blob[class_index];
                if (prob < threshold)
                    continue;
                DetectionObject obj(x, y, height, width, j, prob,
                        static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                        static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                objects.push_back(obj);
            }
        }
    }
}

void drawDetections(cv::Mat& img, const std::vector<DetectionObject>& detections, const std::vector<cv::Scalar>& colors) {
    for (const DetectionObject& f : detections) {
        cv::rectangle(img,
                      cv::Rect2f(static_cast<float>(f.xmin),
                                 static_cast<float>(f.ymin),
                                 static_cast<float>((f.xmax-f.xmin)),
                                 static_cast<float>((f.ymax-f.ymin))),
                      colors[static_cast<int>(f.class_id)],
                      2);
    }
}

const size_t DISP_WIDTH  = 1920;
const size_t DISP_HEIGHT = 1080;
const size_t MAX_INPUTS  = 25;

struct DisplayParams {
    std::string name;
    cv::Size windowSize;
    cv::Size frameSize;
    size_t count;
    cv::Point points[MAX_INPUTS];
};

DisplayParams prepareDisplayParams(size_t count) {
    DisplayParams params;
    params.count = count;
    params.windowSize = cv::Size(DISP_WIDTH, DISP_HEIGHT);

    size_t gridCount = static_cast<size_t>(ceil(sqrt(count)));
    size_t gridStepX = static_cast<size_t>(DISP_WIDTH/gridCount);
    size_t gridStepY = static_cast<size_t>(DISP_HEIGHT/gridCount);
    if (gridStepX == 0 || gridStepY == 0) {
        throw std::logic_error("Can't display every input: there are too many of them");
    }
    params.frameSize = cv::Size(gridStepX, gridStepY);

    for (size_t i = 0; i < count; i++) {
        cv::Point p;
        p.x = gridStepX * (i/gridCount);
        p.y = gridStepY * (i%gridCount);
        params.points[i] = p;
    }
    return params;
}

std::map<std::string, YoloParams> GetYoloParams(const std::vector<std::string>& outputDataBlobNames, InferenceEngine::CNNNetwork &network) {
    std::map<std::string, YoloParams> __yoloParams;

    for (auto &output_name : outputDataBlobNames) {
        YoloParams params;

        if (auto ngraphFunction = network.getFunction()) {
            for (const auto op : ngraphFunction->get_ops()) {
                if (op->get_friendly_name() == output_name) {
                    auto regionYolo = std::dynamic_pointer_cast<ngraph::op::RegionYolo>(op);
                    if (!regionYolo) {
                        throw std::runtime_error("Invalid output type: " +
                            std::string(regionYolo->get_type_info().name) + ". RegionYolo expected");
                    }

                    params = regionYolo;
                    break;
                }
            }
        } else {
            throw std::runtime_error("Can't get ngraph::Function. Make sure the provided model is in IR version 10 or greater.");
        }
        __yoloParams.insert(std::pair<std::string, YoloParams>(output_name.c_str(), params));
    }

    return __yoloParams;
}

void displayNSources(const std::vector<std::shared_ptr<VideoFrame>>& data,
                     float time,
                     const std::string& stats,
                     const DisplayParams& params,
                     const std::vector<cv::Scalar> &colors
                     //Presenter& presenter
                    ){
    cv::Mat windowImage = cv::Mat::zeros(params.windowSize, CV_8UC3);
    auto loopBody = [&](size_t i) {
        auto& elem = data[i];
        if (!elem->frame.empty()) {
            cv::Rect rectFrame = cv::Rect(params.points[i], params.frameSize);
            cv::Mat windowPart = windowImage(rectFrame);
            cv::resize(elem->frame, windowPart, params.frameSize);
            drawDetections(windowPart, elem->detections.get<std::vector<DetectionObject>>(), colors);
        }
    };

    auto drawStats = [&]() {
        if (FLAGS_show_stats && !stats.empty()) {
            static const cv::Point posPoint = cv::Point(3*DISP_WIDTH/4, 4*DISP_HEIGHT/5);
            auto pos = posPoint + cv::Point(0, 25);
            size_t currPos = 0;
            while (true) {
                auto newPos = stats.find('\n', currPos);
                cv::putText(windowImage, stats.substr(currPos, newPos - currPos), pos, cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 0.8,  cv::Scalar(0, 0, 255), 1);
                if (newPos == std::string::npos) {
                    break;
                }
                pos += cv::Point(0, 25);
                currPos = newPos + 1;
            }
        }
    };

    for (size_t i = 0; i < data.size(); ++i) {
        loopBody(i);
    }

    //presenter.drawGraphs(windowImage);
    drawStats();

    char str[256];
    snprintf(str, sizeof(str), "%5.2f fps", static_cast<double>(1000.0f/time));
    cv::putText(windowImage, str, cv::Point(800, 100), cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 2.0,  cv::Scalar(0, 255, 0), 2);
    cv::imshow(params.name, windowImage);
}

int main(){
	std::cout << "InferenceEngine" << std::endl;

    std::size_t found = modelPath.find_last_of(".");
    if (found > modelPath.size()) {
        std::cout << "Invalid model name: " << modelPath << std::endl;
        std::cout << "Expected to be <model_name>.xml" << std::endl;
        return -1;
    }

    std::cout << "Model   path: " << modelPath << std::endl;

    std::map<std::string, YoloParams> yoloParams;

    IEGraph::InitParams graphParams;
    // graphParams.batchSize       = 1;
    // graphParams.maxRequests     = 1;
    // graphParams.collectStats    = 0;
    // graphParams.reportPerf      = 0;
    // graphParams.modelPath       = modelPath;
    // // graphParams.cpuExtPath      = '';
    // // graphParams.cldnnConfigPath = '';
    // graphParams.deviceName      = "GPU";
    graphParams.batchSize       = FLAGS_bs;
    graphParams.maxRequests     = FLAGS_nireq;
    graphParams.collectStats    = FLAGS_show_stats;
    graphParams.reportPerf      = FLAGS_pc;
    graphParams.modelPath       = modelPath;
    // graphParams.cpuExtPath      = FLAGS_l;
    // graphParams.cldnnConfigPath = FLAGS_c;
    graphParams.deviceName      = FLAGS_d;
    graphParams.postLoadFunc    = [&yoloParams](const std::vector<std::string>& outputDataBlobNames,
                                                InferenceEngine::CNNNetwork &network) {
                                                    yoloParams = GetYoloParams(outputDataBlobNames, network);
                                                };

    std::shared_ptr<IEGraph> network(new IEGraph(graphParams));
    auto inputDims = network->getInputDims();
    if (4 != inputDims.size()) {
        throw std::runtime_error("Invalid network input dimensions");
    }

    VideoSources::InitParams vsParams;
    // vsParams.inputs               = videoPath;
    // vsParams.loop                 = 0;
    // vsParams.queueSize            = 1;
    // vsParams.collectStats         = 0;
    // vsParams.realFps              = 10;
    vsParams.inputs               = FLAGS_i;
    vsParams.loop                 = FLAGS_loop;
    vsParams.queueSize            = FLAGS_n_iqs;
    vsParams.collectStats         = FLAGS_show_stats;
    vsParams.realFps              = FLAGS_real_input_fps;
    vsParams.expectedHeight = static_cast<unsigned>(inputDims[2]);
    vsParams.expectedWidth  = static_cast<unsigned>(inputDims[3]);

    VideoSources sources(vsParams);
    DisplayParams params = prepareDisplayParams(sources.numberOfInputs() * FLAGS_duplicate_num);
    sources.start();

    size_t currentFrame = 0;

    std::vector<cv::Scalar> colors;
    if (yoloParams.size() > 0)
        for (int i = 0; i < static_cast<int>(yoloParams.begin()->second.classes); ++i)
            colors.push_back(cv::Scalar(rand() % 256, rand() % 256, rand() % 256));

    network->start([&](VideoFrame& img) {
        img.sourceIdx = currentFrame;
        size_t camIdx = currentFrame / FLAGS_duplicate_num;
        currentFrame = (currentFrame + 1) % (sources.numberOfInputs() * FLAGS_duplicate_num);
        return sources.getFrame(camIdx, img);
    }, [&yoloParams](InferenceEngine::InferRequest::Ptr req,
            const std::vector<std::string>& outputDataBlobNames,
            cv::Size frameSize
            ) {
        unsigned long resized_im_h = 416;
        unsigned long resized_im_w = 416;

        std::vector<DetectionObject> objects;
        // Parsing outputs
        for (auto &output_name :outputDataBlobNames) {
            parseYOLOOutput(req, output_name, yoloParams[output_name], resized_im_h, resized_im_w, frameSize.height, frameSize.width, FLAGS_t, objects);
        }
        // Filtering overlapping boxes and lower confidence object
        std::sort(objects.begin(), objects.end(), std::greater<DetectionObject>());
        for (size_t i = 0; i < objects.size(); ++i) {
            if (objects[i].confidence == 0)
                continue;
            for (size_t j = i + 1; j < objects.size(); ++j)
                if (IntersectionOverUnion(objects[i], objects[j]) >= 0.4)
                    objects[j].confidence = 0;
        }

        std::vector<Detections> detections(1);
        detections[0].set(new std::vector<DetectionObject>);

        for (auto &object : objects) {
            if (object.confidence < FLAGS_t)
                continue;
            detections[0].get<std::vector<DetectionObject>>().push_back(object);
        }

        return detections;
    });

    network->setDetectionConfidence(static_cast<float>(FLAGS_t));

    std::atomic<float> averageFps = {0.0f};

    std::vector<std::shared_ptr<VideoFrame>> batchRes;

    std::mutex statMutex;
    std::stringstream statStream;

    std::cout << "To close the application, press 'CTRL+C' here";
    if (!FLAGS_no_show) {
        std::cout << " or switch to the output window and press ESC key";
    }
    std::cout << std::endl;

    cv::Size graphSize{static_cast<int>(params.windowSize.width / 4), 60};
    // Presenter presenter(FLAGS_u, params.windowSize.height - graphSize.height - 10, graphSize);

    // const size_t outputQueueSize = 1;
    // AsyncOutput output(FLAGS_show_stats, outputQueueSize,
    // [&](const std::vector<std::shared_ptr<VideoFrame>>& result) {
    //     std::string str;
    //     if (FLAGS_show_stats) {
    //         std::unique_lock<std::mutex> lock(statMutex);
    //         str = statStream.str();
    //     }
    //     displayNSources(result, averageFps, str, params, colors, presenter);
    //     int key = cv::waitKey(1);
    //     presenter.handleKey(key);

    //     return (key != 27);
    // });

    // output.start();

    using timer = std::chrono::high_resolution_clock;
    using duration = std::chrono::duration<float, std::milli>;
    timer::time_point lastTime = timer::now();
    duration samplingTimeout(FLAGS_fps_sp);

    size_t fpsCounter = 0;

    size_t perfItersCounter = 0;

    while (sources.isRunning() || network->isRunning()) {
        bool readData = true;
        while (readData) {
            auto br = network->getBatchData(params.frameSize);
            if (br.empty()) {
                break;
            }
            for (size_t i = 0; i < br.size(); i++) {
                auto val = static_cast<unsigned int>(br[i]->sourceIdx);
                auto it = find_if(batchRes.begin(), batchRes.end(), [val] (const std::shared_ptr<VideoFrame>& vf) { return vf->sourceIdx == val; } );
                if (it != batchRes.end()) {
                    if (!FLAGS_no_show) {
                        //output.push(std::move(batchRes));
                    }
                    batchRes.clear();
                    readData = false;
                }
                batchRes.push_back(std::move(br[i]));
            }
        }
        ++fpsCounter;

        // if (!output.isAlive()) {
        //     break;
        // }

        auto currTime = timer::now();
        auto deltaTime = (currTime - lastTime);
        if (deltaTime >= samplingTimeout) {
            auto durMsec =
                    std::chrono::duration_cast<duration>(deltaTime).count();
            auto frameTime = durMsec / static_cast<float>(fpsCounter);
            fpsCounter = 0;
            lastTime = currTime;

            if (FLAGS_no_show) {
                //slog::info << "Average Throughput : " << 1000.f/frameTime << " fps" << slog::endl;
                if (++perfItersCounter >= FLAGS_n_sp) {
                    break;
                }
            } else {
                averageFps = frameTime;
            }
        }

        network.reset();
    }
}


// }
// int main(int argc, char* argv[]) {
//     try {

//         // slog::info << "InferenceEngine: "
//         //     << printable(*InferenceEngine::GetInferenceEngineVersion()) << slog::endl;

//         // // ------------------------------ Parsing and validation of input args ---------------------------------
//         // if (!ParseAndCheckCommandLine(argc, argv)) {
//         //     return 0;
//         // }

//         std::string modelPath = "/home/slava/Source/YoloV3_OVN_Inference/mo_converted/yolo-v3-tf.xml";
//         std::string videoPath = "input.avi";
//         float treshhold = 0.7;

//         std::size_t found = modelPath.find_last_of(".");
//         if (found > modelPath.size()) {
//             slog::info << "Invalid model name: " << modelPath << slog::endl;
//             slog::info << "Expected to be <model_name>.xml" << slog::endl;
//             return -1;
//         }
//         //slog::info << "Model   path: " << modelPath << slog::endl;

//         std::map<std::string, YoloParams> yoloParams;

//         IEGraph::InitParams graphParams;
//         graphParams.batchSize       = 1;
//         graphParams.maxRequests     = 1;
//         graphParams.modelPath       = modelPath;
//         graphParams.deviceName      = "GPU";
//         graphParams.postLoadFunc    = [&yoloParams](const std::vector<std::string>& outputDataBlobNames,
//                                                     InferenceEngine::CNNNetwork &network) {
//                                                         yoloParams = GetYoloParams(outputDataBlobNames, network);
//                                                     };

//         std::shared_ptr<IEGraph> network(new IEGraph(graphParams));
//         auto inputDims = network->getInputDims();
//         if (4 != inputDims.size()) {
//             throw std::runtime_error("Invalid network input dimensions");
//         }

//         VideoSources::InitParams vsParams;
//         vsParams.inputs               = videoPath;
//         vsParams.realFps              = 10;
//         vsParams.expectedHeight = static_cast<unsigned>(inputDims[2]);
//         vsParams.expectedWidth  = static_cast<unsigned>(inputDims[3]);

//         VideoSources sources(vsParams);
//         DisplayParams params = prepareDisplayParams(sources.numberOfInputs() * 1);
//         sources.start();

//         size_t currentFrame = 0;

//         std::vector<cv::Scalar> colors;
//         if (yoloParams.size() > 0)
//             for (int i = 0; i < static_cast<int>(yoloParams.begin()->second.classes); ++i)
//                 colors.push_back(cv::Scalar(rand() % 256, rand() % 256, rand() % 256));

//         network->start([&](VideoFrame& img) {
//             img.sourceIdx = currentFrame;
//             size_t camIdx = currentFrame / FLAGS_duplicate_num;
//             currentFrame = (currentFrame + 1) % (sources.numberOfInputs() * FLAGS_duplicate_num);
//             return sources.getFrame(camIdx, img);
//         }, [&yoloParams](InferenceEngine::InferRequest::Ptr req,
//                 const std::vector<std::string>& outputDataBlobNames,
//                 cv::Size frameSize
//                 ) {
//             unsigned long resized_im_h = 416;
//             unsigned long resized_im_w = 416;

//             std::vector<DetectionObject> objects;
//             // Parsing outputs
//             for (auto &output_name :outputDataBlobNames) {
//                 parseYOLOOutput(req, output_name, yoloParams[output_name], resized_im_h, resized_im_w, frameSize.height, frameSize.width, treshhold, objects);
//             }
//             // Filtering overlapping boxes and lower confidence object
//             std::sort(objects.begin(), objects.end(), std::greater<DetectionObject>());
//             for (size_t i = 0; i < objects.size(); ++i) {
//                 if (objects[i].confidence == 0)
//                     continue;
//                 for (size_t j = i + 1; j < objects.size(); ++j)
//                     if (IntersectionOverUnion(objects[i], objects[j]) >= 0.4)
//                         objects[j].confidence = 0;
//             }

//             std::vector<Detections> detections(1);
//             detections[0].set(new std::vector<DetectionObject>);

//             for (auto &object : objects) {
//                 if (object.confidence < FLAGS_t)
//                     continue;
//                 detections[0].get<std::vector<DetectionObject>>().push_back(object);
//             }

//             return detections;
//         });

//         network->setDetectionConfidence(static_cast<float>(FLAGS_t));

//         std::atomic<float> averageFps = {0.0f};

//         std::vector<std::shared_ptr<VideoFrame>> batchRes;

//         std::mutex statMutex;
//         std::stringstream statStream;

//         std::cout << "To close the application, press 'CTRL+C' here";
//         if (!FLAGS_no_show) {
//             std::cout << " or switch to the output window and press ESC key";
//         }
//         std::cout << std::endl;

//         cv::Size graphSize{static_cast<int>(params.windowSize.width / 4), 60};
//         Presenter presenter(FLAGS_u, params.windowSize.height - graphSize.height - 10, graphSize);

//         const size_t outputQueueSize = 1;
//         AsyncOutput output(FLAGS_show_stats, outputQueueSize,
//         [&](const std::vector<std::shared_ptr<VideoFrame>>& result) {
//             std::string str;
//             if (FLAGS_show_stats) {
//                 std::unique_lock<std::mutex> lock(statMutex);
//                 str = statStream.str();
//             }
//             displayNSources(result, averageFps, str, params, colors, presenter);
//             int key = cv::waitKey(1);
//             presenter.handleKey(key);

//             return (key != 27);
//         });

//         output.start();

//         using timer = std::chrono::high_resolution_clock;
//         using duration = std::chrono::duration<float, std::milli>;
//         timer::time_point lastTime = timer::now();
//         duration samplingTimeout(FLAGS_fps_sp);

//         size_t fpsCounter = 0;

//         size_t perfItersCounter = 0;

//         while (sources.isRunning() || network->isRunning()) {
//             bool readData = true;
//             while (readData) {
//                 auto br = network->getBatchData(params.frameSize);
//                 if (br.empty()) {
//                     break;
//                 }
//                 for (size_t i = 0; i < br.size(); i++) {
//                     auto val = static_cast<unsigned int>(br[i]->sourceIdx);
//                     auto it = find_if(batchRes.begin(), batchRes.end(), [val] (const std::shared_ptr<VideoFrame>& vf) { return vf->sourceIdx == val; } );
//                     if (it != batchRes.end()) {
//                         if (!FLAGS_no_show) {
//                             output.push(std::move(batchRes));
//                         }
//                         batchRes.clear();
//                         readData = false;
//                     }
//                     batchRes.push_back(std::move(br[i]));
//                 }
//             }
//             ++fpsCounter;

//             if (!output.isAlive()) {
//                 break;
//             }

//             auto currTime = timer::now();
//             auto deltaTime = (currTime - lastTime);
//             if (deltaTime >= samplingTimeout) {
//                 auto durMsec =
//                         std::chrono::duration_cast<duration>(deltaTime).count();
//                 auto frameTime = durMsec / static_cast<float>(fpsCounter);
//                 fpsCounter = 0;
//                 lastTime = currTime;

//                 if (false) {
//                     //slog::info << "Average Throughput : " << 1000.f/frameTime << " fps" << slog::endl;
//                     if (++perfItersCounter >= FLAGS_n_sp) {
//                         break;
//                     }
//                 } else {
//                     averageFps = frameTime;
//                 }

//                 if (false) {
//                     auto inputStat = sources.getStats();
//                     auto inferStat = network->getStats();
//                     auto outputStat = output.getStats();

//                     std::unique_lock<std::mutex> lock(statMutex);
//                     statStream.str(std::string());
//                     statStream << std::fixed << std::setprecision(1);
//                     statStream << "Input reads: ";
//                     for (size_t i = 0; i < inputStat.readTimes.size(); ++i) {
//                         if (0 == (i % 4)) {
//                             statStream << std::endl;
//                         }
//                         statStream << inputStat.readTimes[i] << "ms ";
//                     }
//                     statStream << std::endl;
//                     statStream << "HW decoding latency: "
//                                << inputStat.decodingLatency << "ms";
//                     statStream << std::endl;
//                     statStream << "Preprocess time: "
//                                << inferStat.preprocessTime << "ms";
//                     statStream << std::endl;
//                     statStream << "Plugin latency: "
//                                << inferStat.inferTime << "ms";
//                     statStream << std::endl;

//                     statStream << "Render time: " << outputStat.renderTime
//                                << "ms" << std::endl;

//                     if (FLAGS_no_show) {
//                         slog::info << statStream.str() << slog::endl;
//                     }
//                 }
//             }
//         }

//         network.reset();

//         std::cout << presenter.reportMeans() << '\n';
//     }
//     catch (const std::exception& error) {
//         slog::err << error.what() << slog::endl;
//         return 1;
//     }
//     catch (...) {
//         slog::err << "Unknown/internal exception happened." << slog::endl;
//         return 1;
//     }

//     slog::info << "Execution successful" << slog::endl;
//     return 0;
// }








// int main(){
// 	std::string modelPath = "/home/slava/Source/YoloV3_OVN_Inference/mo_converted/yolo-v3-tf.xml";
// 	std::string videoPath = "input.avi";
// 	std::string device = "GPU"

// 	cout << "Model name = " << modelPath << endl;
// 	cout << "starting" << endl;

// 	// --------------------------- 1. Load inference engine -------------------------------------
// 	cout << "Creating Inference Engine" << endl;

// 	Core ie;
// 	// ------------------------------------------------------------------------------------------


// 	// --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
// 	cout << "Loading network files" << endl;

// 	/** Read network model **/
// 	auto network = ie.ReadNetwork(modelPath);
// 	cout << "network layer count: " << network.layerCount() << endl;
// 	// -----------------------------------------------------------------------------------------------------

// 	// --------------------------- 3. Configure input & output ---------------------------------------------

// 	// --------------------------- Prepare input blobs -----------------------------------------------------
// 	cout << "Preparing input blobs" << endl;

// 	// Taking information about all topology inputs
// 	InputsDataMap inputInfo(network.getInputsInfo());
// 	if (inputInfo.size() != 1){
// 		throw std::logic_error("Sample supports topologies with 1 input only");
// 	}

// 	auto inputInfoItem = *inputInfo.begin();

// 	inputInfoItem.second->setPrecision(Precision::U8);//U8 or FP16 or FP32 original is U8
// 	inputInfoItem.second->setLayout(Layout::NCHW);

// 	network.setBatchSize(1);
// 	size_t batchSize = network.getBatchSize();
// 	cout << "Batch size is " << std::to_string(batchSize) << endl;
// 	// -----------------------------------------------------------------------------------------------------


// 	// --------------------------- 4. Loading model to the device ------------------------------------------
// 	cout << "Loading model to the device: " << device << endl;
// 	auto executable_network = ie.LoadNetwork(network, device);
// 	// -----------------------------------------------------------------------------------------------------


// 	// --------------------------- 5. Create infer request -------------------------------------------------
// 	cout << "Create infer request" << endl;
// 	InferRequest inferRequest_regular = executable_network.CreateInferRequest();
// 	// -----------------------------------------------------------------------------------------------------

// 	VideoWriter video("V3outcpp.avi",cv::VideoWriter::fourcc('M','J','P','G'),10, cv::Size(inputInfoItem.second->getTensorDesc().getDims()[3], inputInfoItem.second->getTensorDesc().getDims()[2]));

// 	//open the video file for reading
// 	VideoCapture cap(videoPath); 

// 	// if not success, exit program
// 	if (cap.isOpened() == false)  
// 	{
// 		cout << "Cannot open the video file" << endl;
// 		cin.get(); //wait for any key press
// 		return -1;
// 	}

// 	while (true){
// 		clock_t start = std::clock();

// 		cv::Mat image;
// 		bool bSuccess = cap.read(image); // read a new frame from video 
// 		cv::resize(image, image, cv::Size(inputInfoItem.second->getTensorDesc().getDims()[3], inputInfoItem.second->getTensorDesc().getDims()[2]), 0, 0, cv::INTER_AREA);

// 		//Breaking the while loop at the end of the video
// 		if (bSuccess == false) 
// 		{
// 			cout << "Found the end of the video" << endl;
// 			break;
// 		}

// 		// --------------------------- 6. Prepare input --------------------------------------------------------
// 		for (auto & item : inputInfo) 
// 		{
// 			Blob::Ptr inputBlob = inferRequest_regular.GetBlob(item.first);

// 			SizeVector dims = inputBlob->getTensorDesc().getDims();

// 			// Fill input tensor with images. First b channel, then g and r channels
// 			size_t num_channels = dims[1];
// 			//std::cout << "num_channles = " << num_channels << std::endl;
// 			size_t image_size = dims[3] * dims[2];

// 			MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
// 			if (!minput)
// 			{
// 				cout << "We expect MemoryBlob from inferRequest_regular, but in fact we were not able to cast inputBlob to MemoryBlob" << endl;
// 				return 1;
// 			}

// 			// locked memory holder should be alive all time while access to its buffer happens
// 			auto minputHolder = minput->wmap();

// 			auto data = minputHolder.as<PrecisionTrait<Precision::U8>::value_type *>();//U8 or FP16 or FP32 original is U8
// 			unsigned char* pixels = (unsigned char*)(image.data);

// 			//cout << "image_size = " << image_size << endl;
// 			// Iterate over all pixel in image (b,g,r)
// 			for (size_t pid = 0; pid < image_size; pid++) 
// 			{
// 				// Iterate over all channels
// 				for (size_t ch = 0; ch < num_channels; ++ch) 
// 				{
// 					data[ch * image_size + pid] = pixels[pid*num_channels + ch];
// 				}
// 			}
// 		}
// 		// -----------------------------------------------------------------------------------------------------
// 	}