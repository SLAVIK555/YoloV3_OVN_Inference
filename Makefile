IncludeOCVPath = /usr/local/include/opencv4
LibOCVPath = /usr/local/lib
OCVlibs = -lopencv_core -lopencv_dnn -lopencv_imgproc -lopencv_imgcodecs -lopencv_img_hash -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_videoio -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_flann -lopencv_face -lopencv_photo -lopencv_xphoto

IncludeOVNPath = /opt/intel/openvino_2021/deployment_tools/inference_engine/include
LibOVNPath = /opt/intel/openvino_2021/deployment_tools/inference_engine/lib/intel64
OVNlibs = -linference_engine

IncludeNGraphPath = /opt/intel/openvino_2021/deployment_tools/ngraph/include
LibNGraphPath = /opt/intel/openvino_2021/deployment_tools/ngraph/lib
NGraphlibs = -lngraph

Inc1 = /opt/intel/openvino_2021/deployment_tools/open_model_zoo/demos/common/cpp/monitors/include
Inc2 = /opt/intel/openvino_2021/deployment_tools/open_model_zoo/demos/common/cpp/utils/include
Inc3 = /opt/intel/openvino_2021/deployment_tools/open_model_zoo/demos/multi_channel_common/cpp###

# all: Main MainInference

# # 	g++ -I$(IncludeOCVPath) -I$(IncludeOVNPath) MainInference.cpp -o MainInference -L$(LibOCVPath) $(OCVlibs) -L$(LibOVNPath) $(OVNlibs)
# 	#g++ -I$(IncludeOCVPath) -I$(IncludeOVNPath) FHInference.cpp -o FHInference -L$(LibOCVPath) $(OCVlibs) -L$(LibOVNPath) $(OVNlibs)

# Main: Main.o Infer.o
# 	g++ Main.o Infer.o -o Main -L$(LibOCVPath) $(OCVlibs) -L$(LibOVNPath) $(OVNlibs)

# Main.o: Main.cpp
# 	g++ Main.cpp -c -I$(IncludeOCVPath) -I$(IncludeOVNPath)

# Infer.o: Infer.cpp
# 	g++ Infer.cpp -c -I$(IncludeOCVPath) -I$(IncludeOVNPath)

#MainInference:
	#g++ -I$(IncludeOCVPath) -I$(IncludeOVNPath) MainInference.cpp -o MainInference -L$(LibOCVPath) $(OCVlibs) -L$(LibOVNPath) $(OVNlibs)

	
	#g++ -I$(IncludeOCVPath) -I$(IncludeOVNPath) HelloWorld.cpp -o HelloWorld -L$(LibOCVPath) $(OCVlibs) -L$(LibOVNPath) $(OVNlibs)
	#g++ -I$(IncludeOCVPath) -I$(IncludeOVNPath) Infer.cpp -o Infer -L$(LibOCVPath) $(OCVlibs) -L$(LibOVNPath) $(OVNlibs)
	#g++ -I$(IncludeOCVPath) -I$(IncludeOVNPath) FHInference.cpp -o FHInference -L$(LibOCVPath) $(OCVlibs) -L$(LibOVNPath) $(OVNlibs)

all:
	g++ -I$(IncludeOCVPath) -I$(IncludeOVNPath) -I$(IncludeNGraphPath) -I$(Inc1) -I$(Inc2) -I$(Inc3) V3MainInference.cpp -o V3MainInference -L$(LibOCVPath) $(OCVlibs) -L$(LibOVNPath) $(OVNlibs) -L$(LibNGraphPath) $(NGraphlibs)


