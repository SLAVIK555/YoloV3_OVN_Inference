import tensorflow as tf
import os
import gfile

model_dir = './export_dir/0'
model_name = 'saved_model.pb'

def create_graph():
    with gfile.GFile(os.path.join(model_dir, model_name), 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read(-1)) 
        tf.import_graph_def(graph_def, name='')

create_graph()
tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
for tensor_name in tensor_name_list:
    print(tensor_name, '\n')

