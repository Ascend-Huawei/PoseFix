import tensorflow as tf 
print("111")
tf.reset_default_graph()
saver = tf.train.import_meta_graph("../output/model_dump/COCO/snapshot_140.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess, "../output/model_dump/COCO/snapshot_140.ckpt")
    output_node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        ["Mean","Mean_1"])
    tf.train.write_graph(frozen_graph_def,'../output/pb_dump/COCO/','posefix.pbtxt', as_text=True)
    
    # Save the frozen graph
    with open("../output/pb_dump/COCO/posefix.pb", 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())