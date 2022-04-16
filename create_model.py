try:
    import tensorflow as tf

    if tf.__version__.split(".")[0] != "2":
        raise Exception("This script requires tensorflow >= 2")
except:
    print(
        "// -----------------------\n[requirement] pip install tensorflow>=2\n// --------------------------"
    )
    raise

from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

# default input shape 224x224x3
model = tf.keras.applications.MobileNetV3Small(
    input_shape=[224, 224, 3], weights="imagenet"
)

x = tf.TensorSpec(None, tf.float32, name="input")
model_fn = tf.function(model).get_concrete_function(x)
frozen_model = convert_variables_to_constants_v2(model_fn)

directory = "model"
tf.io.write_graph(frozen_model.graph, directory, "model.pb", as_text=False)
