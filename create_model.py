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


def predict(x):
    buf = tf.io.decode_base64(x)
    img = tf.io.decode_png(buf, channels=3, dtype=tf.uint8)
    img = tf.image.resize(img, (224, 224), antialias=True)
    x = tf.expand_dims(img, 0)
    return model(x)


x = tf.TensorSpec(None, tf.string, name="input")
model_fn = tf.function(predict).get_concrete_function(x)
frozen_model = convert_variables_to_constants_v2(model_fn)

directory = "model"
tf.io.write_graph(frozen_model.graph, directory, "model.pb", as_text=False)
