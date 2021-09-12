try:
    import tensorflow as tf

    if tf.__version__.split(".")[0] != "2":
        raise Exception("This script requires tensorflow >= 2")
except:
    print(
        "// -----------------------\n[requirement] pip install tensorflow>=2\n// --------------------------"
    )
    raise


class ModelWithPreprocess(tf.Module):
# default input shape 224x224x3
    def __init__(self):
        super(ModelWithPreprocess, self).__init__()
        self.model = tf.keras.applications.MobileNetV3Small(
            input_shape=[224, 224, 3], weights="imagenet"
        )

    @tf.function(input_signature=[tf.TensorSpec(None, tf.string, name="input")])
    def __call__(self, x):
        buf = tf.io.decode_base64(x)
        img = tf.io.decode_png(buf, channels=3, dtype=tf.uint8)
        img = tf.image.resize(img, (224, 224), antialias=True)
        x = tf.expand_dims(img, 0)
        return self.model(x)

module = ModelWithPreprocess()

directory = "model"
tf.saved_model.save(module, directory)
