import tensorflow as tf


class TfLiteModel(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.input_tensor = []
        self.output_tensor = []
        self.get_model(model_path)

    def get_model(self, model_path):
        self.model = tf.lite.Interpreter(model_path=model_path)
        self.model.allocate_tensors()

        # Get input and output tensors.
        self.input_tensor = self.model.get_input_details()
        self.output_tensor = self.model.get_output_details()
        # print(self.input_tensor)

    def model_predict(self, image):
        self.model.set_tensor(self.input_tensor[0]['index'], image)
        self.model.invoke()
        if len(self.output_tensor) == 2:
            prediction_a = self.model.get_tensor(self.output_tensor[0]['index'])
            prediction_b = self.model.get_tensor(self.output_tensor[1]['index'])
            return prediction_a, prediction_b
        else:
            prediction_a = self.model.get_tensor(self.output_tensor[0]['index'])
        return prediction_a
