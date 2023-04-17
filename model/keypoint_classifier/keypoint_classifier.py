#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


# class for identifying the landmarks compared to the trained model
class KeyPointClassifier(object):
    # instance created
    def __init__(
            # defines the model path
            self,
            model_path='model/keypoint_classifier/keypoint_classifier.tflite',
            num_threads=1,
    ):
        # Interpreter interface for running TensorFlow Lite models.
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        # Since TensorFlow Lite pre-plans tensor allocations to optimize inference, the user needs to call
        # allocate_tensors() before any inference.
        self.interpreter.allocate_tensors()
        # A list in which each item is a dictionary with details about an input tensor. Each dictionary contains the
        # following fields that describe the tensor:
        self.input_details = self.interpreter.get_input_details()
        # Gets model output tensor details.
        self.output_details = self.interpreter.get_output_details()

    # takes a list of landmarks as input, runs inference on a TensorFlow Lite interpreter,
    # and returns the index of the highest probability class predicted by the model.
    def __call__(
            # defines list
            self,
            landmark_list,
    ):
        # retrieves the index of the input tensor of the TensorFlow Lite interpreter
        # stored as an instance variable in the class.
        input_details_tensor_index = self.input_details[0]['index']
        # his sets the input tensor of the interpreter to the landmark_list
        # converted into a NumPy array of dtype np.float32.
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        # invokes the interpreter to run the inference process on the input data.
        self.interpreter.invoke()
        # retrieves the index of the output tensor of the TensorFlow Lite interpreter
        # stored as an instance variable in the class.
        output_details_tensor_index = self.output_details[0]['index']
        # retrieves the output tensor of the interpreter
        result = self.interpreter.get_tensor(output_details_tensor_index)
        # This computes the index of the highest probability class based on the output tensor by using the
        # argmax function on a flattened version of the result tensor.
        result_index = np.argmax(np.squeeze(result))
        # returns the index of the highest probability class
        return result_index
