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

    # instance called
    def __call__(
            # defines list
            self,
            landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index
