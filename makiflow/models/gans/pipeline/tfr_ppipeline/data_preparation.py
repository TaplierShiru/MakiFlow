# Copyright (C) 2020  Igor Kilbas, Danil Gribanov, Artem Mukhin
#
# This file is part of MakiFlow.
#
# MakiFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.


import tensorflow as tf
from makiflow.generators.pipeline.tfr.utils import _tensor_to_byte_feature

# Save form
SAVE_FORM = "{0}_{1}.tfrecord"


# Feature names
TARGET_X_FNAME = 'TARGET_X_FNAME'
GEN_INPUT_X_FNAME = 'GEN_INPUT_X_FNAME'


# Serialize Data Point
def serialize_gans_data_point(target_tensor, gen_input, sess=None):

    feature = {
        TARGET_X_FNAME: _tensor_to_byte_feature(target_tensor, sess)
    }

    if gen_input is not None:
        feature.update({GEN_INPUT_X_FNAME: _tensor_to_byte_feature(gen_input, sess)})

    features = tf.train.Features(feature=feature)
    example_proto = tf.train.Example(features=features)

    return example_proto.SerializeToString()


def record_gans_train_data(target_tensors, gen_inputs, tfrecord_path, sess=None):
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for i, (target_tensor, gen_input) in enumerate(zip(target_tensors, gen_inputs)):

            serialized_data_point = serialize_gans_data_point(
                target_tensor=target_tensor,
                gen_input=gen_input,
                sess=sess
            )

            writer.write(serialized_data_point)


# Record data into multiple tfrecords
def record_mp_gans_train_data(
        target_tensors,
        prefix,
        dp_per_record,
        gen_inputs=None,
        sess=None):
    """
    Creates tfrecord dataset where each tfrecord contains `dp_per_second` data points

    Parameters
    ----------
    target_tensors : list or ndarray
        Array of target tensors.
    prefix : str
        Prefix for the tfrecords' names. All the filenames will have the same naming pattern:
        `prefix`_`tfrecord index`.tfrecord
    dp_per_record : int
        Data point per tfrecord. Defines how many tensors (locs, loc_masks, labels) will be
        put into one tfrecord file. It's better to use such `dp_per_record` that
        yields tfrecords of size 300-200 megabytes.
    gen_inputs : list or ndarray
        Array of input tensors for generator.
        By default equal to None, i.e. not used in training
    sess : tf.Session
        In case if you can't or don't want to run TensorFlow eagerly, you can pass in the session object.
    """
    for i in range(len(target_tensors) // dp_per_record):
        target_tensor = target_tensors[dp_per_record * i: (i + 1) * dp_per_record]

        if gen_inputs is not None:
            gen_input = gen_inputs[dp_per_record * i: (i + 1) * dp_per_record]
        else:
            gen_input = [None] * dp_per_record

        tfrecord_name = SAVE_FORM.format(prefix, i)

        record_gans_train_data(
            target_tensors=target_tensor,
            gen_inputs=gen_input,
            tfrecord_path=tfrecord_name,
            sess=sess
        )
