
�groot"_tf_keras_network*�f{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 60, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract", "inbound_nodes": [["input_1", 0, 0, {"y": [[0.14158571721409033], [0.14158571721409033], [0.14158571721409033], [0.14158571721409033], [0.14158571721409033], [0.14158571721409033], [0.14158571721409033], [0.14158571721409033], [0.14158571721409033], [0.14158571861876376], [0.14158571861876376], [0.14158571861876376], [0.14158571861876376], [0.14158571861876376], [0.14158571861876376], [0.14158573323739052], [0.14158574785601724], [0.14158582263829655], [0.14158587802193234], [0.14158593143784612], [0.14158598457223562], [0.14158605148215303], [0.1415861237336618], [0.14158620948064812], [0.1415862949461102], [0.14158636719761897], [0.14158652632073718], [0.1415866379311076], [0.14158674476293517], [0.14158684006848113], [0.1415869845729726], [0.14158714285299198], [0.1415873314963619], [0.14158740852788748], [0.14158749821031777], [0.14158755949711951], [0.14158755949711951], [0.14158755949711951], [0.14158755949711951], [0.14158755949711951], [0.14158761713000156], [0.1415876747628836], [0.1415876747628836], [0.14158768403991895], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.1415877096232527], [0.14158773211424136], [0.1415877779393175], [0.14158785890628703]], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.truediv", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "name": "tf.math.truediv", "inbound_nodes": [["tf.math.subtract", 0, 0, {"y": [[0.42543424555971926], [0.42543424555971926], [0.4254342455597192], [0.4254342455597192], [0.4254342455597192], [0.4254342455597192], [0.4254342455597192], [0.4254342455597192], [0.4254342455597192], [0.425434245086676], [0.425434245086676], [0.425434245086676], [0.425434245086676], [0.425434245086676], [0.425434245086676], [0.4254342419238574], [0.4254342387784311], [0.42543425848790706], [0.4254342644917091], [0.4254342694587436], [0.42543427430141234], [0.42543428771692193], [0.4254343052920522], [0.42543433539569664], [0.42543436517990146], [0.42543438276702056], [0.4254345317193893], [0.42543459387690713], [0.4254346493364297], [0.4254346900354398], [0.42543480844893666], [0.4254349555522841], [0.4254351765099354], [0.42543519820443104], [0.4254352324873343], [0.4254352420138679], [0.4254352420138679], [0.4254352420138679], [0.42543524201386795], [0.42543524201386795], [0.42543524932434207], [0.42543525665652054], [0.42543525665652054], [0.4254352542440956], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.425435249845585], [0.4254352463723251], [0.42543524787301057], [0.42543527318224256]], "name": null}]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["tf.math.truediv", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "shared_object_id": 15, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 60, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 1]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 60, 1]}, "float32", "input_1"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 60, 1]}, "float32", "input_1"]}, "keras_version": "2.10.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 60, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract", "inbound_nodes": [["input_1", 0, 0, {"y": [[0.14158571721409033], [0.14158571721409033], [0.14158571721409033], [0.14158571721409033], [0.14158571721409033], [0.14158571721409033], [0.14158571721409033], [0.14158571721409033], [0.14158571721409033], [0.14158571861876376], [0.14158571861876376], [0.14158571861876376], [0.14158571861876376], [0.14158571861876376], [0.14158571861876376], [0.14158573323739052], [0.14158574785601724], [0.14158582263829655], [0.14158587802193234], [0.14158593143784612], [0.14158598457223562], [0.14158605148215303], [0.1415861237336618], [0.14158620948064812], [0.1415862949461102], [0.14158636719761897], [0.14158652632073718], [0.1415866379311076], [0.14158674476293517], [0.14158684006848113], [0.1415869845729726], [0.14158714285299198], [0.1415873314963619], [0.14158740852788748], [0.14158749821031777], [0.14158755949711951], [0.14158755949711951], [0.14158755949711951], [0.14158755949711951], [0.14158755949711951], [0.14158761713000156], [0.1415876747628836], [0.1415876747628836], [0.14158768403991895], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.1415877096232527], [0.14158773211424136], [0.1415877779393175], [0.14158785890628703]], "name": null}]], "shared_object_id": 1}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.truediv", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "name": "tf.math.truediv", "inbound_nodes": [["tf.math.subtract", 0, 0, {"y": [[0.42543424555971926], [0.42543424555971926], [0.4254342455597192], [0.4254342455597192], [0.4254342455597192], [0.4254342455597192], [0.4254342455597192], [0.4254342455597192], [0.4254342455597192], [0.425434245086676], [0.425434245086676], [0.425434245086676], [0.425434245086676], [0.425434245086676], [0.425434245086676], [0.4254342419238574], [0.4254342387784311], [0.42543425848790706], [0.4254342644917091], [0.4254342694587436], [0.42543427430141234], [0.42543428771692193], [0.4254343052920522], [0.42543433539569664], [0.42543436517990146], [0.42543438276702056], [0.4254345317193893], [0.42543459387690713], [0.4254346493364297], [0.4254346900354398], [0.42543480844893666], [0.4254349555522841], [0.4254351765099354], [0.42543519820443104], [0.4254352324873343], [0.4254352420138679], [0.4254352420138679], [0.4254352420138679], [0.42543524201386795], [0.42543524201386795], [0.42543524932434207], [0.42543525665652054], [0.42543525665652054], [0.4254352542440956], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.425435249845585], [0.4254352463723251], [0.42543524787301057], [0.42543527318224256]], "name": null}]], "shared_object_id": 2}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["tf.math.truediv", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["dense", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dropout_1", 0, 0, {}]]], "shared_object_id": 14}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanSquaredError", "config": {"name": "mean_squared_error", "dtype": "float32"}, "shared_object_id": 17}, {"class_name": "MeanAbsoluteError", "config": {"name": "mean_absolute_error", "dtype": "float32"}, "shared_object_id": 18}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0002500000118743628, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}2
�root.layer-0"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 60, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 60, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}2
�root.layer-1"_tf_keras_layer*�{"name": "tf.math.subtract", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "preserve_input_structure_in_config": true, "autocast": false, "class_name": "TFOpLambda", "config": {"name": "tf.math.subtract", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "inbound_nodes": [["input_1", 0, 0, {"y": [[0.14158571721409033], [0.14158571721409033], [0.14158571721409033], [0.14158571721409033], [0.14158571721409033], [0.14158571721409033], [0.14158571721409033], [0.14158571721409033], [0.14158571721409033], [0.14158571861876376], [0.14158571861876376], [0.14158571861876376], [0.14158571861876376], [0.14158571861876376], [0.14158571861876376], [0.14158573323739052], [0.14158574785601724], [0.14158582263829655], [0.14158587802193234], [0.14158593143784612], [0.14158598457223562], [0.14158605148215303], [0.1415861237336618], [0.14158620948064812], [0.1415862949461102], [0.14158636719761897], [0.14158652632073718], [0.1415866379311076], [0.14158674476293517], [0.14158684006848113], [0.1415869845729726], [0.14158714285299198], [0.1415873314963619], [0.14158740852788748], [0.14158749821031777], [0.14158755949711951], [0.14158755949711951], [0.14158755949711951], [0.14158755949711951], [0.14158755949711951], [0.14158761713000156], [0.1415876747628836], [0.1415876747628836], [0.14158768403991895], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.14158768713226408], [0.1415877096232527], [0.14158773211424136], [0.1415877779393175], [0.14158785890628703]], "name": null}]], "shared_object_id": 1, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 1]}}2
�root.layer-2"_tf_keras_layer*�{"name": "tf.math.truediv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "preserve_input_structure_in_config": true, "autocast": false, "class_name": "TFOpLambda", "config": {"name": "tf.math.truediv", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "inbound_nodes": [["tf.math.subtract", 0, 0, {"y": [[0.42543424555971926], [0.42543424555971926], [0.4254342455597192], [0.4254342455597192], [0.4254342455597192], [0.4254342455597192], [0.4254342455597192], [0.4254342455597192], [0.4254342455597192], [0.425434245086676], [0.425434245086676], [0.425434245086676], [0.425434245086676], [0.425434245086676], [0.425434245086676], [0.4254342419238574], [0.4254342387784311], [0.42543425848790706], [0.4254342644917091], [0.4254342694587436], [0.42543427430141234], [0.42543428771692193], [0.4254343052920522], [0.42543433539569664], [0.42543436517990146], [0.42543438276702056], [0.4254345317193893], [0.42543459387690713], [0.4254346493364297], [0.4254346900354398], [0.42543480844893666], [0.4254349555522841], [0.4254351765099354], [0.42543519820443104], [0.4254352324873343], [0.4254352420138679], [0.4254352420138679], [0.4254352420138679], [0.42543524201386795], [0.42543524201386795], [0.42543524932434207], [0.42543525665652054], [0.42543525665652054], [0.4254352542440956], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.42543525329306603], [0.425435249845585], [0.4254352463723251], [0.42543524787301057], [0.42543527318224256]], "name": null}]], "shared_object_id": 2, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 1]}}2
�root.layer-3"_tf_keras_layer*�{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["tf.math.truediv", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 1]}}2
�root.layer_with_weights-0"_tf_keras_layer*�{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}2
�root.layer-5"_tf_keras_layer*�{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense", 0, 0, {}]]], "shared_object_id": 7, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}2
�root.layer_with_weights-1"_tf_keras_layer*�{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 21}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}2
�root.layer-7"_tf_keras_layer*�{"name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 11, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}2
�	root.layer_with_weights-2"_tf_keras_layer*�{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_1", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}2
��root.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 23}2
��root.keras_api.metrics.1"_tf_keras_metric*�{"class_name": "MeanSquaredError", "name": "mean_squared_error", "dtype": "float32", "config": {"name": "mean_squared_error", "dtype": "float32"}, "shared_object_id": 17}2
��root.keras_api.metrics.2"_tf_keras_metric*�{"class_name": "MeanAbsoluteError", "name": "mean_absolute_error", "dtype": "float32", "config": {"name": "mean_absolute_error", "dtype": "float32"}, "shared_object_id": 18}2