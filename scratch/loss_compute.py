def loss_single_step(predictions, labels, num_classes):
	""" loss_single_step : Compute loss given softmax
	computed on top of logits

	Args:
		predictions : softmax applied on logits [batch_size, num_classes]
		labels : The true label [0, num_classes) [batch_size]
		num_classes : Number of classes
	"""
	with tf.name_scope("loss_single_step"):
		batch_size = tf.size(labels)
		labels = tf.expand_dims(labels, 1)
		indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
		concated = tf.concat(1, [indices, labels])
		onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, num_classes]), 1.0, 0.0)

		cross_entropy = -onehot_labels * tf.log(predictions)
		loss = tf.reduce_sum(cross_entropy, name="xentropy")
	
	return loss

def loss_multiple_step(predictions, labels, num_classes):
	""" loss_multiple_step : Compute loss given softmax 
	computed on top of logits for multiple time steps.

	Args:
		predictions : softmax on logits: list of 2D tensors
		[batch_size, num_classes]
		labels : list of 1D tensors of [batch_size]
		num_classes: number of classes
	"""

	for (pred, label) in zip(predictions, labels):
		loss += loss_single_step(pred, label, num_classes)
	
	return loss
