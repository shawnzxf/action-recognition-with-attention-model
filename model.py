import tensorflow as tf
import pdb

def _variable_on_cpu(name, shape, initializer):
  # Define a variable on cpu
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, 
                                shape, 
                                wd, 
                                initializer=tf.contrib.layers.xavier_initializer()):
  # Define a variable with weight decay on cpu
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var)*wd
    tf.add_to_collection('weightdecay_losses', weight_decay)
  return var

class ConvLSTMCell(tf.nn.rnn_cell.RNNCell):
  """A LSTM cell with convolutions instead of multiplications.

  Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning 
    approach for precipitation nowcasting." Advances in Neural Information 
    Processing Systems. 2015.

  Args:
    shape: int tuple, Width and height of input, i.e. [width, height].
    filters: int, Number of output channels of the convolutional kernels inside
      ConvLSTM.
    kernel: tf variable, Convolutional kernels inside ConvLSTM with shape 
      [filter_width, filter_height, num_input_channels, num_output_channels].
      N.B., In LSTM, num_input_channels equals 2*filters and num_output_channels
      equals 4*filters.
    bias: tf variable, Biases for ConvLSTM with shape [num_output_channels].
    forget_bias: float, Biases of the forget gate are initialized by default
      to 1 in order to reduce the scale of forgetting at the beginning
      of the training.
    activation: activation function, Default tf.tanh, Nonlinear activate 
      function after convolution operation in LSTM.
    normalize: bool, Default False, set True to skip adding biases.
    data_format: string, Default 'channels_last', format of input tensor, 
      default [width, height, channels]. Set 'channels_first' if input has 
      the shape of [channels, width, height].
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
  """
  def __init__(self, 
               shape, 
               filters, 
               kernel,
               bias,
               forget_bias=1.0, 
               activation=tf.tanh, 
               normalize=False, 
               data_format='channels_last', 
               reuse=None):
    super(ConvLSTMCell, self).__init__(_reuse=reuse)
    self._kernel = kernel
    self._filters = filters
    self._forget_bias = forget_bias
    self._activation = activation
    self._normalize = normalize
    self._bias = bias
    if data_format == 'channels_last':
        self._size = tf.TensorShape(shape + [self._filters])
        self._feature_axis = self._size.ndims
        self._data_format = None
    elif data_format == 'channels_first':
        self._size = tf.TensorShape([self._filters] + shape)
        self._feature_axis = 0
        self._data_format = 'NC'
    else:
        raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

  @property
  def output_size(self):
    return self._size

  def call(self, x, state):
    """Update state in LSTM at each timestep.

    Args:
      x: '4-D' tensor, Input at each timestep, with shape [batch_size, 
        width, height, channels].
      state: An LSTMStateTuple of state tensors, each shaped [batch_size, 
        width, height, channels].
    """
    c, h = state  # c:cell state, h:hidden state

    # Convolution operation in matrix for ConvLSTM
    x = tf.concat([x, h], axis=self._feature_axis)
    W = self._kernel
    y = tf.nn.convolution(x, W, 'SAME', data_format=self._data_format)
    if not self._normalize:
      y += self._bias
    j, i, f, o = tf.split(y, 4, axis=self._feature_axis)
    # j: new input, i: input gate, f: forget gate, o: output gate

    f = tf.sigmoid(f + self._forget_bias)
    i = tf.sigmoid(i)
    c = c * f + i * self._activation(j)
    o = tf.sigmoid(o)
    h = o * self._activation(c)

    state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

    return h, state

class AttenConvLSTMCell(tf.nn.rnn_cell.RNNCell):
  """A LSTM cell with convolutions equipped with attention model.

  Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning 
    approach for precipitation nowcasting." Advances in Neural Information 
    Processing Systems. 2015.

  Args:
    shape: int tuple, Width and height of input, i.e. [width, height].
    filters: int, Number of output channels of the convolutional kernels inside
      ConvLSTM.
    kernel: tf variable, Convolutional kernels inside ConvLSTM with shape 
      [filter_width, filter_height, num_input_channels, num_output_channels].
      N.B., In LSTM, num_input_channels equals 2*filters and num_output_channels
      equals 4*filters.
    bias: tf variable, Biases for ConvLSTM with shape [num_output_channels].
    atten_kernel: tf variable, Convolutional kernels for attention values 
      generation with shape [filter_width, filter_height, num_input_channels, 
      num_output_channels]. N.B., In this attention model, num_input_channels 
      equals 2*filters and num_output_channels equals filters.
    atten_bias: tf variable, Biases for attention values generation with shape 
      [num_output_channels].
    W_z: tf variable, Convolutional kernels for attention map normalization 
      with shape [filter_width, filter_height, num_input_channels, 
      num_output_channels]. N.B., num_output_channels equals one, since 
      attention map is a single channel map to assign different weight in 
      different region of a frame.
    W_z_bias: tf variable, Biases for attention map normalization with shape
      [num_output_channels]. N.B., num_output_channels equals one here.
    forget_bias: float, Biases of the forget gate are initialized by default
      to 1 in order to reduce the scale of forgetting at the beginning
      of the training.
    activation: activation function, Default tf.tanh, Nonlinear activate 
      function after convolution operation in LSTM.
    normalize: bool, Default False, set True to skip adding biases.
    data_format: string, Default 'channels_last', format of input tensor, 
      default [width, height, channels]. Set 'channels_first' if input has 
      the shape of [channels, width, height].
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
  """

  def __init__(self, 
               shape, 
               filters, 
               kernel,
               bias,
               atten_kernel,
               atten_bias,
               W_z, 
               W_z_bias,
               forget_bias=1.0, 
               activation=tf.tanh, 
               normalize=False, 
               data_format='channels_last', 
               reuse=None):
    super(AttenConvLSTMCell, self).__init__(_reuse=reuse)
    self._kernel = kernel
    self._filters = filters
    self._forget_bias = forget_bias
    self._activation = activation
    self._normalize = normalize
    self._bias = bias
    self._atten_kernel = atten_kernel
    self._atten_bias = atten_bias
    self._W_z = W_z
    self._W_z_bias = W_z_bias
    if data_format == 'channels_last':
        self._size = tf.TensorShape(shape + [self._filters])
        self._feature_axis = self._size.ndims
        self._data_format = None
    elif data_format == 'channels_first':
        self._size = tf.TensorShape([self._filters] + shape)
        self._feature_axis = 0
        self._data_format = 'NC'
    else:
        raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

  @property
  def output_size(self):
    return self._size

  def call(self, x, state):
    """Update state in LSTM at each timestep.

    Args:
      x: '4-D' tensor, Input at each timestep, with shape [batch_size, 
        width, height, channels].
      state: An LSTMStateTuple of state tensors, each shaped [batch_size, 
        width, height, channels].
    """
    c, h = state  # c:cell state, h:hidden state

    inputs = tf.concat([x, h], axis=self._feature_axis)

    # Compute attention map in matrix first
    W_atten = self._atten_kernel
    W_z = self._W_z
    atten = tf.nn.convolution(
      inputs, W_atten,'SAME', data_format=self._data_format) + self._atten_bias
    atten = tf.tanh(atten)
    # Normalize attention values
    atten = tf.nn.convolution(
      atten, W_z, 'SAME', data_format=self._data_format) + self._W_z_bias
    atten = tf.nn.softmax(atten, name='atten_weight')

    # Assign attention map to input
    x_atten = atten * x
    new_inputs = tf.concat([x_atten, h], axis=self._feature_axis)

    # Convolution operation in matrix for ConvLSTM
    W = self._kernel
    y = tf.nn.convolution(new_inputs, W, 'SAME', data_format=self._data_format)
    if not self._normalize:
      y += self._bias
    j, i, f, o = tf.split(y, 4, axis=self._feature_axis)
    # j: new input, i: input gate, f: forget gate, o: output gate

    f = tf.sigmoid(f + self._forget_bias)
    i = tf.sigmoid(i)
    c = c * f + i * self._activation(j)
    o = tf.sigmoid(o)
    h = o * self._activation(c)

    state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

    return h, state

class AttenCell(tf.nn.rnn_cell.RNNCell):
  """An attention model with LSTM inside for attention generation.

  Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning 
    approach for precipitation nowcasting." Advances in Neural Information 
    Processing Systems. 2015.

  Args:
    shape: int tuple, Width and height of input, i.e. [width, height].
    filters: int, Number of output channels of the convolutional kernels inside
      ConvLSTM.
    kernel: tf variable, Convolutional kernels inside ConvLSTM with shape 
      [filter_width, filter_height, num_input_channels, num_output_channels].
      N.B., In LSTM, num_input_channels equals 2*filters and num_output_channels
      equals 4*filters.
    bias: tf variable, Biases for ConvLSTM with shape [num_output_channels].
    atten_kernel: tf variable, Convolutional kernels for attention values 
      generation with shape [filter_width, filter_height, num_input_channels, 
      num_output_channels]. N.B., In this attention model, num_input_channels 
      equals 2*filters and num_output_channels equals filters.
    atten_bias: tf variable, Biases for attention values generation with shape 
      [num_output_channels].
    W_z: tf variable, Convolutional kernels for attention map normalization 
      with shape [filter_width, filter_height, num_input_channels, 
      num_output_channels]. N.B., num_output_channels equals one, since 
      attention map is a single channel map to assign different weight in 
      different region of a frame.
    W_z_bias: tf variable, Biases for attention map normalization with shape
      [num_output_channels]. N.B., num_output_channels equals one here.
    forget_bias: float, Biases of the forget gate are initialized by default
      to 1 in order to reduce the scale of forgetting at the beginning
      of the training.
    activation: activation function, Default tf.tanh, Nonlinear activate 
      function after convolution operation in LSTM.
    normalize: bool, Default False, set True to skip adding biases.
    data_format: string, Default 'channels_last', format of input tensor, 
      default [width, height, channels]. Set 'channels_first' if input has 
      the shape of [channels, width, height].
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
  """
  def __init__(self, 
               shape, 
               filters, 
               kernel,
               bias,
               atten_kernel,
               atten_bias,
               W_z, 
               W_z_bias,
               forget_bias=1.0, 
               activation=tf.tanh, 
               normalize=False, 
               data_format='channels_last', 
               reuse=None):
    super(AttenCell, self).__init__(_reuse=reuse)
    self._kernel = kernel
    self._filters = filters
    self._forget_bias = forget_bias
    self._activation = activation
    self._normalize = normalize
    self._bias = bias
    self._atten_kernel = atten_kernel
    self._atten_bias = atten_bias
    self._W_z = W_z
    self._W_z_bias = W_z_bias
    if data_format == 'channels_last':
        self._size = tf.TensorShape(shape + [self._filters])
        self._feature_axis = self._size.ndims
        self._data_format = None
    elif data_format == 'channels_first':
        self._size = tf.TensorShape([self._filters] + shape)
        self._feature_axis = 0
        self._data_format = 'NC'
    else:
        raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

  @property
  def output_size(self):
    return self._size

  def call(self, x, state):
    """Update state in LSTM at each timestep.

    Args:
    x: '4-D' tensor, Input at each timestep, with shape [batch_size, 
      width, height, channels].
    state: An LSTMStateTuple of state tensors, each shaped [batch_size, 
      width, height, channels].
    """
    c, h = state # c:cell state, h:hidden state

    inputs = tf.concat([x, h], axis=self._feature_axis)

    # Compute the values for attention map first
    W_atten = self._atten_kernel
    W_z = self._W_z
    atten = tf.nn.convolution(
      inputs, W_atten,'SAME', data_format=self._data_format) + self._atten_bias
    atten = tf.tanh(atten)    
    # Normalize attention values
    atten = tf.nn.convolution(
      atten, W_z, 'SAME', data_format=self._data_format) + self._W_z_bias
    atten = tf.nn.softmax(atten, name='atten_weight')

    # Assign attention map to input
    x_atten = atten * x
    new_inputs = tf.concat([x_atten, h], axis=self._feature_axis)

    # Convolution operation in matrix for ConvLSTM
    W = self._kernel
    y = tf.nn.convolution(new_inputs, W, 'SAME', data_format=self._data_format)
    if not self._normalize:
      y += self._bias
    j, i, f, o = tf.split(y, 4, axis=self._feature_axis)

    f = tf.sigmoid(f + self._forget_bias)
    i = tf.sigmoid(i)
    c = c * f + i * self._activation(j)
    o = tf.sigmoid(o)
    h = o * self._activation(c)

    state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

    return x_atten, state

def cham(_X, batch_size, timesteps, crop_size, num_class, drop_out):
  """Construct CHAM network, i.e. convolutional hierarchical attention model.

  Args:
    _X: '4-D' tensor, Input to the network with shape[batch_size, time_steps,
      height, width, channels].
    batch_size: int, Number of video clips in a batch.
    timesteps: int, Number of timesteps in a video clip.
    crop_size: int, Crop size of a cropped video frame.
    num_class: int, Number of all classes in the dataset.  
    drop_out: float, Rate of drop out while training.
  """

  with tf.variable_scope('cham'):
    # Define variables for CHAM
    weights = {
      'conv_kernel_fir': 
        _variable_with_weight_decay('conv_kernel_fir', [3,3,1024,2048], 0.001),
      'conv_kernel_sec': 
        _variable_with_weight_decay('conv_kernel_sec', [3,3,1024,2048], 0.001),
      'atten_kernel':
        _variable_with_weight_decay('atten_kernel', [3, 3, 1024, 512], 0.001),
      'W_z':
        _variable_with_weight_decay('W_z', [1, 1, 512, 1], 0.001),
      'fc_weight': 
        _variable_with_weight_decay('fc_weight', [8192 ,num_class], 0.001)}
    biases = {
      'conv_bias_fir': 
        _variable_with_weight_decay('conv_bias_fir', [2048], 0.000, tf.zeros_initializer()),
      'conv_bias_sec': 
        _variable_with_weight_decay('conv_bias_sec', [2048], 0.000, tf.zeros_initializer()),
      'atten_bias': 
        _variable_with_weight_decay('atten_bias', [512], 0.000, tf.zeros_initializer()),
      'W_z_bias': 
        _variable_with_weight_decay('W_z_bias', [1], 0.000, tf.zeros_initializer()),
      'fc_bias': 
        _variable_with_weight_decay('fc_bias', [num_class], 0.000, tf.zeros_initializer())}

    # First layer of convolutional LSTM with attention
    atten_cell = AttenConvLSTMCell(shape = [crop_size, crop_size], 
                                   filters = 512, 
                                   kernel = weights['conv_kernel_fir'],
                                   bias = biases['conv_bias_fir'],
                                   atten_kernel = weights['atten_kernel'],
                                   atten_bias = biases['atten_bias'],
                                   W_z = weights['W_z'],
                                   W_z_bias = biases['W_z_bias'])
    atten_cell = tf.contrib.rnn.DropoutWrapper(atten_cell, drop_out)
    initial_state_0 = atten_cell.zero_state(batch_size, dtype=tf.float32)
    l1_outputs, l1_state = tf.nn.dynamic_rnn(cell = atten_cell, 
                                             inputs = _X,
                                             initial_state = initial_state_0,
                                             dtype=tf.float32)

    # Second layer of convolutional LSTM without attention
    l1_outputs_skip = l1_outputs[:,::2,:,:,:]  # Skip 1 timestep every twice
    basic_cell = ConvLSTMCell(shape = [crop_size, crop_size], 
                              filters = 512, 
                              kernel = weights['conv_kernel_sec'],
                              bias = biases['conv_bias_sec'])
    basic_cell = tf.contrib.rnn.DropoutWrapper(basic_cell, drop_out)
    initial_state = basic_cell.zero_state(batch_size, dtype=tf.float32)
    l2_outputs, l2_state = tf.nn.dynamic_rnn(cell = basic_cell, 
                                             inputs = l1_outputs_skip,
                                             initial_state = initial_state,
                                             dtype=tf.float32)

    # Concatenate the output features of the first and the second layer
    concat_outputs = tf.concat(values = [l1_outputs, l2_outputs], axis = 1)

    # Build an average pooling layer1_outputs
    avg_outputs = tf.nn.avg_pool3d(input = concat_outputs, 
                                   ksize = [1, 1, 2, 2, 1], 
                                   strides = [1, 1, 2, 2, 1], 
                                   padding = 'SAME',
                                   name = 'avg_pool')
    
    # Build a Fully Connected layer
    # Change data format for Fully Connected layer
    avg_outputs = tf.transpose(avg_outputs, perm =  [0,1,4,2,3])
    avg_outputs = tf.reshape(avg_outputs, [batch_size,int(timesteps*1.5), -1])

    dense_list = []
    for batch_index in range(batch_size):
      dense = tf.matmul(
        avg_outputs[batch_index,:,:], weights['fc_weight']) + biases['fc_bias']
      dense = tf.reshape(dense, [1,int(timesteps*1.5),num_class])
      dense_list.append(dense)
    dense = tf.concat([x for x in dense_list], axis=0)
    dense = tf.nn.relu(dense, name='fc') # Relu activation
    dense = tf.nn.dropout(dense, drop_out)

    # Compute the frame-level average prediction
    out = tf.reduce_mean(dense,axis=1,name='output')

    # Build a softmax layer--do it in train_cham/test_cham script
    # out = tf.nn.softmax(out, name='pred')
  
  return out

def conv_atten(_X, batch_size, timesteps, crop_size, num_class, drop_out):
  """Build single layer of ConvLSTM network with attention model.

  Args:
    _X: '4-D' tensor, Input to the network with shape[batch_size, time_steps,
      height, width, channels].
    batch_size: int, Number of video clips in a batch.
    timesteps: int, Number of timesteps in a video clip.
    crop_size: int, Crop size of a cropped video frame.
    num_class: int, Number of all classes in the dataset.  
    drop_out: float, Rate of drop out while training.
  """

  with tf.variable_scope('conv_atten'):
    # Define variables for Conv-Atten
    weights = {
      'conv_kernel_fir': 
        _variable_with_weight_decay('conv_kernel_fir', [3,3,1024,2048], 0.001),
      'atten_kernel':
        _variable_with_weight_decay('atten_kernel', [3, 3, 1024, 512], 0.001),
      'W_z':
        _variable_with_weight_decay('W_z', [1, 1, 512, 1], 0.001),
      'fc_weight': 
        _variable_with_weight_decay('fc_weight', [8192 ,num_class], 0.001)}
    biases = {
      'conv_bias_fir': 
        _variable_with_weight_decay('conv_bias_fir', [2048], 0.000, tf.zeros_initializer()),
      'atten_bias': 
        _variable_with_weight_decay('atten_bias', [512], 0.000, tf.zeros_initializer()),
      'W_z_bias': 
        _variable_with_weight_decay('W_z_bias', [1], 0.000, tf.zeros_initializer()),
      'fc_bias': 
        _variable_with_weight_decay('fc_bias', [num_class], 0.000, tf.zeros_initializer())}

    # First layer of convolutional LSTM with attention
    atten_cell = AttenConvLSTMCell(shape = [crop_size, crop_size], 
                                   filters = 512, 
                                   kernel = weights['conv_kernel_fir'],
                                   bias = biases['conv_bias_fir'],
                                   atten_kernel = weights['atten_kernel'],
                                   atten_bias = biases['atten_bias'],
                                   W_z = weights['W_z'],
                                   W_z_bias = biases['W_z_bias'])
    atten_cell = tf.contrib.rnn.DropoutWrapper(atten_cell, drop_out)
    initial_state_0 = atten_cell.zero_state(batch_size, dtype=tf.float32)
    l1_outputs, l1_state = tf.nn.dynamic_rnn(cell = atten_cell, 
                                             inputs = _X,
                                             initial_state = initial_state_0,
                                             dtype=tf.float32)

    # Build an average pooling layer1_outputs
    avg_outputs = tf.nn.avg_pool3d(input = l1_outputs, 
                                   ksize = [1, 1, 2, 2, 1], 
                                   strides = [1, 1, 2, 2, 1], 
                                   padding = 'SAME',
                                   name = 'avg_pool')
    
    # Build a Fully Connected layer
    # Change data format for Fully Connected layer
    avg_outputs = tf.transpose(avg_outputs, perm =  [0,1,4,2,3])
    avg_outputs = tf.reshape(avg_outputs, [batch_size,int(timesteps), -1])

    dense_list = []
    for batch_index in range(batch_size):
      dense = tf.matmul(
        avg_outputs[batch_index,:,:], weights['fc_weight']) + biases['fc_bias']
      dense = tf.reshape(dense, [1,int(timesteps),num_class])
      dense_list.append(dense)
    dense = tf.concat([x for x in dense_list], axis=0)
    dense = tf.nn.relu(dense, name='fc') # Relu activation
    dense = tf.nn.dropout(dense, drop_out)

    # Compute the frame-level average prediction
    out = tf.reduce_mean(dense,axis=1,name='output')

    # Build a softmax layer--do it in train_cham/test_cham script
    # out = tf.nn.softmax(out, name='pred')
  
  return out

def fc_atten(_X, batch_size, timesteps, crop_size, num_class, drop_out):
  """Build a network of fully connected layer with attention model.

  Args:
    _X: '4-D' tensor, Input to the network with shape[batch_size, time_steps,
      height, width, channels].
    batch_size: int, Number of video clips in a batch.
    timesteps: int, Number of timesteps in a video clip.
    crop_size: int, Crop size of a cropped video frame.
    num_class: int, Number of all classes in the dataset.  
    drop_out: float, Rate of drop out while training.
  """

  with tf.variable_scope('fc_atten'):
    # Define variables for FC-Atten
    weights = {
      'conv_kernel_fir': 
        _variable_with_weight_decay('conv_kernel_fir', [3,3,1024,2048], 0.001),
      'atten_kernel':
        _variable_with_weight_decay('atten_kernel', [3, 3, 1024, 512], 0.001),
      'W_z':
        _variable_with_weight_decay('W_z', [1, 1, 512, 1], 0.001),
      'fc_weight': 
        _variable_with_weight_decay('fc_weight', [8192 ,num_class], 0.001)}
    biases = {
      'conv_bias_fir': 
        _variable_with_weight_decay('conv_bias_fir', [2048], 0.000, tf.zeros_initializer()),
      'atten_bias': 
        _variable_with_weight_decay('atten_bias', [512], 0.000, tf.zeros_initializer()),
      'W_z_bias': 
        _variable_with_weight_decay('W_z_bias', [1], 0.000, tf.zeros_initializer()),
      'fc_bias': 
        _variable_with_weight_decay('fc_bias', [num_class], 0.000, tf.zeros_initializer())}

    # First layer of convolutional LSTM for attention generation
    atten_cell = AttenCell(shape = [crop_size, crop_size], 
                           filters = 512, 
                           kernel = weights['conv_kernel_fir'],
                           bias = biases['conv_bias_fir'],
                           atten_kernel = weights['atten_kernel'],
                           atten_bias = biases['atten_bias'],
                           W_z = weights['W_z'],
                           W_z_bias = biases['W_z_bias'])
    atten_cell = tf.contrib.rnn.DropoutWrapper(atten_cell, drop_out)
    initial_state_0 = atten_cell.zero_state(batch_size, dtype=tf.float32)
    l1_outputs, l1_state = tf.nn.dynamic_rnn(cell = atten_cell, 
                                             inputs = _X,
                                             initial_state = initial_state_0,
                                             dtype=tf.float32)

    # Build an average pooling layer1_outputs
    avg_outputs = tf.nn.avg_pool3d(input = l1_outputs, 
                                   ksize = [1, 1, 2, 2, 1], 
                                   strides = [1, 1, 2, 2, 1], 
                                   padding = 'SAME',
                                   name = 'avg_pool')
    
    # Build a Fully Connected layer
    # Change data format for Fully Connected layer
    avg_outputs = tf.transpose(avg_outputs, perm =  [0,1,4,2,3])
    avg_outputs = tf.reshape(avg_outputs, [batch_size,int(timesteps), -1])

    dense_list = []
    for batch_index in range(batch_size):
      dense = tf.matmul(
        avg_outputs[batch_index,:,:], weights['fc_weight']) + biases['fc_bias']
      dense = tf.reshape(dense, [1,int(timesteps),num_class])
      dense_list.append(dense)
    dense = tf.concat([x for x in dense_list], axis=0)
    dense = tf.nn.relu(dense, name='fc') # Relu activation
    dense = tf.nn.dropout(dense, drop_out)

    # Compute the frame-level average prediction
    out = tf.reduce_mean(dense,axis=1,name='output')

    # Build a softmax layer--do it in train_cham/test_cham script
    # out = tf.nn.softmax(out, name='pred')
  
  return out