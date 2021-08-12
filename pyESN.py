import sys
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

seed = 1000
np.random.seed(seed)  # Numpy module.


def correct_dimensions(s, target_length):
    """checks the dimensionality of arguments
    Args:
        s: None, scalar or 1D array
        target_length: expected length of s
    Returns:
        None if s is None, else numpy vector of length target_length
    """
    if s is not None:
        s = np.array(s)
        if s.ndim == 0:
            s = np.array([s] * target_length)   # expand the array
        elif s.ndim == 1:
            if not len(s) == target_length:
                raise ValueError("arg must have length " + str(target_length))
        else:
            raise ValueError("Invalid argument")
    return s


# activation function
def identity(x):
    return x


class ESN:

    def __init__(self, n_inputs, n_outputs, n_reservoir=200,
                 spectral_radius=0.95, density=0, noise=0.001, noise_ignore=True, input_shift=None,
                 input_scaling=None, teacher_forcing=True, feedback_scaling=1, teacher_scaling=None,
                 teacher_shift=None, out_activation=identity, inverse_out_activation=identity):
        """
        Args:
            n_inputs: input dimensions
            n_outputs: output dimensions
            n_reservoir: reservoir neurons
            spectral_radius: spectral radius of the recurrent weight matrix
            density: proportion of recurrent weights set to zero
            noise: noise added to each neuron (regularization)
            input_shift: scalar or vector of length n_inputs to add to each
                        input dimension before feeding it to the network.
            input_scaling: scalar or vector of length n_inputs to multiply
                        with each input dimension before feeding it to the network.
            teacher_forcing: if True, feed the target back into output units
            teacher_scaling: factor applied to the target signal
            teacher_shift: additive term applied to the target signal
            out_activation: output activation function (applied to the readout)
            inverse_out_activation: inverse of the output activation function (for train)
        """
        # check for proper dimensionality of all arguments and write them down.
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.density = density
        self.noise = noise
        self.noise_ignore = noise_ignore
        self.input_shift = correct_dimensions(input_shift, n_inputs)
        self.input_scaling = correct_dimensions(input_scaling, n_inputs)
        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift
        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation
        self.teacher_forcing = teacher_forcing
        self.feedback_scaling = feedback_scaling

        # build the free parameters
        self.weight_state, self.weight_in, self.weight_feedback = None, None, None
        self.weight_out, self.last_input, self.last_output = None, None, None
        self.last_state = None
        self.init_weights()

    def init_weights(self):
        # initialize recurrent weights between states:
        # get a uniform distribution from [-0.5, 0.5]
        weight_state = scipy.sparse.rand(self.n_reservoir, self.n_reservoir, density=self.density, format='coo')
        elg_value, elg_vector = scipy.sparse.linalg.eigs(weight_state)
        self.weight_state = weight_state / max(np.abs(elg_value)) * self.spectral_radius
        self.weight_state = self.weight_state.todense()
        self.weight_in = np.random.rand(self.n_reservoir, self.n_inputs) * 2 - 1
        self.weight_feedback = np.random.rand(self.n_reservoir, self.n_outputs) * 2 - 1

    def _update(self, state, input_pattern, output_pattern):
        """
        performs one update step.
        i.e., computes the next network state by applying the recurrent weights
        to the last state & and feeding in the current input and output patterns
        """
        # feedback loop
        if self.teacher_forcing:
            pre_activation = (np.dot(self.weight_state, state) +
                              np.dot(self.weight_in, input_pattern) +
                              np.dot(self.weight_feedback, output_pattern)*self.feedback_scaling)
        else:
            pre_activation = (np.dot(self.weight_state, state) +
                              np.dot(self.weight_in, input_pattern))

        # noise term
        if self.noise_ignore:
            return np.tanh(pre_activation)
        else:
            return np.tanh(pre_activation) + self.noise * (np.random.rand(self.n_reservoir) - 0.5)

    def _scale_inputs(self, inputs):
        """
        for each input dimension j: multiplies by the jth entry in the
        input_scaling argument, then adds the jth entry of the input_shift
        argument.
        """
        # default dtype of input is array
        if self.input_scaling is not None:
            # inputs = np.dot(inputs, np.diag(self.input_scaling))
            inputs = inputs*self.input_scaling
        if self.input_shift is not None:
            inputs = inputs + self.input_shift
        return inputs

    def _scale_teacher(self, teacher):
        """multiplies the teacher/target signal by the teacher_scaling argument,
        then adds the teacher_shift argument to it."""
        if self.teacher_scaling is not None:
            teacher = teacher * self.teacher_scaling
        if self.teacher_shift is not None:
            teacher = teacher + self.teacher_shift
        return teacher

    def _unscale_teacher(self, teacher_scaled):
        """inverse operation of the _scale_teacher method."""
        if self.teacher_shift is not None:
            teacher_scaled = teacher_scaled - self.teacher_shift
        if self.teacher_scaling is not None:
            teacher_scaled = teacher_scaled / self.teacher_scaling
        return teacher_scaled

    def train(self, inputs, outputs, inspect=False):
        """
        Collect the network's reaction to training data, train readout weights.
        Args:
            inputs: array of dimensions (N_training_samples x n_inputs)
            outputs: array of dimension (N_training_samples x n_outputs)
            inspect: show a visualisation of the collected reservoir states
        Returns:
            the network's output on the training data, using the trained weights
        """
        # transform any vectors of shape (x,) into vectors of shape (x,1):
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))
        # transform input and teacher signal:
        inputs_scaled = self._scale_inputs(inputs)
        teachers_scaled = self._scale_teacher(outputs)

        print("harvesting states...")
        # step the reservoir through the given input,output pairs:
        states = np.zeros((inputs.shape[0], self.n_reservoir))
        for n in range(1, inputs.shape[0]):
            states[n, :] = self._update(states[n - 1, :], inputs_scaled[n, :], teachers_scaled[n - 1, :])

        print("fitting...")
        # we'll disregard the first few states:
        transient = min(int(inputs.shape[1] / 10), 100)
        # include the raw inputs:
        bias_term = np.ones((inputs.shape[0], 1))
        extended_states = np.hstack((states, inputs_scaled, bias_term))

        # Solve for W_out:
        if np.max(teachers_scaled) > 1 and self.out_activation == np.tanh:
            print('the magnitude of output signal is too large for tanh function !')
            sys.exit()
        self.weight_out = np.dot(np.linalg.pinv(extended_states[transient:, :]),
                                 self.inverse_out_activation(teachers_scaled[transient:, :])).T

        # remember the last state for later:
        self.last_state = states[-1, :]
        self.last_input = inputs[-1, :]
        self.last_output = teachers_scaled[-1, :]

        # optionally visualize the collected states
        if inspect:
            # (^-- we depend on matplotlib only if this option is used)
            plt.figure()
            plt.imshow(extended_states.T, aspect='auto', interpolation='nearest')
            plt.colorbar()

        pre_train = self._unscale_teacher(self.out_activation(np.dot(extended_states, self.weight_out.T)))
        print("training error:{}".format(np.sqrt(np.var(pre_train - outputs))))
        return pre_train

    def test(self, inputs, continuation=True):
        """
        Apply the learned weights to the network's reactions to new input.
        Args:
            inputs: array of dimensions (N_test_samples x n_inputs)
            continuation: if True, start the network from the last training state
        Returns:
            Array of output activations
        """
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        n_samples = inputs.shape[0]

        if continuation:
            last_state = self.last_state
            last_input = self.last_input
            last_output = self.last_output
        else:
            last_state = np.zeros(self.n_reservoir)
            last_input = np.zeros(self.n_inputs)
            last_output = np.zeros(self.n_outputs)

        inputs = np.vstack([last_input, self._scale_inputs(inputs)])
        states = np.vstack([last_state, np.zeros((n_samples, self.n_reservoir))])
        outputs = np.vstack([last_output, np.zeros((n_samples, self.n_outputs))])
        bias_term = np.ones((n_samples+1, 1))

        for n in range(n_samples):
            states[n + 1, :] = self._update(states[n, :], inputs[n + 1, :], outputs[n, :])
            outputs[n + 1, :] = self.out_activation(np.dot(self.weight_out,
                                                    np.concatenate([states[n + 1, :], inputs[n + 1, :],
                                                                    bias_term[n + 1, :]])))

        return self._unscale_teacher(outputs[1:])
