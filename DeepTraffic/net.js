//<![CDATA[
// DQN for DeepTraffic by A7

// Variables to create state
lanesSide = 2;
patchesAhead = 7;
patchesBehind = 3;

//Variables for Neural Network
trainIterations = 10000;

// the number of other autonomous vehicles controlled by your network
otherAgents = 3; // max of 9

// Network Model Configuration
var num_inputs = (lanesSide * 2 + 1) * (patchesAhead + patchesBehind);
var num_actions = 5;
var temporal_window = 5;
var network_size = num_inputs * temporal_window + num_actions * temporal_window + num_inputs;

// NN Model Architecture

var layer_defs = [];
    layer_defs.push({
    type: 'input',
    out_sx: 1,
    out_sy: 1,
    out_depth: network_size
});
layer_defs.push({
    type: 'fc',
    num_neurons: 10,
    activation: 'relu'
});
layer_defs.push({
    type: 'fc',
    num_neurons: 10,
    activation: 'relu'
});
layer_defs.push({
    type: 'regression',
    num_neurons: num_actions
});

// Q-Learning Options | Temporal Difference Learner

var tdtrainer_options = { learning_rate: 0.01, momentum: 0.0, batch_size: 16, l2_decay: 0.01};

var opt = {};
opt.temporal_window = temporal_window;
opt.experience_size = 3000; // increase experience size
opt.start_learn_threshold = 500;
opt.gamma = 0.7; // high initial gamme
opt.learning_steps_total = 10000;
opt.learning_steps_burnin = 1000;
opt.epsilon_min = 0.0;
opt.epsilon_test_time = 0.0;
opt.layer_defs = layer_defs;
opt.tdtrainer_options = tdtrainer_options;

// Instantiate new brain
brain = new deepqlearn.Brain(num_inputs, num_actions, opt);

// Main NN code runs from here
learn = function (state, lastReward) {
brain.backward(lastReward);
var action = brain.forward(state);

draw_net();
draw_stats();

return action;
}

/* DQN with Experience Replay, Fixed Target Network, Reward Clipping, Skipping Frames

initialize replay memory D
initialize Q-network with random weights
observe initial s (forward pass)
	repeat:
		select action a:
			with random probability epsilon, select random action
			otherwise select a = argmax[Q(s,a')]
		execute action a
		observe reward r and new state s'
		store experience <s,a,r,s'> in D
		sample random transitions <ss,aa,rr,ss'> from DQN
		calculate target for each minibatch transition:
			if ss' is terminal state then tt=rr
			else tt=rr+ gamma(max.Q(ss',aa'))
		train network using (tt-Q(ss,aa))^2 as loss
		s=s'
	end
*/

//]]>
