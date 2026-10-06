# Walkthrough

## The project in one paragraph

I recorded myself pushing a small rigid cover with a marked caliper. I tracked the cover and tool to turn video into state/action examples. A small neural network learns how object position and orientation change when the tool moves. In simulation, a planner asks that model what different pushes might do, selects one, executes it with a Panda arm, then observes and replans.

The final learned controller reaches 23 of 24 simulation targets within 5 mm. A geometric controller reaches all 24 and is more accurate and efficient. My contribution is the personally collected data and the complete, inspectable experiment; I do not claim learning beats geometry. Codex provided substantial assistance with implementation, experimental design, debugging, evaluation and documentation, and Claude (Anthropic) helped with documentation and reproducibility checks.

## Design questions

**What is the world model?** A function that predicts a state change from the current state and an action. Here it predicts planar translation and yaw change, not pixels or video. Two 32-unit tanh layers and an output layer contain 1,571 learned parameters.

**What counts as an action?** In the recordings, the observed displacement of the estimated tool contact point over roughly 0.1 seconds. During planning, candidate tool displacements are fed to the learned model. The simulator's lower-level controller turns a selected push into robot joint motion.

**What is being trained?** The dynamics predictor, using supervised regression. There is no learned RL policy, reward-optimization loop or VLA. The planner searches hand-designed candidate actions using the predictor.

**Why this model?** The available recordings directly provide state/action changes. Markers make those examples inspectable, and the tiny network trains cheaply. The experiment does not prove this is the best possible architecture or that it is superior to RL.

**How does a push happen?** The arm lifts, moves to a candidate contact position, descends, pushes, and settles. Motor control and contact physics move the cuboid. The object's pose is not set directly after initialization.

**What changed for precision?** I kept the network frozen, allowed 12 pushes, stopped within 5 mm, and added 2/5 mm candidate strokes within 20 mm of the goal. Both controllers received those changes. The earlier 15 mm evaluation had different cases and a smaller budget.

**Why does geometry win?** A straight-translation model already solves this simple position task well. Learning introduces prediction errors, and better one-step prediction on some recorded examples does not ensure better closed-loop control. Data bias and physical mismatch are possible contributors; I have not isolated them experimentally.

**Why did the remaining case fail?** It chose twelve 20 mm strokes and never reached the fine-action region. All pushes were predicted to improve distance, but six made it worse. It exhausted the budget 56.37 mm from the goal. The [trace](FAILURE_ANALYSIS.md) shows the mismatch; it does not isolate a root cause.

**Where is vision used?** In extracting the training data. Evaluation controllers receive exact simulator poses. A camera-based controller needs calibrated pose estimation, missing-detection handling and a fresh paired evaluation. This is a major gap, not already implemented functionality.

**What does 23/24 prove?** It is an observed result on one fresh simulation sample in one environment family. It does not establish real-world 5 mm accuracy, unseen-object generalization or a population-wide reliability rate.

## Read the code in this order

1. [One training example, traced end to end](ONE_TRAINING_EXAMPLE.md).
2. `scripts/stabilize_dataset.py`: which observations become examples, and which are rejected.
3. `scripts/train_stabilized.py`: normalization, model, loss and validation checkpoint selection.
4. `scripts/goal_control.py`: `FrozenDynamics`, candidate construction, observation and execution.
5. `scripts/precision_experiment.py`: shorter actions, model rollouts and action selection.
6. `scripts/evaluate_precision_frozen.py`: saved scenarios, hash guards and paired evaluation.
