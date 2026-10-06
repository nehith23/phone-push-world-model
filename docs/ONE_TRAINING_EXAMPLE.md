# One training example, traced end to end

This is a training example from `left push.mp4`, starting at about 6.50 seconds. It explains the computation; it is not an example of held-out generalization.

At that moment, the tracked cover is at approximately `(26.16, 18.16)` in tabletop-projected centimetres, with its marker line turned about 6.2 degrees from the x-axis. The tool's position relative to the cover and its direction are also inputs.

Over the following 0.10 seconds:

| Quantity | Rightward change | Forward change |
|---|---:|---:|
| Tool movement supplied as the action | 0.63 cm | -0.03 cm |
| Cover movement measured from the video | 0.69 cm | 0.01 cm |
| Cover movement predicted by the trained model | 0.56 cm | 0.02 cm |

These centimetres are projected coordinates, with known geometry bias. They are not independent measurements of true physical displacement.

The model learns from many such examples. It takes the **current cover/tool arrangement and proposed tool movement**, and predicts **how the cover will move and rotate**. Its weights are adjusted to reduce the prediction error on training examples. Validation examples select when to stop training.

## How that becomes robot control

The controller considers 24 candidate pushes around the object (48 when the object is within 20 mm of the goal, adding 2 and 5 mm strokes). For each one, the learned model predicts where the cover would end up. The controller chooses the prediction closest to the goal, executes that push with the Panda, reads the new object pose, and repeats.

The neural model does not directly output Panda joint angles. Inverse kinematics and motor controllers convert the selected tool motion into robot joint motion.
