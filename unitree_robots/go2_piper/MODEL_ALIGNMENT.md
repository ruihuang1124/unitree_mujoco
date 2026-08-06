# Go2-Piper model alignment

`go2_piper_fake_ee.xml` is the canonical MuJoCo robot model used by the
Go2-Arm deployment scenes. Its source of truth is the training URDF:

```text
/home/ray/colcon_ws/src/quadruped_control_ros2/robot_description/
go2_arm_description/xacro/go2_arm_front_nuc_low.urdf
```

The canonical model matches the URDF rigid-body masses, centers of mass,
full inertia tensors, fixed and actuated joint transforms, joint axes and
limits, actuator effort limits, and primitive collision geometry. The Piper
collision meshes are byte-identical copies of the URDF meshes. The
`fake_ee` OptiTrack visualization frame is deliberately massless.

The default keyframe uses the joint pose from the Isaac Lab training robot.
Foot friction uses the nominal training value. MuJoCo and PhysX still have
different contact solvers, so matching the robot description does not make
their contact dynamics mathematically identical.

Run the alignment audit after editing either model:

```bash
cd /home/ray/projects/unitree_mujoco
python3 unitree_robots/go2_piper/tools/validate_urdf_alignment.py
```
