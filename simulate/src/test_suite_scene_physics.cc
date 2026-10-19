#include "door_latch.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  char error[2048]{};
  mjModel* model = mj_loadXML(argv[1], nullptr, error, sizeof(error));
  if (!model) { std::cerr << error << '\n'; return 1; }
  mjData* data = mj_makeData(model);
  int result = 0;
  try {
    require(model->nu == 20, "scene must retain exactly the robot's 20 actuators");
    const int home = mj_name2id(model, mjOBJ_KEY, "home");
    require(home >= 0, "home keyframe missing");
    mj_resetDataKeyframe(model, data, home);
    // Robot occupies qpos [0,27); all scene joints must retain their MJCF
    // initial poses when the launcher selects home or the viewer resets.
    for (int i = 27; i < model->nq; ++i)
      require(std::abs(data->qpos[i] - model->qpos0[i]) < 1.e-6,
              "home keyframe moved a scene object away from its configured pose");
    mj_forward(model, data);
    const int lock = mj_name2id(model, mjOBJ_EQUALITY, "door_latch_lock");
    if (lock >= 0) {
      const int door = mj_name2id(model, mjOBJ_JOINT, "door_hinge");
      const int lever = mj_name2id(model, mjOBJ_JOINT, "door_lever_hinge");
      require(door >= 0 && lever >= 0, "latch joint missing");
      const int dq = model->jnt_qposadr[door], lq = model->jnt_qposadr[lever];
      const int dv = model->jnt_dofadr[door];
      const double direction = model->jnt_range[2*door] < -1 ? -1 : 1;
      require(data->eq_active[lock], "latch must start locked");
      for (int i=0; i<500; ++i) {
        UpdateDoorLatch(model, data);
        mj_step(model, data);
      }
      require(data->eq_active[lock], "lever gravity must not unlock the latch by itself");
      for (int i=0; i<100; ++i) {
        data->qfrc_applied[dv] = direction * 2.0;
        UpdateDoorLatch(model, data);
        mj_step(model, data);
      }
      require(std::abs(data->qpos[dq]) < .01, "door opened without depressing the lever");
      for (int i=0; i<150; ++i) {
        data->qpos[lq] = .65;
        data->qvel[model->jnt_dofadr[lever]] = 0;
        UpdateDoorLatch(model, data);
        mj_step(model, data);
      }
      require(!data->eq_active[lock], "depressed lever must unlock the latch");
      require(direction * data->qpos[dq] > .02, "unlocked door must swing in the specified direction");
      data->qpos[lq] = 0;
      data->qpos[dq] = direction * .5;
      UpdateDoorLatch(model, data);
      require(!data->eq_active[lock], "open door must stay unlocked after lever release");
      data->qpos[dq] = 0;
      UpdateDoorLatch(model, data);
      require(data->eq_active[lock], "closed door must relatch");
      mj_resetDataKeyframe(model, data, home);
      require(data->eq_active[lock], "reset must restore the latch");
    }
    std::cout << "Scene reset / actuator contract / latch physics: passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; result = 1;
  }
  mj_deleteData(data);
  mj_deleteModel(model);
  return result;
}
