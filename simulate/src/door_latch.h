#pragma once
#include <mujoco/mujoco.h>

// A latch abstraction, not a grasp weld: the handle must physically rotate
// before the door-hinge lock releases. Once away from the jamb, the door stays
// free even after the spring-loaded lever returns. Reset restores the lock.
inline void UpdateDoorLatch(const mjModel* model, mjData* data)
{
    const int lock = mj_name2id(model, mjOBJ_EQUALITY, "door_latch_lock");
    if (lock < 0) return;
    const int lever = mj_name2id(model, mjOBJ_JOINT, "door_lever_hinge");
    const int door = mj_name2id(model, mjOBJ_JOINT, "door_hinge");
    if (lever < 0 || door < 0) return;
    const mjtNum lever_angle = data->qpos[model->jnt_qposadr[lever]];
    const mjtNum door_angle = data->qpos[model->jnt_qposadr[door]];
    if (lever_angle >= 0.45) {
        data->eq_active[lock] = 0;
    } else if (mju_abs(door_angle) < 0.01 && lever_angle < 0.15) {
        data->eq_active[lock] = 1;
    }
}
