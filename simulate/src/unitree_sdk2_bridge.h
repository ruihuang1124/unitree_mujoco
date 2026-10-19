#include "reference_recorder.h"
#pragma once

#include <mujoco/mujoco.h>

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/dds_wrapper/robots/go2/go2.h>
#include <unitree/dds_wrapper/robots/g1/g1.h>
#include <unitree/idl/hg/BmsState_.hpp>
#include <unitree/idl/hg/IMUState_.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

#include "param.h"
#include "physics_joystick.h"

#define MOTOR_SENSOR_NUM 3

class UnitreeSDK2BridgeBase
{
public:
    UnitreeSDK2BridgeBase(mjModel *model, mjData *data)
    : mj_model_(model), mj_data_(data)
    {
        _check_sensor();
        if (ee_link7_body_id_ >= 0 || ee_link8_body_id_ >= 0 || ee_quat_body_id_ >= 0 ||
            fake_ee_body_id_ >= 0) {
            std::cout << "[unitree_bridge] EE debug body ids: link7=" << ee_link7_body_id_
                      << " link8=" << ee_link8_body_id_
                      << " quat_body(link6)=" << ee_quat_body_id_
                      << " tracked_body(fake_ee)=" << fake_ee_body_id_ << std::endl;
        }
        if(param::config.print_scene_information == 1) {
            printSceneInformation();
        }
        if(param::config.use_joystick == 1) {
            if(param::config.joystick_type == "xbox") {
                joystick = std::make_shared<XBoxJoystick>(param::config.joystick_device, param::config.joystick_bits);
            } else if(param::config.joystick_type == "switch") {
                joystick  = std::make_shared<SwitchJoystick>(param::config.joystick_device, param::config.joystick_bits);
            } else {
                std::cerr << "Unsupported joystick type: " << param::config.joystick_type << std::endl;
                exit(EXIT_FAILURE);
            }
        }

    }

    virtual void start() {}

    void printSceneInformation()
    {
        auto printObjects = [this](const char* title, int count, int type, auto getIndex) {
            std::cout << "<<------------- " << title << " ------------->> " << std::endl;
            for (int i = 0; i < count; i++) {
                const char* name = mj_id2name(mj_model_, type, i);
                if (name) {
                    std::cout << title << "_index: " << getIndex(i) << ", " << "name: " << name;
                    if (type == mjOBJ_SENSOR) {
                        std::cout << ", dim: " << mj_model_->sensor_dim[i];
                    }
                    std::cout << std::endl;
                }
            }
            std::cout << std::endl;
        };
    
        printObjects("Link", mj_model_->nbody, mjOBJ_BODY, [](int i) { return i; });
        printObjects("Joint", mj_model_->njnt, mjOBJ_JOINT, [](int i) { return i; });
        printObjects("Actuator", mj_model_->nu, mjOBJ_ACTUATOR, [](int i) { return i; });
    
        int sensorIndex = 0;
        printObjects("Sensor", mj_model_->nsensor, mjOBJ_SENSOR, [&](int i) {
            int currentIndex = sensorIndex;
            sensorIndex += mj_model_->sensor_dim[i];
            return currentIndex;
        });
    }

protected:
    int num_motor_ = 0;
    int dim_motor_sensor_ = 0;

    mjData *mj_data_;
    mjModel *mj_model_;

    // Sensor data indices
    int imu_quat_adr_ = -1;
    int imu_gyro_adr_ = -1;
    int imu_acc_adr_ = -1;
    int frame_pos_adr_ = -1;
    int frame_vel_adr_ = -1;
    int ee_link7_body_id_ = -1;
    int ee_link8_body_id_ = -1;
    int ee_quat_body_id_ = -1;
    int fake_ee_body_id_ = -1;
    int ee_debug_print_count_ = 0;

    int secondary_imu_quat_adr_ = -1;
    int secondary_imu_gyro_adr_ = -1;
    int secondary_imu_acc_adr_ = -1;

    std::shared_ptr<unitree::common::UnitreeJoystick> joystick = nullptr;

    void _check_sensor()
    {
        num_motor_ = mj_model_->nu;
        dim_motor_sensor_ = MOTOR_SENSOR_NUM * num_motor_;
    
        // Find sensor addresses by name
        int sensor_id = -1;
        
        // IMU quaternion
        sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "imu_quat");
        if (sensor_id >= 0) {
            imu_quat_adr_ = mj_model_->sensor_adr[sensor_id];
        }
        
        // IMU gyroscope
        sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "imu_gyro");
        if (sensor_id >= 0) {
            imu_gyro_adr_ = mj_model_->sensor_adr[sensor_id];
        }
        
        // IMU accelerometer
        sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "imu_acc");
        if (sensor_id >= 0) {
            imu_acc_adr_ = mj_model_->sensor_adr[sensor_id];
        }
        
        // Frame position
        sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "frame_pos");
        if (sensor_id >= 0) {
            frame_pos_adr_ = mj_model_->sensor_adr[sensor_id];
        }
        
        // Frame velocity
        sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "frame_vel");
        if (sensor_id >= 0) {
            frame_vel_adr_ = mj_model_->sensor_adr[sensor_id];
        }

        // Optional Go2-Piper EE debug payload.  These body ids are used to
        // publish measured MuJoCo EE pose through SportModeState::foot_position_body.
        ee_link7_body_id_ = mj_name2id(mj_model_, mjOBJ_BODY, "link7");
        ee_link8_body_id_ = mj_name2id(mj_model_, mjOBJ_BODY, "link8");
        ee_quat_body_id_ = mj_name2id(mj_model_, mjOBJ_BODY, "link6");
        fake_ee_body_id_ = mj_name2id(mj_model_, mjOBJ_BODY, "fake_ee");

        // Secondary IMU quaternion
        sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "secondary_imu_quat");
        if (sensor_id >= 0) {
            secondary_imu_quat_adr_ = mj_model_->sensor_adr[sensor_id];
        }

        // Secondary IMU gyroscope
        sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "secondary_imu_gyro");
        if (sensor_id >= 0) {
            secondary_imu_gyro_adr_ = mj_model_->sensor_adr[sensor_id];
        }

        // Secondary IMU accelerometer
        sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, "secondary_imu_acc");
        if (sensor_id >= 0) {
            secondary_imu_acc_adr_ = mj_model_->sensor_adr[sensor_id];
        }
    }
};

template <typename LowCmd_t, typename LowState_t>
class RobotBridge : public UnitreeSDK2BridgeBase
{
using HighState_t = unitree::robot::go2::publisher::SportModeState;
using WirelessController_t = unitree::robot::go2::publisher::WirelessController;

public:
    RobotBridge(mjModel *model, mjData *data) : UnitreeSDK2BridgeBase(model, data)
    {
        lowcmd = std::make_shared<LowCmd_t>("rt/lowcmd");
        lowstate = std::make_unique<LowState_t>();
        lowstate->joystick = joystick;
        highstate = std::make_unique<HighState_t>();
        wireless_controller = std::make_unique<WirelessController_t>();
        wireless_controller->joystick = joystick;
        initializeStartupHold();
    }

    void start()
    {
        thread_ = std::make_shared<unitree::common::RecurrentThread>(
            "unitree_bridge", UT_CPU_ID_NONE, 1000, [this]() { this->run(); });
    }

    virtual void run()
    {
        std::unique_lock<std::recursive_mutex> simulation_lock;
        if(param::lmd_sim_mutex) simulation_lock=std::unique_lock<std::recursive_mutex>(*param::lmd_sim_mutex);
        if(!mj_data_) return;
        if(lowstate->joystick) { lowstate->joystick->update(); }
        // lowcmd
        {
            std::lock_guard<std::mutex> lock(lowcmd->mutex_);
            if (mj_data_->time + 1.0e-9 < last_sim_time_) {
                if(param::lmd_sim_mutex) {
                    for(auto& motor:lowcmd->msg_.motor_cmd()) {motor.kp()=0;motor.kd()=1;motor.tau()=0;motor.dq()=0;}
                }
                startup_hold_active_ = startup_hold_enabled_;
                startup_hold_release_logged_ = false;
            }
            last_sim_time_ = mj_data_->time;

            bool position_command_received = false;
            // robot_main always drives the twelve legs in FixStand.  Limit
            // release detection to those channels: the two simulated finger
            // channels are outside its 18-DoF FSM and can contain SDK default
            // values that must not release the hold while still in Passive.
            const int controlled_leg_count = std::min(num_motor_, 12);
            for(int i(0); i<controlled_leg_count; i++) {
                const auto & motor = lowcmd->msg_.motor_cmd()[i];
                if (motor.kp() > 0.1f) {
                    position_command_received = true;
                    break;
                }
            }
            if (startup_hold_active_ && position_command_received) {
                startup_hold_active_ = false;
                if (!startup_hold_release_logged_) {
                    std::cout << "[StartupHold] Position command received; handing control to robot_main."
                              << std::endl;
                    startup_hold_release_logged_ = true;
                }
            }

            if (!reference::directory().empty()) {
                reference::command_sim_time=mj_data_->time;
                reference::command_wall_time=reference::wallSeconds();
                reference::startup_hold=startup_hold_active_;
                for(int i=0;i<std::min(num_motor_,20);++i){
                    const auto& c=lowcmd->msg_.motor_cmd()[i];
                    const double values[5]={c.q(),c.dq(),c.kp(),c.kd(),c.tau()};
                    std::copy(values,values+5,reference::motor_commands.begin()+5*i);
                }
            }
            for(int i(0); i<num_motor_; i++) {
                if (startup_hold_active_ && startup_hold_valid_[i]) {
                    mj_data_->ctrl[i] =
                        param::config.startup_hold_kp *
                            (startup_hold_q_[i] - mj_data_->sensordata[i]) -
                        param::config.startup_hold_kd *
                            mj_data_->sensordata[i + num_motor_];
                } else {
                    auto & motor = lowcmd->msg_.motor_cmd()[i];
                    mj_data_->ctrl[i] = motor.tau() +
                                        motor.kp() * (motor.q() - mj_data_->sensordata[i]) +
                                        motor.kd() * (motor.dq() - mj_data_->sensordata[i + num_motor_]);
                }
            }
        }

        // lowstate
        if(lowstate->trylock()) {
            for(int i(0); i<num_motor_; i++) {
                lowstate->msg_.motor_state()[i].q() = mj_data_->sensordata[i];
                lowstate->msg_.motor_state()[i].dq() = mj_data_->sensordata[i + num_motor_];
                lowstate->msg_.motor_state()[i].tau_est() = mj_data_->sensordata[i + 2 * num_motor_];
            }
            
            if(imu_quat_adr_ >= 0) {
                lowstate->msg_.imu_state().quaternion()[0] = mj_data_->sensordata[imu_quat_adr_ + 0];
                lowstate->msg_.imu_state().quaternion()[1] = mj_data_->sensordata[imu_quat_adr_ + 1];
                lowstate->msg_.imu_state().quaternion()[2] = mj_data_->sensordata[imu_quat_adr_ + 2];
                lowstate->msg_.imu_state().quaternion()[3] = mj_data_->sensordata[imu_quat_adr_ + 3];

                double w = lowstate->msg_.imu_state().quaternion()[0];
                double x = lowstate->msg_.imu_state().quaternion()[1];
                double y = lowstate->msg_.imu_state().quaternion()[2];
                double z = lowstate->msg_.imu_state().quaternion()[3];

                lowstate->msg_.imu_state().rpy()[0] = atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
                lowstate->msg_.imu_state().rpy()[1] = asin(2 * (w * y - z * x));
                lowstate->msg_.imu_state().rpy()[2] = atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
            }
            
            if(imu_gyro_adr_ >= 0) {
                lowstate->msg_.imu_state().gyroscope()[0] = mj_data_->sensordata[imu_gyro_adr_ + 0];
                lowstate->msg_.imu_state().gyroscope()[1] = mj_data_->sensordata[imu_gyro_adr_ + 1];
                lowstate->msg_.imu_state().gyroscope()[2] = mj_data_->sensordata[imu_gyro_adr_ + 2];
            }

            if(imu_acc_adr_ >= 0) {
                lowstate->msg_.imu_state().accelerometer()[0] = mj_data_->sensordata[imu_acc_adr_ + 0];
                lowstate->msg_.imu_state().accelerometer()[1] = mj_data_->sensordata[imu_acc_adr_ + 1];
                lowstate->msg_.imu_state().accelerometer()[2] = mj_data_->sensordata[imu_acc_adr_ + 2];
            }
            
            lowstate->msg_.tick() = std::round(mj_data_->time / 1e-3);
            lowstate->unlockAndPublish();
        }
        // highstate
        if(highstate->trylock()) {
            if(frame_pos_adr_ >= 0) {
                highstate->msg_.position()[0] = mj_data_->sensordata[frame_pos_adr_ + 0];
                highstate->msg_.position()[1] = mj_data_->sensordata[frame_pos_adr_ + 1];
                highstate->msg_.position()[2] = mj_data_->sensordata[frame_pos_adr_ + 2];
            }
            if(frame_vel_adr_ >= 0) {
                highstate->msg_.velocity()[0] = mj_data_->sensordata[frame_vel_adr_ + 0];
                highstate->msg_.velocity()[1] = mj_data_->sensordata[frame_vel_adr_ + 1];
                highstate->msg_.velocity()[2] = mj_data_->sensordata[frame_vel_adr_ + 2];
            }
            auto& ee_debug = highstate->msg_.foot_position_body();
            ee_debug.fill(0.0f);
            if (ee_link7_body_id_ >= 0 && ee_link8_body_id_ >= 0 && ee_quat_body_id_ >= 0) {
                const mjtNum* link7_pos = mj_data_->xpos + 3 * ee_link7_body_id_;
                const mjtNum* link8_pos = mj_data_->xpos + 3 * ee_link8_body_id_;
                const mjtNum* actual_quat = mj_data_->xquat + 4 * ee_quat_body_id_;
                const double actual_x = 0.5 * (link7_pos[0] + link8_pos[0]);
                const double actual_y = 0.5 * (link7_pos[1] + link8_pos[1]);
                const double actual_z = 0.5 * (link7_pos[2] + link8_pos[2]);

                // Match the real OptiTrack path: publish the measured marker
                // rigid-body pose, not the trained virtual EE pose.  The robot
                // deployment applies the calibrated T_fake_actual transform.
                const mjtNum* measured_pos = fake_ee_body_id_ >= 0
                                                 ? mj_data_->xpos + 3 * fake_ee_body_id_
                                                 : nullptr;
                const mjtNum* measured_quat = fake_ee_body_id_ >= 0
                                                  ? mj_data_->xquat + 4 * fake_ee_body_id_
                                                  : actual_quat;
                const double measured_x = measured_pos ? measured_pos[0] : actual_x;
                const double measured_y = measured_pos ? measured_pos[1] : actual_y;
                const double measured_z = measured_pos ? measured_pos[2] : actual_z;

                ee_debug[0] = static_cast<float>(measured_x);
                ee_debug[1] = static_cast<float>(measured_y);
                ee_debug[2] = static_cast<float>(measured_z);
                ee_debug[3] = static_cast<float>(measured_quat[0]);
                ee_debug[4] = static_cast<float>(measured_quat[1]);
                ee_debug[5] = static_cast<float>(measured_quat[2]);
                ee_debug[6] = static_cast<float>(measured_quat[3]);
                // Keep the actual EE position in the auxiliary payload so the
                // visualization/diagnostics can compare both frames directly.
                ee_debug[7] = static_cast<float>(actual_x);
                ee_debug[8] = static_cast<float>(actual_y);
                ee_debug[9] = static_cast<float>(actual_z);
                ee_debug[10] = 1.0f;
                ee_debug[11] = static_cast<float>(mj_data_->time);
                if (ee_debug_print_count_ < 5) {
                    std::cout << "[unitree_bridge] fake EE payload valid measured_pos=("
                              << ee_debug[0] << ", " << ee_debug[1] << ", " << ee_debug[2]
                              << ") quat=(" << ee_debug[3] << ", " << ee_debug[4] << ", "
                              << ee_debug[5] << ", " << ee_debug[6] << ") actual_pos=("
                              << ee_debug[7] << ", " << ee_debug[8] << ", " << ee_debug[9]
                              << ") t="
                              << ee_debug[11] << std::endl;
                    ++ee_debug_print_count_;
                }
            }
            highstate->unlockAndPublish();
        }
        // wireless_controller
        if(wireless_controller->joystick) {
            wireless_controller->unlockAndPublish();
        }
    }

    std::unique_ptr<HighState_t> highstate;
    std::unique_ptr<WirelessController_t> wireless_controller;
    std::shared_ptr<LowCmd_t> lowcmd;
    std::unique_ptr<LowState_t> lowstate;
    
private:
    void initializeStartupHold()
    {
        startup_hold_q_.assign(num_motor_, 0.0);
        startup_hold_valid_.assign(num_motor_, false);
        startup_hold_enabled_ =
            param::config.hold_reset_keyframe_until_control &&
            !param::config.reset_keyframe.empty();
        startup_hold_active_ = startup_hold_enabled_;
        if (!startup_hold_enabled_) {
            return;
        }

        const int keyframe_id = mj_name2id(
            mj_model_, mjOBJ_KEY, param::config.reset_keyframe.c_str());
        if (keyframe_id < 0) {
            startup_hold_enabled_ = false;
            startup_hold_active_ = false;
            std::cerr << "[StartupHold] Keyframe '" << param::config.reset_keyframe
                      << "' is unavailable; startup hold disabled." << std::endl;
            return;
        }

        for (int actuator = 0; actuator < num_motor_; ++actuator) {
            if (mj_model_->actuator_trntype[actuator] != mjTRN_JOINT) {
                continue;
            }
            const int joint = mj_model_->actuator_trnid[2 * actuator];
            if (joint < 0 || joint >= mj_model_->njnt) {
                continue;
            }
            const int qpos_address = mj_model_->jnt_qposadr[joint];
            if (qpos_address < 0 || qpos_address >= mj_model_->nq) {
                continue;
            }
            startup_hold_q_[actuator] =
                mj_model_->key_qpos[keyframe_id * mj_model_->nq + qpos_address];
            startup_hold_valid_[actuator] = true;
        }
        std::cout << "[StartupHold] Holding keyframe '" << param::config.reset_keyframe
                  << "' until robot_main sends a position command (kp="
                  << param::config.startup_hold_kp << ", kd="
                  << param::config.startup_hold_kd << ")." << std::endl;
    }

    unitree::common::RecurrentThreadPtr thread_;
    std::vector<mjtNum> startup_hold_q_;
    std::vector<bool> startup_hold_valid_;
    bool startup_hold_enabled_ = false;
    bool startup_hold_active_ = false;
    bool startup_hold_release_logged_ = false;
    double last_sim_time_ = 0.0;
};

using Go2Bridge = RobotBridge<unitree::robot::go2::subscription::LowCmd, unitree::robot::go2::publisher::LowState>;

class G1Bridge : public RobotBridge<unitree::robot::g1::subscription::LowCmd, unitree::robot::g1::publisher::LowState>
{
public:
    G1Bridge(mjModel *model, mjData *data) : RobotBridge(model, data)
    {
        if (param::config.robot.find("g1") != std::string::npos) {
            auto* g1_lowstate = dynamic_cast<unitree::robot::g1::publisher::LowState*>(lowstate.get());
            if (g1_lowstate) {
                auto scene = param::config.robot_scene.filename().string();
                g1_lowstate->msg_.mode_machine() = scene.find("23") != std::string::npos ? 4 : 5;
            }
        }

        bmsstate = std::make_unique<BmsState_t>("rt/lf/bmsstate");
        bmsstate->msg_.soc() = 100;

        secondary_imustate = std::make_unique<IMUState_t>("rt/secondary_imu");
    }

    void run() override
    {
        RobotBridge::run();

        // secondary IMU state
        if (secondary_imustate->trylock()) {
            if(secondary_imu_quat_adr_ >= 0) {
                secondary_imustate->msg_.quaternion()[0] = mj_data_->sensordata[secondary_imu_quat_adr_ + 0];
                secondary_imustate->msg_.quaternion()[1] = mj_data_->sensordata[secondary_imu_quat_adr_ + 1];
                secondary_imustate->msg_.quaternion()[2] = mj_data_->sensordata[secondary_imu_quat_adr_ + 2];
                secondary_imustate->msg_.quaternion()[3] = mj_data_->sensordata[secondary_imu_quat_adr_ + 3];

                double w = secondary_imustate->msg_.quaternion()[0];
                double x = secondary_imustate->msg_.quaternion()[1];
                double y = secondary_imustate->msg_.quaternion()[2];
                double z = secondary_imustate->msg_.quaternion()[3];

                secondary_imustate->msg_.rpy()[0] = atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
                secondary_imustate->msg_.rpy()[1] = asin(2 * (w * y - z * x));
                secondary_imustate->msg_.rpy()[2] = atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
            }

            if(secondary_imu_gyro_adr_ >= 0) {
                secondary_imustate->msg_.gyroscope()[0] = mj_data_->sensordata[secondary_imu_gyro_adr_ + 0];
                secondary_imustate->msg_.gyroscope()[1] = mj_data_->sensordata[secondary_imu_gyro_adr_ + 1];
                secondary_imustate->msg_.gyroscope()[2] = mj_data_->sensordata[secondary_imu_gyro_adr_ + 2];
            }

            if(secondary_imu_acc_adr_ >= 0) {
                secondary_imustate->msg_.accelerometer()[0] = mj_data_->sensordata[secondary_imu_acc_adr_ + 0];
                secondary_imustate->msg_.accelerometer()[1] = mj_data_->sensordata[secondary_imu_acc_adr_ + 1];
                secondary_imustate->msg_.accelerometer()[2] = mj_data_->sensordata[secondary_imu_acc_adr_ + 2];
            }

            secondary_imustate->unlockAndPublish();
        }

        // In practice, bmsstate is sent at a low frequency; here it is sent with the main loop
        bmsstate->unlockAndPublish();
    }

    using BmsState_t = unitree::robot::RealTimePublisher<unitree_hg::msg::dds_::BmsState_>;
    using IMUState_t = unitree::robot::RealTimePublisher<unitree_hg::msg::dds_::IMUState_>;
    std::unique_ptr<BmsState_t> bmsstate;
    std::unique_ptr<IMUState_t> secondary_imustate;
};
