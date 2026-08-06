// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// !!! hack code: make glfw_adapter.window_ public
#define private public
#include "glfw_adapter.h"
#undef private

#include <algorithm>
#include <chrono>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <thread>

#include <mujoco/mujoco.h>
#include "simulate.h"
#include "array_safety.h"
#include "unitree_sdk2_bridge.h"
#include "param.h"

#define MUJOCO_PLUGIN_DIR "mujoco_plugin"
#define NUM_MOTOR_IDL_GO 20

extern "C"
{
#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <sys/errno.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
}

class ElasticBand
{
public:
  ElasticBand(){};
  void Advance(std::vector<double> x, std::vector<double> dx)
  {
    std::vector<double> delta_x = {0.0, 0.0, 0.0};
    delta_x[0] = point_[0] - x[0];
    delta_x[1] = point_[1] - x[1];
    delta_x[2] = point_[2] - x[2];
    double distance = sqrt(delta_x[0] * delta_x[0] + delta_x[1] * delta_x[1] + delta_x[2] * delta_x[2]);

    std::vector<double> direction = {0.0, 0.0, 0.0};
    direction[0] = delta_x[0] / distance;
    direction[1] = delta_x[1] / distance;
    direction[2] = delta_x[2] / distance;

    double v = dx[0] * direction[0] + dx[1] * direction[1] + dx[2] * direction[2];

    f_[0] = (stiffness_ * (distance - length_) - damping_ * v) * direction[0];
    f_[1] = (stiffness_ * (distance - length_) - damping_ * v) * direction[1];
    f_[2] = (stiffness_ * (distance - length_) - damping_ * v) * direction[2];
  }


  double stiffness_ = 200;
  double damping_ = 100;
  std::vector<double> point_ = {0, 0, 3};
  double length_ = 0.0;
  bool enable_ = true;
  std::vector<double> f_ = {0, 0, 0};
};
inline ElasticBand elastic_band;


namespace
{
  namespace mj = ::mujoco;
  namespace mju = ::mujoco::sample_util;

  // constants
  const double syncMisalign = 0.1;       // maximum mis-alignment before re-sync (simulation seconds)
  const double simRefreshFraction = 0.7; // fraction of refresh available for simulation
  const int kErrorLength = 1024;         // load error string length

  // model and data
  mjModel *m = nullptr;
  mjData *d = nullptr;

  // control noise variables
  mjtNum *ctrlnoise = nullptr;

  using Seconds = std::chrono::duration<double>;

  constexpr int kHighLevelDebugUdpPort = 39001;
  constexpr mjtNum kDebugFrameAxisLength = 0.14;
  constexpr mjtNum kDebugFrameAxisWidth = 0.008;

  struct HighLevelDebugPose
  {
    bool target_valid = false;
    bool sub_target_valid = false;
    bool ee_valid = false;
    std::array<mjtNum, 3> target_pos{0.0, 0.0, 0.0};
    std::array<mjtNum, 4> target_quat{1.0, 0.0, 0.0, 0.0};
    std::array<mjtNum, 3> sub_target_pos{0.0, 0.0, 0.0};
    std::array<mjtNum, 4> sub_target_quat{1.0, 0.0, 0.0, 0.0};
    std::array<mjtNum, 3> ee_pos{0.0, 0.0, 0.0};
    std::array<mjtNum, 4> ee_quat{1.0, 0.0, 0.0, 0.0};
    double progress = 0.0;
    double error = 0.0;
  };

  std::mutex debug_pose_mutex;
  HighLevelDebugPose debug_pose;

  void NormalizeQuat(std::array<mjtNum, 4>& quat)
  {
    const mjtNum norm = std::sqrt(
        quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);
    if (norm < mjMINVAL) {
      quat = {1.0, 0.0, 0.0, 0.0};
      return;
    }
    for (mjtNum& value : quat) {
      value /= norm;
    }
  }

  void AddSphere(mjvScene* scene, const std::array<mjtNum, 3>& pos, mjtNum radius, const float rgba[4])
  {
    if (!scene || scene->ngeom >= scene->maxgeom) {
      return;
    }
    mjtNum size[3] = {radius, radius, radius};
    mjtNum mat[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    mjv_initGeom(&scene->geoms[scene->ngeom++], mjGEOM_SPHERE, size, pos.data(), mat, rgba);
  }

  void AddArrow(
      mjvScene* scene,
      const std::array<mjtNum, 3>& start,
      const std::array<mjtNum, 3>& end,
      mjtNum width,
      const float rgba[4])
  {
    if (!scene || scene->ngeom >= scene->maxgeom) {
      return;
    }
    mjvGeom* geom = &scene->geoms[scene->ngeom++];
    mjv_initGeom(geom, mjGEOM_ARROW, nullptr, nullptr, nullptr, rgba);
    mjv_connector(geom, mjGEOM_ARROW, width, start.data(), end.data());
    for (int i = 0; i < 4; ++i) {
      geom->rgba[i] = rgba[i];
    }
  }

  void AddFrame(
      mjvScene* scene,
      const std::array<mjtNum, 3>& pos,
      const std::array<mjtNum, 4>& quat,
      mjtNum length,
      mjtNum width,
      bool target)
  {
    std::array<mjtNum, 4> q = quat;
    NormalizeQuat(q);
    mjtNum rot[9];
    mju_quat2Mat(rot, q.data());

    const float red[4] = {1.0f, target ? 0.15f : 0.0f, 0.0f, 1.0f};
    const float green[4] = {0.0f, 1.0f, target ? 0.15f : 0.0f, 1.0f};
    const float blue[4] = {0.0f, target ? 0.25f : 0.0f, 1.0f, 1.0f};
    const float* colors[3] = {red, green, blue};

    for (int axis = 0; axis < 3; ++axis) {
      std::array<mjtNum, 3> end = pos;
      for (int row = 0; row < 3; ++row) {
        end[row] += length * rot[3 * row + axis];
      }
      AddArrow(scene, pos, end, width, colors[axis]);
    }
  }

  void UpdateDebugPoseFromPacket(const char* buffer)
  {
    int version = 0;
    HighLevelDebugPose pose;
    int matched = std::sscanf(
        buffer,
        "HLCDBG %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
        &version,
        &pose.target_pos[0],
        &pose.target_pos[1],
        &pose.target_pos[2],
        &pose.target_quat[0],
        &pose.target_quat[1],
        &pose.target_quat[2],
        &pose.target_quat[3],
        &pose.sub_target_pos[0],
        &pose.sub_target_pos[1],
        &pose.sub_target_pos[2],
        &pose.sub_target_quat[0],
        &pose.sub_target_quat[1],
        &pose.sub_target_quat[2],
        &pose.sub_target_quat[3],
        &pose.ee_pos[0],
        &pose.ee_pos[1],
        &pose.ee_pos[2],
        &pose.ee_quat[0],
        &pose.ee_quat[1],
        &pose.ee_quat[2],
        &pose.ee_quat[3],
        &pose.progress,
        &pose.error);
    if (version == 2 && matched == 24) {
      pose.sub_target_valid = true;
      NormalizeQuat(pose.sub_target_quat);
    } else {
      matched = std::sscanf(
          buffer,
          "HLCDBG %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
          &version,
          &pose.target_pos[0],
          &pose.target_pos[1],
          &pose.target_pos[2],
          &pose.target_quat[0],
          &pose.target_quat[1],
          &pose.target_quat[2],
          &pose.target_quat[3],
          &pose.ee_pos[0],
          &pose.ee_pos[1],
          &pose.ee_pos[2],
          &pose.ee_quat[0],
          &pose.ee_quat[1],
          &pose.ee_quat[2],
          &pose.ee_quat[3],
          &pose.progress,
          &pose.error);
      if (version != 1 || matched != 17) {
        return;
      }
    }

    pose.target_valid = true;
    pose.ee_valid = true;
    NormalizeQuat(pose.target_quat);
    NormalizeQuat(pose.ee_quat);

    std::lock_guard<std::mutex> lock(debug_pose_mutex);
    debug_pose = pose;
  }

  void HighLevelDebugUdpThread()
  {
    int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
      std::cerr << "[DebugVis] Failed to open UDP socket: " << std::strerror(errno) << std::endl;
      return;
    }

    int reuse = 1;
    (void)::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(kHighLevelDebugUdpPort);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
      std::cerr << "[DebugVis] Failed to bind UDP port " << kHighLevelDebugUdpPort
                << ": " << std::strerror(errno) << std::endl;
      ::close(fd);
      return;
    }

    std::cout << "[DebugVis] Listening for HLC target poses on UDP port "
              << kHighLevelDebugUdpPort << std::endl;

    char buffer[512];
    while (true) {
      const ssize_t n = ::recv(fd, buffer, sizeof(buffer) - 1, 0);
      if (n <= 0) {
        continue;
      }
      buffer[n] = '\0';
      UpdateDebugPoseFromPacket(buffer);
    }
  }

  //---------------------------------------- plugin handling -----------------------------------------

  // return the path to the directory containing the current executable
  // used to determine the location of auto-loaded plugin libraries
  std::string getExecutableDir()
  {
#if defined(_WIN32) || defined(__CYGWIN__)
    constexpr char kPathSep = '\\';
    std::string realpath = [&]() -> std::string
    {
      std::unique_ptr<char[]> realpath(nullptr);
      DWORD buf_size = 128;
      bool success = false;
      while (!success)
      {
        realpath.reset(new (std::nothrow) char[buf_size]);
        if (!realpath)
        {
          std::cerr << "cannot allocate memory to store executable path\n";
          return "";
        }

        DWORD written = GetModuleFileNameA(nullptr, realpath.get(), buf_size);
        if (written < buf_size)
        {
          success = true;
        }
        else if (written == buf_size)
        {
          // realpath is too small, grow and retry
          buf_size *= 2;
        }
        else
        {
          std::cerr << "failed to retrieve executable path: " << GetLastError() << "\n";
          return "";
        }
      }
      return realpath.get();
    }();
#else
    constexpr char kPathSep = '/';
#if defined(__APPLE__)
    std::unique_ptr<char[]> buf(nullptr);
    {
      std::uint32_t buf_size = 0;
      _NSGetExecutablePath(nullptr, &buf_size);
      buf.reset(new char[buf_size]);
      if (!buf)
      {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }
      if (_NSGetExecutablePath(buf.get(), &buf_size))
      {
        std::cerr << "unexpected error from _NSGetExecutablePath\n";
      }
    }
    const char *path = buf.get();
#else
    const char *path = "/proc/self/exe";
#endif
    std::string realpath = [&]() -> std::string
    {
      std::unique_ptr<char[]> realpath(nullptr);
      std::uint32_t buf_size = 128;
      bool success = false;
      while (!success)
      {
        realpath.reset(new (std::nothrow) char[buf_size]);
        if (!realpath)
        {
          std::cerr << "cannot allocate memory to store executable path\n";
          return "";
        }

        std::size_t written = readlink(path, realpath.get(), buf_size);
        if (written < buf_size)
        {
          realpath.get()[written] = '\0';
          success = true;
        }
        else if (written == -1)
        {
          if (errno == EINVAL)
          {
            // path is already not a symlink, just use it
            return path;
          }

          std::cerr << "error while resolving executable path: " << strerror(errno) << '\n';
          return "";
        }
        else
        {
          // realpath is too small, grow and retry
          buf_size *= 2;
        }
      }
      return realpath.get();
    }();
#endif

    if (realpath.empty())
    {
      return "";
    }

    for (std::size_t i = realpath.size() - 1; i > 0; --i)
    {
      if (realpath.c_str()[i] == kPathSep)
      {
        return realpath.substr(0, i);
      }
    }

    // don't scan through the entire file system's root
    return "";
  }

  // scan for libraries in the plugin directory to load additional plugins
  void scanPluginLibraries()
  {
    // check and print plugins that are linked directly into the executable
    int nplugin = mjp_pluginCount();
    if (nplugin)
    {
      std::printf("Built-in plugins:\n");
      for (int i = 0; i < nplugin; ++i)
      {
        std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
      }
    }

    // define platform-specific strings
#if defined(_WIN32) || defined(__CYGWIN__)
    const std::string sep = "\\";
#else
    const std::string sep = "/";
#endif

    // try to open the ${EXECDIR}/plugin directory
    // ${EXECDIR} is the directory containing the simulate binary itself
    const std::string executable_dir = getExecutableDir();
    if (executable_dir.empty())
    {
      return;
    }

    const std::string plugin_dir = getExecutableDir() + sep + MUJOCO_PLUGIN_DIR;
    mj_loadAllPluginLibraries(
        plugin_dir.c_str(), +[](const char *filename, int first, int count)
                            {
        std::printf("Plugins registered by library '%s':\n", filename);
        for (int i = first; i < first + count; ++i) {
          std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
        } });
  }

  //------------------------------------------- simulation -------------------------------------------

  mjModel *LoadModel(const char *file, mj::Simulate &sim)
  {
    // this copy is needed so that the mju::strlen call below compiles
    char filename[mj::Simulate::kMaxFilenameLength];
    mju::strcpy_arr(filename, file);

    // make sure filename is not empty
    if (!filename[0])
    {
      return nullptr;
    }

    // load and compile
    char loadError[kErrorLength] = "";
    mjModel *mnew = 0;
    if (mju::strlen_arr(filename) > 4 &&
        !std::strncmp(filename + mju::strlen_arr(filename) - 4, ".mjb",
                      mju::sizeof_arr(filename) - mju::strlen_arr(filename) + 4))
    {
      mnew = mj_loadModel(filename, nullptr);
      if (!mnew)
      {
        mju::strcpy_arr(loadError, "could not load binary model");
      }
    }
    else
    {
      mnew = mj_loadXML(filename, nullptr, loadError, kErrorLength);
      // remove trailing newline character from loadError
      if (loadError[0])
      {
        int error_length = mju::strlen_arr(loadError);
        if (loadError[error_length - 1] == '\n')
        {
          loadError[error_length - 1] = '\0';
        }
      }
    }

    mju::strcpy_arr(sim.load_error, loadError);

    if (!mnew)
    {
      std::printf("%s\n", loadError);
      return nullptr;
    }

    // compiler warning: print and pause
    if (loadError[0])
    {
      // mj_forward() below will print the warning message
      std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
      sim.run = 0;
    }

    return mnew;
  }

  // Publish task ground-truth poses through atomic YAML files.  The original
  // single-object file remains stable for key-4 pick/place.  The scene file is
  // a replaceable perception interface for the door, wiping and drawer tasks.
  void PublishTaskObjectPose(const mjModel* model, const mjData* data)
  {
    constexpr double kPublishPeriodS = 0.02;
    constexpr const char* kPickBodyName = "pick_object";
    constexpr const char* kPickPoseFile = "/tmp/mujoco_pick_object_pose.yaml";
    constexpr const char* kPickTempFile = "/tmp/mujoco_pick_object_pose.yaml.tmp";
    constexpr const char* kScenePoseFile = "/tmp/mujoco_scene_task_poses.yaml";
    constexpr const char* kSceneTempFile = "/tmp/mujoco_scene_task_poses.yaml.tmp";
    constexpr std::array<const char*, 10> kSceneBodyNames = {
        "door_handle",
        "table_eraser",
        "board_eraser",
        "table_wipe_frame",
        "blackboard_wipe_frame",
        "drawer_handle",
        "drawer_bottle",
        "drawer_drop_frame",
        "trolley_handle",
        "trolley_pull_target"};
    // A body may expose the coordinate of the articulated joint that moves it.
    // nullptr means that only the world pose is published for that body.
    constexpr std::array<const char*, kSceneBodyNames.size()> kSceneJointNames = {
        "door_hinge", nullptr, nullptr, nullptr, nullptr, "drawer_slide", nullptr, nullptr,
        nullptr, nullptr};

    static const mjModel* cached_model = nullptr;
    static int pick_body_id = -1;
    static std::array<int, kSceneBodyNames.size()> scene_body_ids{};
    static std::array<int, kSceneBodyNames.size()> scene_joint_ids{};
    static double last_publish_time_s = -kPublishPeriodS;
    if (!model || !data) {
      return;
    }
    if (cached_model != model) {
      cached_model = model;
      pick_body_id = mj_name2id(model, mjOBJ_BODY, kPickBodyName);
      for (size_t i = 0; i < kSceneBodyNames.size(); ++i) {
        scene_body_ids[i] = mj_name2id(model, mjOBJ_BODY, kSceneBodyNames[i]);
        scene_joint_ids[i] = kSceneJointNames[i]
                                 ? mj_name2id(model, mjOBJ_JOINT, kSceneJointNames[i])
                                 : -1;
      }
      last_publish_time_s = -kPublishPeriodS;
      if (pick_body_id >= 0) {
        std::cout << "[PickPlace] Publishing '" << kPickBodyName << "' world pose to "
                  << kPickPoseFile << std::endl;
      }
      if (std::any_of(
              scene_body_ids.begin(),
              scene_body_ids.end(),
              [](int id) { return id >= 0; })) {
        std::cout << "[SceneTasks] Publishing door/drawer/trolley/tool/surface poses to "
                  << kScenePoseFile << std::endl;
      }
    }
    const bool has_scene_objects = std::any_of(
        scene_body_ids.begin(), scene_body_ids.end(), [](int id) { return id >= 0; });
    if (pick_body_id < 0 && !has_scene_objects) {
      return;
    }
    if (data->time >= last_publish_time_s && data->time - last_publish_time_s < kPublishPeriodS) {
      return;
    }
    last_publish_time_s = data->time;

    if (pick_body_id >= 0) {
      const mjtNum* pos = data->xpos + 3 * pick_body_id;
      const mjtNum* quat = data->xquat + 4 * pick_body_id;
      mjtNum velocity[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      mj_objectVelocity(model, data, mjOBJ_BODY, pick_body_id, velocity, 0);
      std::ofstream out(kPickTempFile, std::ios::trunc);
      if (out.is_open()) {
        out << std::fixed << std::setprecision(7);
        out << "body_name: " << kPickBodyName << "\n";
        out << "sim_time_s: " << data->time << "\n";
        out << "position_w: [" << pos[0] << ", " << pos[1] << ", " << pos[2] << "]\n";
        out << "quaternion_wxyz: [" << quat[0] << ", " << quat[1] << ", " << quat[2] << ", " << quat[3] << "]\n";
        out << "linear_velocity_w: [" << velocity[3] << ", " << velocity[4] << ", "
            << velocity[5] << "]\n";
        out.close();
        (void)std::rename(kPickTempFile, kPickPoseFile);
      }
    }

    if (has_scene_objects) {
      std::ofstream out(kSceneTempFile, std::ios::trunc);
      if (!out.is_open()) {
        return;
      }
      out << std::fixed << std::setprecision(7);
      out << "sim_time_s: " << data->time << "\n";
      out << "objects:\n";
      for (size_t i = 0; i < kSceneBodyNames.size(); ++i) {
        const int body_id = scene_body_ids[i];
        if (body_id < 0) {
          continue;
        }
        const mjtNum* pos = data->xpos + 3 * body_id;
        const mjtNum* quat = data->xquat + 4 * body_id;
        out << "  " << kSceneBodyNames[i] << ":\n";
        out << "    position_w: [" << pos[0] << ", " << pos[1] << ", " << pos[2] << "]\n";
        out << "    quaternion_wxyz: [" << quat[0] << ", " << quat[1] << ", "
            << quat[2] << ", " << quat[3] << "]\n";
        if (scene_joint_ids[i] >= 0) {
          const int qpos_address = model->jnt_qposadr[scene_joint_ids[i]];
          out << "    joint_name: " << kSceneJointNames[i] << "\n";
          out << "    joint_position: " << data->qpos[qpos_address] << "\n";
        }
      }
      out.close();
      (void)std::rename(kSceneTempFile, kScenePoseFile);
    }
  }

  // Drive the dedicated dynamic-pick carrier in 3D without adding actuators
  // (and therefore without changing the robot actuator indexing expected by
  // UnitreeSdk2Bridge).  X retains the variable-speed shuttle motion while Y
  // and Z follow independent smooth sinusoids.  Different frequencies produce
  // a video-friendly spatial path without requiring a future trajectory model
  // in the robot-side visual servo.
  void UpdateMovingPickPlatform(const mjModel* model, mjData* data)
  {
    constexpr const char* kJointNameX = "moving_platform_slide";
    constexpr const char* kJointNameY = "moving_platform_slide_y";
    constexpr const char* kJointNameZ = "moving_platform_slide_z";
    constexpr const char* kSpeedNumericName = "moving_platform_speed";
    constexpr const char* kSpeedAmplitudeNumericName = "moving_platform_speed_amplitude";
    constexpr const char* kSpeedFrequencyNumericName = "moving_platform_speed_frequency_hz";
    constexpr const char* kYAmplitudeNumericName = "moving_platform_y_amplitude_m";
    constexpr const char* kYFrequencyNumericName = "moving_platform_y_frequency_hz";
    constexpr const char* kYPhaseNumericName = "moving_platform_y_phase_rad";
    constexpr const char* kZAmplitudeNumericName = "moving_platform_z_amplitude_m";
    constexpr const char* kZFrequencyNumericName = "moving_platform_z_frequency_hz";
    constexpr const char* kZPhaseNumericName = "moving_platform_z_phase_rad";
    constexpr mjtNum kFallbackSpeedMps = 0.06;
    constexpr mjtNum kLimitMarginM = 1.0e-4;

    static const mjModel* cached_model = nullptr;
    static std::array<int, 3> joint_ids{-1, -1, -1};
    static std::array<int, 3> qpos_addresses{-1, -1, -1};
    static std::array<int, 3> dof_addresses{-1, -1, -1};
    static mjtNum speed_mps = kFallbackSpeedMps;
    static mjtNum speed_amplitude_mps = 0.0;
    static mjtNum speed_frequency_hz = 0.0;
    static mjtNum y_amplitude_m = 0.0;
    static mjtNum y_frequency_hz = 0.0;
    static mjtNum y_phase_rad = 0.0;
    static mjtNum z_amplitude_m = 0.0;
    static mjtNum z_frequency_hz = 0.0;
    static mjtNum z_phase_rad = 0.0;
    static mjtNum x_direction = 1.0;
    static mjtNum last_sim_time_s = -1.0;

    if (!model || !data) {
      return;
    }
    const bool data_reset =
        last_sim_time_s >= 0.0 && data->time + mjMINVAL < last_sim_time_s;
    if (cached_model != model || data_reset) {
      cached_model = model;
      joint_ids = {
          mj_name2id(model, mjOBJ_JOINT, kJointNameX),
          mj_name2id(model, mjOBJ_JOINT, kJointNameY),
          mj_name2id(model, mjOBJ_JOINT, kJointNameZ)};
      qpos_addresses = {-1, -1, -1};
      dof_addresses = {-1, -1, -1};
      speed_mps = kFallbackSpeedMps;
      speed_amplitude_mps = 0.0;
      speed_frequency_hz = 0.0;
      y_amplitude_m = 0.0;
      y_frequency_hz = 0.0;
      y_phase_rad = 0.0;
      z_amplitude_m = 0.0;
      z_frequency_hz = 0.0;
      z_phase_rad = 0.0;
      x_direction = 1.0;

      const bool joints_valid = std::all_of(
          joint_ids.begin(),
          joint_ids.end(),
          [model](int id) {
            return id >= 0 && model->jnt_type[id] == mjJNT_SLIDE;
          });
      if (joints_valid) {
        for (size_t axis = 0; axis < joint_ids.size(); ++axis) {
          qpos_addresses[axis] = model->jnt_qposadr[joint_ids[axis]];
          dof_addresses[axis] = model->jnt_dofadr[joint_ids[axis]];
        }
        const auto numericValue = [model](const char* name, mjtNum fallback) {
          const int numeric_id = mj_name2id(model, mjOBJ_NUMERIC, name);
          if (numeric_id < 0 || model->numeric_size[numeric_id] < 1) {
            return fallback;
          }
          return model->numeric_data[model->numeric_adr[numeric_id]];
        };
        speed_mps = std::abs(numericValue(kSpeedNumericName, kFallbackSpeedMps));
        speed_amplitude_mps =
            std::abs(numericValue(kSpeedAmplitudeNumericName, 0.0));
        speed_frequency_hz =
            std::abs(numericValue(kSpeedFrequencyNumericName, 0.0));
        y_amplitude_m = std::abs(numericValue(kYAmplitudeNumericName, 0.0));
        y_frequency_hz = std::abs(numericValue(kYFrequencyNumericName, 0.0));
        y_phase_rad = numericValue(kYPhaseNumericName, 0.0);
        z_amplitude_m = std::abs(numericValue(kZAmplitudeNumericName, 0.0));
        z_frequency_hz = std::abs(numericValue(kZFrequencyNumericName, 0.0));
        z_phase_rad = numericValue(kZPhaseNumericName, 0.0);
        if (speed_mps < mjMINVAL) {
          speed_mps = kFallbackSpeedMps;
        }
        // Keep the shuttle moving in its current direction even at the
        // sinusoid's minimum.
        speed_amplitude_mps =
            std::min(speed_amplitude_mps, std::max(speed_mps - 0.005, 0.0));
        y_amplitude_m = std::min(
            y_amplitude_m,
            std::abs(model->jnt_range[2 * joint_ids[1] + 1]));
        z_amplitude_m = std::min(
            z_amplitude_m,
            std::abs(model->jnt_range[2 * joint_ids[2] + 1]));
        std::cout << "[DynamicPickPlace] Driving 3D invisible carrier: X speed="
                  << speed_mps << " +/- " << speed_amplitude_mps
                  << " m/s; Y amplitude/frequency=" << y_amplitude_m
                  << " m/" << y_frequency_hz
                  << " Hz; Z amplitude/frequency=" << z_amplitude_m
                  << " m/" << z_frequency_hz << " Hz" << std::endl;
      }
    }
    last_sim_time_s = data->time;

    const bool addresses_valid =
        std::all_of(
            qpos_addresses.begin(),
            qpos_addresses.end(),
            [model](int address) { return address >= 0 && address < model->nq; }) &&
        std::all_of(
            dof_addresses.begin(),
            dof_addresses.end(),
            [model](int address) { return address >= 0 && address < model->nv; });
    if (!addresses_valid) {
      return;
    }

    const mjtNum x_lower = model->jnt_range[2 * joint_ids[0]];
    const mjtNum x_upper = model->jnt_range[2 * joint_ids[0] + 1];
    if (data->qpos[qpos_addresses[0]] <= x_lower + kLimitMarginM) {
      x_direction = 1.0;
    } else if (data->qpos[qpos_addresses[0]] >= x_upper - kLimitMarginM) {
      x_direction = -1.0;
    }
    const mjtNum x_speed_profile =
        speed_mps +
        speed_amplitude_mps *
            std::sin(2.0 * mjPI * speed_frequency_hz * data->time);
    data->qvel[dof_addresses[0]] = x_direction * x_speed_profile;

    const auto setSinusoidalAxis = [model, data](
                                       int joint_id,
                                       int qpos_address,
                                       int dof_address,
                                       mjtNum amplitude_m,
                                       mjtNum frequency_hz,
                                       mjtNum phase_rad) {
      const mjtNum omega = 2.0 * mjPI * frequency_hz;
      const mjtNum phase = omega * data->time + phase_rad;
      const mjtNum lower = model->jnt_range[2 * joint_id];
      const mjtNum upper = model->jnt_range[2 * joint_id + 1];
      data->qpos[qpos_address] =
          std::clamp(amplitude_m * std::sin(phase), lower, upper);
      data->qvel[dof_address] = amplitude_m * omega * std::cos(phase);
    };
    setSinusoidalAxis(
        joint_ids[1],
        qpos_addresses[1],
        dof_addresses[1],
        y_amplitude_m,
        y_frequency_hz,
        y_phase_rad);
    setSinusoidalAxis(
        joint_ids[2],
        qpos_addresses[2],
        dof_addresses[2],
        z_amplitude_m,
        z_frequency_hz,
        z_phase_rad);
  }

  std::array<mjtNum, 4> QuatConjugate(const std::array<mjtNum, 4>& quat)
  {
    return {quat[0], -quat[1], -quat[2], -quat[3]};
  }

  std::array<mjtNum, 4> QuatMultiply(
      const std::array<mjtNum, 4>& lhs,
      const std::array<mjtNum, 4>& rhs)
  {
    return {
        lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2] - lhs[3] * rhs[3],
        lhs[0] * rhs[1] + lhs[1] * rhs[0] + lhs[2] * rhs[3] - lhs[3] * rhs[2],
        lhs[0] * rhs[2] - lhs[1] * rhs[3] + lhs[2] * rhs[0] + lhs[3] * rhs[1],
        lhs[0] * rhs[3] + lhs[1] * rhs[2] - lhs[2] * rhs[1] + lhs[3] * rhs[0]};
  }

  std::array<mjtNum, 3> QuatRotate(
      const std::array<mjtNum, 4>& quat,
      const std::array<mjtNum, 3>& vector)
  {
    const std::array<mjtNum, 4> vector_quat = {0.0, vector[0], vector[1], vector[2]};
    const std::array<mjtNum, 4> rotated =
        QuatMultiply(QuatMultiply(quat, vector_quat), QuatConjugate(quat));
    return {rotated[1], rotated[2], rotated[3]};
  }

  // Simulation-only grasp aid for the dedicated blackboard-wiping scene.
  //
  // A light free body held by two mesh fingers is very sensitive to contact
  // tuning and small sim2sim tracking errors.  Once the fingers start closing
  // near the eraser, preserve the current eraser-to-gripper transform and move
  // the eraser with the gripper.  Requiring blackboard_eraser_tray keeps this
  // behavior out of the other task scenes and all robot-side control code.
  void UpdateBlackboardEraserGraspLatch(const mjModel* model, mjData* data)
  {
    constexpr const char* kSceneMarkerGeomName = "blackboard_eraser_tray";
    constexpr const char* kEraserBodyName = "board_eraser";
    constexpr const char* kEraserJointName = "board_eraser_freejoint";
    constexpr const char* kGripperBodyName = "gripper_base";
    constexpr const char* kFingerBodyNameA = "link7";
    constexpr const char* kFingerBodyNameB = "link8";
    constexpr const char* kFingerJointNameA = "joint7";
    constexpr const char* kFingerJointNameB = "joint8";
    constexpr mjtNum kAttachDistanceM = 0.065;
    // Fully open is approximately 0.070 m.  The 50 mm-wide eraser can stop the
    // physical fingers before they reach the nominal zero-position command.
    constexpr mjtNum kClosingOpeningM = 0.064;

    static const mjModel* cached_model = nullptr;
    static int scene_marker_geom_id = -1;
    static int eraser_body_id = -1;
    static int eraser_joint_id = -1;
    static int gripper_body_id = -1;
    static int finger_body_id_a = -1;
    static int finger_body_id_b = -1;
    static int finger_joint_id_a = -1;
    static int finger_joint_id_b = -1;
    static bool attached = false;
    static mjtNum last_sim_time_s = -1.0;
    static std::array<mjtNum, 3> eraser_pos_gripper{0.0, 0.0, 0.0};
    static std::array<mjtNum, 4> eraser_quat_gripper{1.0, 0.0, 0.0, 0.0};

    if (!model || !data) {
      return;
    }

    const bool data_reset =
        last_sim_time_s >= 0.0 && data->time + mjMINVAL < last_sim_time_s;
    if (cached_model != model || data_reset) {
      cached_model = model;
      scene_marker_geom_id = mj_name2id(model, mjOBJ_GEOM, kSceneMarkerGeomName);
      eraser_body_id = mj_name2id(model, mjOBJ_BODY, kEraserBodyName);
      eraser_joint_id = mj_name2id(model, mjOBJ_JOINT, kEraserJointName);
      gripper_body_id = mj_name2id(model, mjOBJ_BODY, kGripperBodyName);
      finger_body_id_a = mj_name2id(model, mjOBJ_BODY, kFingerBodyNameA);
      finger_body_id_b = mj_name2id(model, mjOBJ_BODY, kFingerBodyNameB);
      finger_joint_id_a = mj_name2id(model, mjOBJ_JOINT, kFingerJointNameA);
      finger_joint_id_b = mj_name2id(model, mjOBJ_JOINT, kFingerJointNameB);
      attached = false;
      eraser_pos_gripper = {0.0, 0.0, 0.0};
      eraser_quat_gripper = {1.0, 0.0, 0.0, 0.0};
    }
    last_sim_time_s = data->time;

    if (scene_marker_geom_id < 0 || eraser_body_id < 0 || eraser_joint_id < 0 ||
        gripper_body_id < 0 || finger_body_id_a < 0 || finger_body_id_b < 0 ||
        finger_joint_id_a < 0 || finger_joint_id_b < 0 ||
        model->jnt_type[eraser_joint_id] != mjJNT_FREE) {
      return;
    }

    const int eraser_qpos_adr = model->jnt_qposadr[eraser_joint_id];
    const int eraser_dof_adr = model->jnt_dofadr[eraser_joint_id];
    const int finger_qpos_adr_a = model->jnt_qposadr[finger_joint_id_a];
    const int finger_qpos_adr_b = model->jnt_qposadr[finger_joint_id_b];
    if (eraser_qpos_adr < 0 || eraser_qpos_adr + 6 >= model->nq ||
        eraser_dof_adr < 0 || eraser_dof_adr + 5 >= model->nv ||
        finger_qpos_adr_a < 0 || finger_qpos_adr_a >= model->nq ||
        finger_qpos_adr_b < 0 || finger_qpos_adr_b >= model->nq) {
      return;
    }

    const mjtNum* gripper_pos_raw = data->xpos + 3 * gripper_body_id;
    const mjtNum* gripper_quat_raw = data->xquat + 4 * gripper_body_id;
    const mjtNum* eraser_pos_raw = data->xpos + 3 * eraser_body_id;
    const mjtNum* eraser_quat_raw = data->xquat + 4 * eraser_body_id;
    const mjtNum* finger_pos_a = data->xpos + 3 * finger_body_id_a;
    const mjtNum* finger_pos_b = data->xpos + 3 * finger_body_id_b;

    std::array<mjtNum, 4> gripper_quat = {
        gripper_quat_raw[0], gripper_quat_raw[1],
        gripper_quat_raw[2], gripper_quat_raw[3]};
    NormalizeQuat(gripper_quat);

    if (!attached) {
      const std::array<mjtNum, 3> finger_center = {
          0.5 * (finger_pos_a[0] + finger_pos_b[0]),
          0.5 * (finger_pos_a[1] + finger_pos_b[1]),
          0.5 * (finger_pos_a[2] + finger_pos_b[2])};
      const std::array<mjtNum, 3> finger_to_eraser = {
          eraser_pos_raw[0] - finger_center[0],
          eraser_pos_raw[1] - finger_center[1],
          eraser_pos_raw[2] - finger_center[2]};
      const mjtNum distance_m = std::sqrt(
          finger_to_eraser[0] * finger_to_eraser[0] +
          finger_to_eraser[1] * finger_to_eraser[1] +
          finger_to_eraser[2] * finger_to_eraser[2]);
      const mjtNum finger_opening_m =
          std::abs(data->qpos[finger_qpos_adr_a]) +
          std::abs(data->qpos[finger_qpos_adr_b]);

      if (distance_m <= kAttachDistanceM && finger_opening_m <= kClosingOpeningM) {
        const std::array<mjtNum, 4> gripper_quat_inv = QuatConjugate(gripper_quat);
        const std::array<mjtNum, 3> gripper_to_eraser = {
            eraser_pos_raw[0] - gripper_pos_raw[0],
            eraser_pos_raw[1] - gripper_pos_raw[1],
            eraser_pos_raw[2] - gripper_pos_raw[2]};
        const std::array<mjtNum, 4> eraser_quat = {
            eraser_quat_raw[0], eraser_quat_raw[1],
            eraser_quat_raw[2], eraser_quat_raw[3]};
        eraser_pos_gripper = QuatRotate(gripper_quat_inv, gripper_to_eraser);
        eraser_quat_gripper = QuatMultiply(gripper_quat_inv, eraser_quat);
        NormalizeQuat(eraser_quat_gripper);
        attached = true;
        std::cout << "[BlackboardWipe] Simulation grasp latch attached board_eraser"
                  << " (distance=" << distance_m
                  << " m, finger_opening=" << finger_opening_m << " m)"
                  << std::endl;
      }
    }

    if (!attached) {
      return;
    }

    const std::array<mjtNum, 3> eraser_offset_w =
        QuatRotate(gripper_quat, eraser_pos_gripper);
    std::array<mjtNum, 4> eraser_quat_w =
        QuatMultiply(gripper_quat, eraser_quat_gripper);
    NormalizeQuat(eraser_quat_w);

    data->qpos[eraser_qpos_adr + 0] = gripper_pos_raw[0] + eraser_offset_w[0];
    data->qpos[eraser_qpos_adr + 1] = gripper_pos_raw[1] + eraser_offset_w[1];
    data->qpos[eraser_qpos_adr + 2] = gripper_pos_raw[2] + eraser_offset_w[2];
    for (int i = 0; i < 4; ++i) {
      data->qpos[eraser_qpos_adr + 3 + i] = eraser_quat_w[i];
    }
    for (int i = 0; i < 6; ++i) {
      data->qvel[eraser_dof_adr + i] = 0.0;
    }

    // Refresh body transforms immediately so task-pose publishing and the
    // renderer see the latched pose from this same physics step.
    mj_forward(model, data);
  }

  void PublishMujocoPairDiagnostic(const mjModel* model, const mjData* data)
  {
    constexpr const char* kBaseBodyName = "base_link";
    constexpr const char* kEeOrientationBodyName = "link6";
    constexpr const char* kEeFingerBodyNameA = "link7";
    constexpr const char* kEeFingerBodyNameB = "link8";

    static const mjModel* cached_model = nullptr;
    static int base_body_id = -1;
    static int ee_orientation_body_id = -1;
    static int ee_finger_body_id_a = -1;
    static int ee_finger_body_id_b = -1;
    static double last_publish_time_s = -1.0;
    static std::uint64_t seq = 0;
    static bool warned_missing_body = false;
    static std::ofstream out;

    if (!param::config.record_mocap_pair) {
      return;
    }
    if (!model || !data) {
      return;
    }
    const double publish_rate_hz =
        param::config.mocap_pair_rate > 0.0 ? param::config.mocap_pair_rate : 50.0;
    const double publish_period_s = 1.0 / publish_rate_hz;
    const std::string& csv_file = param::config.mocap_pair_csv;

    if (cached_model != model) {
      cached_model = model;
      base_body_id = mj_name2id(model, mjOBJ_BODY, kBaseBodyName);
      ee_orientation_body_id = mj_name2id(model, mjOBJ_BODY, kEeOrientationBodyName);
      ee_finger_body_id_a = mj_name2id(model, mjOBJ_BODY, kEeFingerBodyNameA);
      ee_finger_body_id_b = mj_name2id(model, mjOBJ_BODY, kEeFingerBodyNameB);
      last_publish_time_s = -publish_period_s;
      seq = 0;
      warned_missing_body = false;
      out.close();

      if (base_body_id >= 0 && ee_orientation_body_id >= 0 &&
          ee_finger_body_id_a >= 0 && ee_finger_body_id_b >= 0) {
        out.open(csv_file, std::ios::trunc);
        if (out.is_open()) {
          out << "time_s,base_age_s,ee_age_s,base_seq,ee_seq,"
              << "base_px,base_py,base_pz,base_qx,base_qy,base_qz,base_qw,"
              << "ee_px,ee_py,ee_pz,ee_qx,ee_qy,ee_qz,ee_qw,"
              << "ee_in_base_x,ee_in_base_y,ee_in_base_z,"
              << "ee_in_base_qx,ee_in_base_qy,ee_in_base_qz,ee_in_base_qw\n";
          out << std::fixed << std::setprecision(9);
          std::cout << "[MujocoPairDiagnostic] Writing base_link -> piper_ee CSV to "
                    << csv_file << " at " << publish_rate_hz << " Hz" << std::endl;
        } else {
          std::cerr << "[MujocoPairDiagnostic] Unable to open " << csv_file << std::endl;
        }
      }
    }

    if (base_body_id < 0 || ee_orientation_body_id < 0 ||
        ee_finger_body_id_a < 0 || ee_finger_body_id_b < 0) {
      if (!warned_missing_body) {
        warned_missing_body = true;
        std::cerr << "[MujocoPairDiagnostic] Missing one of required bodies: "
                  << kBaseBodyName << ", " << kEeOrientationBodyName << ", "
                  << kEeFingerBodyNameA << ", " << kEeFingerBodyNameB << std::endl;
      }
      return;
    }
    if (!out.is_open()) {
      return;
    }
    if (data->time >= last_publish_time_s &&
        data->time - last_publish_time_s < publish_period_s) {
      return;
    }
    last_publish_time_s = data->time;
    ++seq;

    const mjtNum* base_pos_raw = data->xpos + 3 * base_body_id;
    const mjtNum* base_quat_raw = data->xquat + 4 * base_body_id;
    const mjtNum* ee_quat_raw = data->xquat + 4 * ee_orientation_body_id;
    const mjtNum* ee_pos_a_raw = data->xpos + 3 * ee_finger_body_id_a;
    const mjtNum* ee_pos_b_raw = data->xpos + 3 * ee_finger_body_id_b;

    const std::array<mjtNum, 3> base_pos = {base_pos_raw[0], base_pos_raw[1], base_pos_raw[2]};
    std::array<mjtNum, 4> base_quat = {
        base_quat_raw[0], base_quat_raw[1], base_quat_raw[2], base_quat_raw[3]};
    const std::array<mjtNum, 3> ee_pos = {
        0.5 * (ee_pos_a_raw[0] + ee_pos_b_raw[0]),
        0.5 * (ee_pos_a_raw[1] + ee_pos_b_raw[1]),
        0.5 * (ee_pos_a_raw[2] + ee_pos_b_raw[2])};
    std::array<mjtNum, 4> ee_quat = {
        ee_quat_raw[0], ee_quat_raw[1], ee_quat_raw[2], ee_quat_raw[3]};
    NormalizeQuat(base_quat);
    NormalizeQuat(ee_quat);

    const std::array<mjtNum, 3> ee_delta_w = {
        ee_pos[0] - base_pos[0],
        ee_pos[1] - base_pos[1],
        ee_pos[2] - base_pos[2]};
    const std::array<mjtNum, 4> base_quat_inv = QuatConjugate(base_quat);
    const std::array<mjtNum, 3> ee_pos_b = QuatRotate(base_quat_inv, ee_delta_w);
    std::array<mjtNum, 4> ee_quat_b = QuatMultiply(base_quat_inv, ee_quat);
    NormalizeQuat(ee_quat_b);

    out << data->time << ','
        << 0.0 << ',' << 0.0 << ','
        << seq << ',' << seq << ','
        << base_pos[0] << ',' << base_pos[1] << ',' << base_pos[2] << ','
        << base_quat[1] << ',' << base_quat[2] << ',' << base_quat[3] << ',' << base_quat[0] << ','
        << ee_pos[0] << ',' << ee_pos[1] << ',' << ee_pos[2] << ','
        << ee_quat[1] << ',' << ee_quat[2] << ',' << ee_quat[3] << ',' << ee_quat[0] << ','
        << ee_pos_b[0] << ',' << ee_pos_b[1] << ',' << ee_pos_b[2] << ','
        << ee_quat_b[1] << ',' << ee_quat_b[2] << ',' << ee_quat_b[3] << ',' << ee_quat_b[0] << '\n';
    out.flush();
  }

  void AppendPoseJson(
      std::ostringstream& out,
      const char* name,
      const std::array<mjtNum, 3>& pos,
      const std::array<mjtNum, 4>& quat_wxyz)
  {
    out << "{\"name\":\"" << name << "\","
        << "\"p\":[" << pos[0] << ',' << pos[1] << ',' << pos[2] << "],"
        << "\"q_xyzw\":[" << quat_wxyz[1] << ',' << quat_wxyz[2] << ','
        << quat_wxyz[3] << ',' << quat_wxyz[0] << "]}";
  }

  void PublishDebugRvizFrames(const mjModel* model, const mjData* data)
  {
    constexpr const char* kBaseFrameName = "go2_base";
    constexpr const char* kEeFrameName = "piper_ee";
    constexpr const char* kBaseBodyName = "base_link";
    constexpr const char* kEeOrientationBodyName = "link6";
    constexpr const char* kEeFingerBodyNameA = "link7";
    constexpr const char* kEeFingerBodyNameB = "link8";

    static const mjModel* cached_model = nullptr;
    static int base_body_id = -1;
    static int ee_orientation_body_id = -1;
    static int ee_finger_body_id_a = -1;
    static int ee_finger_body_id_b = -1;
    static int udp_fd = -1;
    static sockaddr_in udp_addr{};
    static std::string cached_host;
    static int cached_port = -1;
    static bool warned_setup = false;
    static bool warned_missing_body = false;
    static double last_publish_time_s = -1.0;

    if (!param::config.debug_rviz) {
      return;
    }
    if (!model || !data) {
      return;
    }

    const double publish_rate_hz =
        param::config.debug_rviz_rate > 0.0 ? param::config.debug_rviz_rate : 50.0;
    const double publish_period_s = 1.0 / publish_rate_hz;
    if (data->time >= last_publish_time_s &&
        data->time - last_publish_time_s < publish_period_s) {
      return;
    }
    last_publish_time_s = data->time;

    if (udp_fd < 0 ||
        cached_host != param::config.debug_rviz_udp_host ||
        cached_port != param::config.debug_rviz_udp_port) {
      if (udp_fd >= 0) {
        close(udp_fd);
        udp_fd = -1;
      }
      cached_host = param::config.debug_rviz_udp_host;
      cached_port = param::config.debug_rviz_udp_port;
      udp_fd = socket(AF_INET, SOCK_DGRAM, 0);
      if (udp_fd < 0) {
        if (!warned_setup) {
          warned_setup = true;
          std::cerr << "[DebugRViz] Failed to create UDP socket: " << std::strerror(errno) << std::endl;
        }
        return;
      }

      std::memset(&udp_addr, 0, sizeof(udp_addr));
      udp_addr.sin_family = AF_INET;
      udp_addr.sin_port = htons(static_cast<std::uint16_t>(cached_port));
      if (inet_pton(AF_INET, cached_host.c_str(), &udp_addr.sin_addr) != 1) {
        if (!warned_setup) {
          warned_setup = true;
          std::cerr << "[DebugRViz] debug_rviz_udp_host must be an IPv4 address, got "
                    << cached_host << std::endl;
        }
        close(udp_fd);
        udp_fd = -1;
        return;
      }
      warned_setup = false;
      std::cout << "[DebugRViz] Publishing MuJoCo frames to "
                << cached_host << ':' << cached_port
                << " at " << publish_rate_hz << " Hz" << std::endl;
    }

    if (cached_model != model) {
      cached_model = model;
      base_body_id = mj_name2id(model, mjOBJ_BODY, kBaseBodyName);
      ee_orientation_body_id = mj_name2id(model, mjOBJ_BODY, kEeOrientationBodyName);
      ee_finger_body_id_a = mj_name2id(model, mjOBJ_BODY, kEeFingerBodyNameA);
      ee_finger_body_id_b = mj_name2id(model, mjOBJ_BODY, kEeFingerBodyNameB);
      warned_missing_body = false;
      last_publish_time_s = -publish_period_s;
    }

    if (base_body_id < 0 || ee_orientation_body_id < 0 ||
        ee_finger_body_id_a < 0 || ee_finger_body_id_b < 0) {
      if (!warned_missing_body) {
        warned_missing_body = true;
        std::cerr << "[DebugRViz] Missing one of required bodies: "
                  << kBaseBodyName << ", " << kEeOrientationBodyName << ", "
                  << kEeFingerBodyNameA << ", " << kEeFingerBodyNameB << std::endl;
      }
      return;
    }

    const mjtNum* base_pos_raw = data->xpos + 3 * base_body_id;
    const mjtNum* base_quat_raw = data->xquat + 4 * base_body_id;
    const mjtNum* ee_quat_raw = data->xquat + 4 * ee_orientation_body_id;
    const mjtNum* ee_pos_a_raw = data->xpos + 3 * ee_finger_body_id_a;
    const mjtNum* ee_pos_b_raw = data->xpos + 3 * ee_finger_body_id_b;

    const std::array<mjtNum, 3> world_pos = {0.0, 0.0, 0.0};
    const std::array<mjtNum, 4> world_quat = {1.0, 0.0, 0.0, 0.0};
    const std::array<mjtNum, 3> base_pos = {base_pos_raw[0], base_pos_raw[1], base_pos_raw[2]};
    std::array<mjtNum, 4> base_quat = {
        base_quat_raw[0], base_quat_raw[1], base_quat_raw[2], base_quat_raw[3]};
    const std::array<mjtNum, 3> ee_pos = {
        0.5 * (ee_pos_a_raw[0] + ee_pos_b_raw[0]),
        0.5 * (ee_pos_a_raw[1] + ee_pos_b_raw[1]),
        0.5 * (ee_pos_a_raw[2] + ee_pos_b_raw[2])};
    std::array<mjtNum, 4> ee_quat = {
        ee_quat_raw[0], ee_quat_raw[1], ee_quat_raw[2], ee_quat_raw[3]};
    NormalizeQuat(base_quat);
    NormalizeQuat(ee_quat);

    std::ostringstream out;
    out << std::fixed << std::setprecision(9)
        << "{\"sim_time_s\":" << data->time << ",\"frames\":[";
    AppendPoseJson(out, "world", world_pos, world_quat);
    out << ',';
    AppendPoseJson(out, kBaseFrameName, base_pos, base_quat);
    out << ',';
    AppendPoseJson(out, kEeFrameName, ee_pos, ee_quat);
    out << "]}\n";

    const std::string payload = out.str();
    const ssize_t sent = sendto(
        udp_fd,
        payload.data(),
        payload.size(),
        0,
        reinterpret_cast<const sockaddr*>(&udp_addr),
        sizeof(udp_addr));
    if (sent < 0) {
      std::cerr << "[DebugRViz] UDP send failed: " << std::strerror(errno) << std::endl;
    }
  }

  // simulate in background thread (while rendering in main thread)
  void PhysicsLoop(mj::Simulate &sim)
  {
    // cpu-sim syncronization point
    std::chrono::time_point<mj::Simulate::Clock> syncCPU;
    mjtNum syncSim = 0;

    // ChannelFactory::Instance()->Init(0);
    // UnitreeDds ud(d);

    // run until asked to exit
    while (!sim.exitrequest.load())
    {
      if (sim.droploadrequest.load())
      {
        sim.LoadMessage(sim.dropfilename);
        mjModel *mnew = LoadModel(sim.dropfilename, sim);
        sim.droploadrequest.store(false);

        mjData *dnew = nullptr;
        if (mnew)
          dnew = mj_makeData(mnew);
        if (dnew)
        {
          sim.Load(mnew, dnew, sim.dropfilename);

          mj_deleteData(d);
          mj_deleteModel(m);

          m = mnew;
          d = dnew;
          mj_forward(m, d);

          // allocate ctrlnoise
          free(ctrlnoise);
          ctrlnoise = (mjtNum *)malloc(sizeof(mjtNum) * m->nu);
          mju_zero(ctrlnoise, m->nu);
        }
        else
        {
          sim.LoadMessageClear();
        }
      }

      if (sim.uiloadrequest.load())
      {
        sim.uiloadrequest.fetch_sub(1);
        sim.LoadMessage(sim.filename);
        mjModel *mnew = LoadModel(sim.filename, sim);
        mjData *dnew = nullptr;
        if (mnew)
          dnew = mj_makeData(mnew);
        if (dnew)
        {
          sim.Load(mnew, dnew, sim.filename);

          mj_deleteData(d);
          mj_deleteModel(m);

          m = mnew;
          d = dnew;
          mj_forward(m, d);

          // allocate ctrlnoise
          free(ctrlnoise);
          ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
          mju_zero(ctrlnoise, m->nu);
        }
        else
        {
          sim.LoadMessageClear();
        }
      }

      // sleep for 1 ms or yield, to let main thread run
      //  yield results in busy wait - which has better timing but kills battery life
      if (sim.run && sim.busywait)
      {
        std::this_thread::yield();
      }
      else
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      {
        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        // run only if model is present
        if (m)
        {
          // running
          if (sim.run)
          {
            bool stepped = false;

            // record cpu time at start of iteration
            const auto startCPU = mj::Simulate::Clock::now();

            // elapsed CPU and simulation time since last sync
            const auto elapsedCPU = startCPU - syncCPU;
            double elapsedSim = d->time - syncSim;

            // inject noise
            if (sim.ctrl_noise_std)
            {
              // convert rate and scale to discrete time (Ornstein–Uhlenbeck)
              mjtNum rate = mju_exp(-m->opt.timestep / mju_max(sim.ctrl_noise_rate, mjMINVAL));
              mjtNum scale = sim.ctrl_noise_std * mju_sqrt(1 - rate * rate);

              for (int i = 0; i < m->nu; i++)
              {
                // update noise
                ctrlnoise[i] = rate * ctrlnoise[i] + scale * mju_standardNormal(nullptr);

                // apply noise
                d->ctrl[i] = ctrlnoise[i];
              }
            }

            // requested slow-down factor
            double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

            // misalignment condition: distance from target sim time is bigger than syncmisalign
            bool misaligned =
                mju_abs(Seconds(elapsedCPU).count() / slowdown - elapsedSim) > syncMisalign;

            // out-of-sync (for any reason): reset sync times, step
            if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 ||
                misaligned || sim.speed_changed)
            {
              // re-sync
              syncCPU = startCPU;
              syncSim = d->time;
              sim.speed_changed = false;

              // run single step, let next iteration deal with timing
              UpdateMovingPickPlatform(m, d);
              mj_step(m, d);
              UpdateBlackboardEraserGraspLatch(m, d);
              stepped = true;
            }

            // in-sync: step until ahead of cpu
            else
            {
              bool measured = false;
              mjtNum prevSim = d->time;

              double refreshTime = simRefreshFraction / sim.refresh_rate;

              // step while sim lags behind cpu and within refreshTime
              while (Seconds((d->time - syncSim) * slowdown) < mj::Simulate::Clock::now() - syncCPU &&
                     mj::Simulate::Clock::now() - startCPU < Seconds(refreshTime))
              {
                // measure slowdown before first step
                if (!measured && elapsedSim)
                {
                  sim.measured_slowdown =
                      std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
                  measured = true;
                }

                // elastic band on base link
                if (param::config.enable_elastic_band == 1)
                {
                  if (elastic_band.enable_)
                  {
                    std::vector<double> x = {d->qpos[0], d->qpos[1], d->qpos[2]};
                    std::vector<double> dx = {d->qvel[0], d->qvel[1], d->qvel[2]};

                    elastic_band.Advance(x, dx);

                    d->xfrc_applied[param::config.band_attached_link] = elastic_band.f_[0];
                    d->xfrc_applied[param::config.band_attached_link + 1] = elastic_band.f_[1];
                    d->xfrc_applied[param::config.band_attached_link + 2] = elastic_band.f_[2];
                  }
                }

                // call mj_step
                UpdateMovingPickPlatform(m, d);
                mj_step(m, d);
                UpdateBlackboardEraserGraspLatch(m, d);
                stepped = true;

                // break if reset
                if (d->time < prevSim)
                {
                  break;
                }
              }
            }

            // save current state to history buffer
            if (stepped)
            {
              sim.AddToHistory();
              PublishTaskObjectPose(m, d);
              PublishMujocoPairDiagnostic(m, d);
              PublishDebugRvizFrames(m, d);
            }
          }

          // paused
          else
          {
            // run mj_forward, to update rendering and joint sliders
            mj_forward(m, d);
            PublishTaskObjectPose(m, d);
            PublishMujocoPairDiagnostic(m, d);
            PublishDebugRvizFrames(m, d);
            sim.speed_changed = true;
          }
        }
      } // release std::lock_guard<std::mutex>
    }
	  }
	} // namespace

extern "C" void UnitreeMujocoDebugAppendGeoms(const mjModel* model, const mjData* data, mjvScene* scene)
{
  if (!model || !data || !scene) {
    return;
  }

  // Scene-local clean-video mode.  Hiding the rendered mjvGeoms instead of
  // deleting fake_ee from MJCF keeps its inertial contribution and the body
  // pose used by the controller/OptiTrack diagnostics unchanged.
  const int hide_debug_numeric_id =
      mj_name2id(model, mjOBJ_NUMERIC, "recording_hide_debug_visuals");
  const bool hide_debug_visuals =
      hide_debug_numeric_id >= 0 &&
      model->numeric_size[hide_debug_numeric_id] >= 1 &&
      model->numeric_data[model->numeric_adr[hide_debug_numeric_id]] > 0.5;
  if (hide_debug_visuals) {
    const int fake_ee_body_id = mj_name2id(model, mjOBJ_BODY, "fake_ee");
    if (fake_ee_body_id >= 0) {
      for (int i = 0; i < scene->ngeom; ++i) {
        mjvGeom& rendered_geom = scene->geoms[i];
        const int object_id = rendered_geom.objid;
        const bool belongs_to_fake_ee =
            (rendered_geom.objtype == mjOBJ_GEOM &&
             object_id >= 0 && object_id < model->ngeom &&
             model->geom_bodyid[object_id] == fake_ee_body_id) ||
            (rendered_geom.objtype == mjOBJ_SITE &&
             object_id >= 0 && object_id < model->nsite &&
             model->site_bodyid[object_id] == fake_ee_body_id);
        if (belongs_to_fake_ee) {
          rendered_geom.rgba[3] = 0.0f;
        }
      }
    }
    return;
  }

  HighLevelDebugPose pose;
  {
    std::lock_guard<std::mutex> lock(debug_pose_mutex);
    pose = debug_pose;
  }

  if (pose.target_valid) {
    const float target_rgba[4] = {1.0f, 0.78f, 0.0f, 0.9f};
    AddSphere(scene, pose.target_pos, 0.035, target_rgba);
    AddFrame(
        scene,
        pose.target_pos,
        pose.target_quat,
        kDebugFrameAxisLength,
        kDebugFrameAxisWidth,
        true);
  }

  if (pose.sub_target_valid) {
    const float sub_target_rgba[4] = {1.0f, 0.0f, 0.9f, 0.9f};
    AddSphere(scene, pose.sub_target_pos, 0.028, sub_target_rgba);
    AddFrame(
        scene,
        pose.sub_target_pos,
        pose.sub_target_quat,
        0.12,
        0.007,
        true);
  }

  std::array<mjtNum, 3> gripper_pos = pose.ee_pos;
  std::array<mjtNum, 4> gripper_quat = pose.ee_quat;
  bool gripper_valid = pose.ee_valid;

  static const mjModel* cached_model = nullptr;
  static int link7_body_id = -1;
  static int link8_body_id = -1;
  static int link6_body_id = -1;
  if (cached_model != model) {
    cached_model = model;
    link7_body_id = mj_name2id(model, mjOBJ_BODY, "link7");
    link8_body_id = mj_name2id(model, mjOBJ_BODY, "link8");
    link6_body_id = mj_name2id(model, mjOBJ_BODY, "link6");
  }

  if (link7_body_id >= 0 && link8_body_id >= 0) {
    const mjtNum* p7 = data->xpos + 3 * link7_body_id;
    const mjtNum* p8 = data->xpos + 3 * link8_body_id;
    for (int i = 0; i < 3; ++i) {
      gripper_pos[i] = 0.5 * (p7[i] + p8[i]);
    }
    gripper_valid = true;
  }
  if (link6_body_id >= 0) {
    const mjtNum* q = data->xquat + 4 * link6_body_id;
    for (int i = 0; i < 4; ++i) {
      gripper_quat[i] = q[i];
    }
  }

  if (gripper_valid) {
    const float gripper_rgba[4] = {0.0f, 0.9f, 1.0f, 0.9f};
    AddSphere(scene, gripper_pos, 0.025, gripper_rgba);
    AddFrame(
        scene,
        gripper_pos,
        gripper_quat,
        0.11,
        0.006,
        false);
  }
}

//-------------------------------------- physics_thread --------------------------------------------

void PhysicsThread(mj::Simulate *sim, const char *filename)
{
  // request loadmodel if file given (otherwise drag-and-drop)
  if (filename != nullptr)
  {
    sim->LoadMessage(filename);
    m = LoadModel(filename, *sim);
    if (m)
      d = mj_makeData(m);
    if (d)
    {
      sim->Load(m, d, filename);
      mj_forward(m, d);

      // allocate ctrlnoise
      free(ctrlnoise);
      ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
      mju_zero(ctrlnoise, m->nu);
    }
    else
    {
      sim->LoadMessageClear();
    }
  }

  PhysicsLoop(*sim);

  // delete everything we allocated
  free(ctrlnoise);
  mj_deleteData(d);
  mj_deleteModel(m);

  exit(0);
}

void *UnitreeSdk2BridgeThread(void *arg)
{
  // Wait for mujoco data
  while (true)
  {
    if (d)
    {
      std::cout << "Mujoco data is prepared" << std::endl;
      break;
    }
    usleep(500000);
  }

  unitree::robot::ChannelFactory::Instance()->Init(param::config.domain_id, param::config.interface);


  int body_id = mj_name2id(m, mjOBJ_BODY, "torso_link");
  if (body_id < 0) {
    body_id = mj_name2id(m, mjOBJ_BODY, "base_link");
  }
  param::config.band_attached_link = 6 * body_id;
  
  std::unique_ptr<UnitreeSDK2BridgeBase> interface = nullptr;
  if (m->nu > NUM_MOTOR_IDL_GO) {
    interface = std::make_unique<G1Bridge>(m, d);
  } else {
    interface = std::make_unique<Go2Bridge>(m, d);
  }
  interface->start();
  
  while (true)
  {
    sleep(1);
  }
}
//------------------------------------------ main --------------------------------------------------

// machinery for replacing command line error by a macOS dialog box when running under Rosetta
#if defined(__APPLE__) && defined(__AVX__)
extern void DisplayErrorDialogBox(const char *title, const char *msg);
static const char *rosetta_error_msg = nullptr;
__attribute__((used, visibility("default"))) extern "C" void _mj_rosettaError(const char *msg)
{
  rosetta_error_msg = msg;
}
#endif

// user keyboard callback
void user_key_cb(GLFWwindow* window, int key, int scancode, int act, int mods) {
  if (act==GLFW_PRESS)
  {
    if(param::config.enable_elastic_band == 1) {
      if (key==GLFW_KEY_9) {
        elastic_band.enable_ = !elastic_band.enable_;
      } else if (key==GLFW_KEY_7 || key==GLFW_KEY_UP) {
        elastic_band.length_ -= 0.1;
      } else if (key==GLFW_KEY_8 || key==GLFW_KEY_DOWN) {
        elastic_band.length_ += 0.1;
      }
    }
    if(key==GLFW_KEY_BACKSPACE) {
      mj_resetData(m, d);
      mj_forward(m, d);
    }
  }
}

// run event loop
int main(int argc, char **argv)
{

  // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
  if (rosetta_error_msg)
  {
    DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
    std::exit(1);
  }
#endif

  // print version, check compatibility
  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER != mj_version())
  {
    mju_error("Headers and library have different versions");
  }

  // scan for libraries in the plugin directory to load additional plugins
  scanPluginLibraries();

  mjvCamera cam;
  mjv_defaultCamera(&cam);

  mjvOption opt;
  mjv_defaultOption(&opt);

  mjvPerturb pert;
  mjv_defaultPerturb(&pert);

  // Load simulation configuration
  std::filesystem::path proj_dir = std::filesystem::path(getExecutableDir()).parent_path();
  param::config.load_from_yaml(proj_dir / "config.yaml");
  param::helper(argc, argv);
  if(param::config.robot_scene.is_relative()) {
    param::config.robot_scene = proj_dir.parent_path() / "unitree_robots" / param::config.robot / param::config.robot_scene;
  }

  // simulate object encapsulates the UI
  auto sim = std::make_unique<mj::Simulate>(
    std::make_unique<mj::GlfwAdapter>(),
    &cam, &opt, &pert, /* is_passive = */ false);

  std::thread unitree_thread(UnitreeSdk2BridgeThread, nullptr);
  std::thread debug_vis_thread(HighLevelDebugUdpThread);
  debug_vis_thread.detach();

  // start physics thread
  std::thread physicsthreadhandle(&PhysicsThread, sim.get(), param::config.robot_scene.c_str());
  // start simulation UI loop (blocking call)
  glfwSetKeyCallback(static_cast<mj::GlfwAdapter*>(sim->platform_ui.get())->window_,user_key_cb);
  sim->RenderLoop();
  physicsthreadhandle.join();

  pthread_exit(NULL);
  return 0;
}
