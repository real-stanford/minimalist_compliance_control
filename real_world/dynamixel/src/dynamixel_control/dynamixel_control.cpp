#include "dynamixel_control.h"
#include "dynamixel_client.h"
#include <stdexcept>
#include <thread>
#include <chrono>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <algorithm>

std::unordered_map<std::string, int> CONTROL_MODE_DICT = {
    {"current", 0},
    {"velocity", 1},
    {"position", 3},
    {"extended_position", 4},
    {"current_based_position", 5},
    {"pwm", 16}};

// Constructor
DynamixelControl::DynamixelControl(const std::string &port,
                                   const std::vector<int> &motor_ids,
                                   const std::vector<float> &kp,
                                   const std::vector<float> &kd,
                                   const std::vector<float> &ki,
                                   const std::vector<float> &zero_pos,
                                   const std::vector<std::string> &control_mode,
                                   int baudrate,
                                   int return_delay_time)
    : port_(port),
      motor_ids_(motor_ids),
      kp_(kp), kd_(kd), ki_(ki),
      zero_pos_(zero_pos),
      control_mode_(control_mode),
      baudrate_(baudrate),
      return_delay_time_(return_delay_time)
{
  if (ki_.empty())
  {
    ki_.assign(motor_ids_.size(), 0.0f);
  }
  kff1_.assign(motor_ids_.size(), 0.0f);
  kff2_.assign(motor_ids_.size(), 0.0f);
  client_ = std::make_unique<DynamixelClient>(motor_ids_, port_, baudrate_, /*lazy=*/true);
}

// Destructor
DynamixelControl::~DynamixelControl()
{
  close_motors();
}

// Initialize motors: reboot, voltage check, configure registers, enable torque
void DynamixelControl::initialize_motors()
{
  // 0) Ensure client is connected
  if (client_ && !client_->is_connected())
  {
    client_->connect();
  }

  // 1) Reboot
  // try
  // {
  //   client_->reboot(motor_ids_);
  //   std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  // }
  // catch (const std::exception &e)
  // {
  //   std::cerr << "[Initialize] reboot failed: " << e.what() << std::endl;
  // }

  // 1) Clear multi-turn and error
  try
  {
    client_->clear_multi_turn(motor_ids_);
    client_->clear_error(motor_ids_);
    std::cout << "[Initialize] Multi-turn and error cleared on port " << port_ << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  catch (const std::exception &e)
  {
    std::cerr << "[Initialize] clear multi-turn/error failed: " << e.what() << std::endl;
  }

  // 2) Voltage check
  try
  {
    auto [t_vin, vin] = client_->read_vin(-1);
    std::cout << "[Initialize] Voltage: [";
    for (size_t i = 0; i < vin.size(); ++i)
    {
      if (vin[i] < 10.0f)
      {
        std::cerr << "[Initialize] voltage too low!" << std::endl;
        break;
      }
      std::cout << vin[i];
      if (i != vin.size() - 1)
        std::cout << ", ";
    }
    std::cout << "] V on port " << port_ << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  catch (const std::exception &e)
  {
    std::cerr << "[Initialize] read_vin failed: " << e.what() << std::endl;
  }

  // 3) Return‐delay & gains
  auto do_sync = [&](const std::vector<int> &data, uint16_t addr, uint16_t len, const char *ctx)
  {
    try
    {
      client_->sync_write(motor_ids_, data, addr, len);
    }
    catch (const std::exception &e)
    {
      std::cerr << "[Initialize] sync_write(" << ctx << ") failed: " << e.what() << std::endl;
    }
  };
  // Return delay time
  std::vector<int> return_delay_vec(motor_ids_.size(), return_delay_time_);
  do_sync(return_delay_vec, 9, 1, "return_delay");
  // Control mode
  std::vector<int> control_mode_vec;
  control_mode_vec.reserve(motor_ids_.size());
  for (auto m : control_mode_)
    control_mode_vec.push_back(CONTROL_MODE_DICT.at(m));
  do_sync(control_mode_vec, 11, 1, "control_mode");
  // Gains
  std::vector<int> kd_vec(kd_.begin(), kd_.end());
  std::vector<int> ki_vec(ki_.begin(), ki_.end());
  std::vector<int> kp_vec(kp_.begin(), kp_.end());
  std::vector<int> kff2_vec(kff2_.begin(), kff2_.end());
  std::vector<int> kff1_vec(kff1_.begin(), kff1_.end());
  do_sync(kd_vec, 80, 2, "kd");
  do_sync(ki_vec, 82, 2, "ki");
  do_sync(kp_vec, 84, 2, "kp");
  do_sync(kff2_vec, 88, 2, "kff2");
  do_sync(kff1_vec, 90, 2, "kff1");

  // 4) Finally, enable torque
  try
  {
    client_->set_torque_enabled(motor_ids_, true);
    std::cout << "[Initialize] Torque enabled on port " << port_ << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  catch (const std::exception &e)
  {
    std::cerr << "[Initialize] enable torque failed: " << e.what() << std::endl;
  }

  // 5) Update zero positions based on current readings
  try
  {
    auto [t, pos] = client_->read_pos(-1);
    for (size_t i = 0; i < pos.size(); ++i)
    {
      float delta = pos[i] - zero_pos_[i];
      delta = std::fmod(delta + static_cast<float>(M_PI), static_cast<float>(2 * M_PI)) - static_cast<float>(M_PI);
      zero_pos_[i] = pos[i] - delta;
    }
    std::cout << "[Initialize] zero positions updated on port " << port_ << std::endl;
  }
  catch (const std::exception &e)
  {
    std::cerr << "[Initialize] update zero point failed: " << e.what() << std::endl;
  }
}

// Force-close all open clients
void DynamixelControl::close_motors()
{
  const auto &clients = DynamixelClient::get_open_clients();
  std::vector<DynamixelClient *> to_close(clients.begin(), clients.end());
  for (auto *client : to_close)
  {
    if (client == nullptr)
      continue;
    try
    {
      client->disconnect();
    }
    catch (const std::exception &e)
    {
      std::cerr << "[DynamixelControl] Failed to disconnect client: " << e.what() << std::endl;
    }
  }
}

// Disable torque on specified or all motors
void DynamixelControl::disable_motors(const std::vector<int> &ids)
{
  const auto &targets = ids.empty() ? motor_ids_ : ids;
  client_->set_torque_enabled(targets, false, 0);
}

// Enable torque on specified or all motors
void DynamixelControl::enable_motors(const std::vector<int> &ids)
{
  const auto &targets = ids.empty() ? motor_ids_ : ids;
  client_->set_torque_enabled(targets, true);
}

void DynamixelControl::set_pd(const std::vector<float> &kp, const std::vector<float> &kd)
{
  if (kp.size() != motor_ids_.size() || kd.size() != motor_ids_.size())
  {
    throw std::invalid_argument("kp/kd size must match number of motors");
  }

  if (client_ && !client_->is_connected())
  {
    client_->connect();
  }

  kp_ = kp;
  kd_ = kd;

  auto clamp_to_ushort = [](float value) -> int
  {
    int v = static_cast<int>(std::round(value));
    v = std::clamp(v, 0, 0xFFFF);
    return v;
  };

  std::vector<int> kp_raw(kp.size());
  std::vector<int> kd_raw(kd.size());
  for (size_t i = 0; i < kp.size(); ++i)
  {
    kp_raw[i] = clamp_to_ushort(kp[i]);
    kd_raw[i] = clamp_to_ushort(kd[i]);
  }

  try
  {
    client_->sync_write(motor_ids_, kd_raw, 80, 2);
    client_->sync_write(motor_ids_, kp_raw, 84, 2);
  }
  catch (const std::exception &e)
  {
    throw std::runtime_error(std::string("Failed to set PD gains: ") + e.what());
  }
}

// Write a new position command (in radians)
void DynamixelControl::set_pos(const std::vector<float> &pos)
{
  std::vector<float> drive;
  drive.reserve(pos.size());
  for (size_t i = 0; i < pos.size(); ++i)
  {
    drive.push_back(zero_pos_[i] + pos[i]);
  }
  client_->write_desired_pos(motor_ids_, drive);
}

// Write a new velocity command (in rad/s)
void DynamixelControl::set_vel(const std::vector<float> &vel)
{
  client_->write_desired_vel(motor_ids_, vel);
}

// Write a new current command
void DynamixelControl::set_current(const std::vector<float> &cur)
{
  client_->write_desired_cur(motor_ids_, cur);
}

void DynamixelControl::set_pwm(const std::vector<float> &pwm)
{
  client_->write_desired_pwm(motor_ids_, pwm);
}

void DynamixelControl::set_control_mode(const std::vector<std::string> &control_mode)
{
  if (control_mode.size() != motor_ids_.size())
  {
    throw std::invalid_argument("control_mode size must match number of motors");
  }

  if (client_ && !client_->is_connected())
  {
    client_->connect();
  }

  control_mode_ = control_mode;
  std::vector<int> control_mode_vec;
  control_mode_vec.reserve(motor_ids_.size());
  for (const auto &mode : control_mode_)
  {
    control_mode_vec.push_back(CONTROL_MODE_DICT.at(mode));
  }
  client_->sync_write(motor_ids_, control_mode_vec, 11, 1);
}

// Read back (time, pos, vel, cur, pwm, vin, temp) adjusted by zero_pos_
std::map<std::string, std::vector<float>> DynamixelControl::get_state(int retries)
{
  auto [t, pwm, cur, vel, pos, vin, temp] = client_->read_all(retries);
  for (size_t i = 0; i < pos.size(); ++i)
  {
    pos[i] -= zero_pos_[i];
  }
  return {
      {"pos", pos},
      {"vel", vel},
      {"cur", cur},
      {"pwm", pwm},
      {"vin", vin},
      {"temp", temp}};
}

std::vector<float> DynamixelControl::get_current_limit(int retries)
{
  auto res = client_->read_cur_limit(retries);
  return res.second;
}
