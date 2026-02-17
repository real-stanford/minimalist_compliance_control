#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "dynamixel_control.h"
#include "dynamixel_sdk/port_handler.h"
#include "dynamixel_sdk/packet_handler.h"
#include <future>
#include <thread>
#include <vector>
#include <tuple>
#include <iostream>
#include <filesystem>
#include <regex>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <cstdlib>  // for std::system
#include <cstdio>   // for popen, fgets, pclose
#include <unistd.h> // for uname
#include <sys/utsname.h>
#include <cerrno>
#include <cstring>

namespace py = pybind11;
namespace fs = std::filesystem;

// You can pass this in if you don't want global state
std::vector<DynamixelControl *> controllers;

std::vector<int> scan_port(const std::string &port_name,
                           int baudrate,
                           int protocol_version = 2.0,
                           int max_motor_id = 32,
                           int retries = 3)
{
  using namespace dynamixel;

  std::vector<int> found_ids;

  PortHandler *portHandler = PortHandler::getPortHandler(port_name.c_str());
  PacketHandler *packetHandler = PacketHandler::getPacketHandler(protocol_version);

  if (!portHandler->openPort())
  {
    std::cerr << "[scan_port] Failed to open " << port_name;
    if (errno != 0)
    {
      std::cerr << " (" << std::strerror(errno) << ")";
    }
    std::cerr << std::endl;
    return found_ids;
  }

  if (!portHandler->setBaudRate(baudrate))
  {
    std::cerr << "[scan_port] Failed to set baudrate on " << port_name << std::endl;
    portHandler->closePort();
    return found_ids;
  }

  for (int id = 0; id < max_motor_id; ++id)
  {
    bool found = false;
    for (int attempt = 0; attempt < retries && !found; ++attempt)
    {
      uint16_t model_number = 0;
      uint8_t error = 0;
      int result = packetHandler->ping(portHandler, id, &model_number, &error);
      if (result == COMM_SUCCESS && error == 0)
      {
        found_ids.push_back(id);
        found = true;
      }
    }
  }

  portHandler->closePort();
  return found_ids;
}

void set_latency_timer(const std::string &port, int latency_value)
{
  std::string os_type;
  struct utsname buffer;
  if (uname(&buffer) == 0)
  {
    os_type = buffer.sysname;
  }
  else
  {
    throw std::runtime_error("Failed to detect OS type");
  }

  std::string command;

  if (os_type == "Linux")
  {
    std::string port_name = port.substr(port.find_last_of('/') + 1);
    command = "echo " + std::to_string(latency_value) +
              " | sudo tee /sys/bus/usb-serial/devices/" + port_name + "/latency_timer";
  }
  else if (os_type == "Darwin")
  {
    command = "./real_world/dynamixel/latency_timer_setter_macOS/set_latency_timer -l " +
              std::to_string(latency_value);
  }
  else
  {
    throw std::runtime_error("Unsupported OS: " + os_type);
  }

  // Use popen to capture output
  FILE *pipe = popen(command.c_str(), "r");
  if (!pipe)
  {
    throw std::runtime_error("Failed to run latency timer command");
  }

  char buffer_output[128];
  std::string result;
  while (fgets(buffer_output, sizeof(buffer_output), pipe) != nullptr)
  {
    result += buffer_output;
  }

  int return_code = pclose(pipe);
  if (return_code != 0)
  {
    throw std::runtime_error("Latency timer command failed with code " + std::to_string(return_code));
  }

  if (!result.empty() && result.back() == '\n')
  {
    result.pop_back();
  }
  std::cout << "Latency Timer set: " << result << std::endl;
}

std::vector<std::shared_ptr<DynamixelControl>> create_controllers(
    const std::string &port_pattern,
    const std::vector<float> &kp,
    const std::vector<float> &kd,
    const std::vector<float> &ki,
    const std::vector<float> &zero_pos,
    const std::vector<std::string> &control_mode,
    int baudrate,
    int return_delay)
{
  if (!(kp.size() == kd.size() && kp.size() == ki.size() &&
        kp.size() == zero_pos.size() && kp.size() == control_mode.size()))
  {
    throw std::invalid_argument("kp, kd, ki, zero_pos, and control_mode must have the same length");
  }

  std::vector<std::shared_ptr<DynamixelControl>> controllers;
  std::map<std::string, std::vector<int>> port_to_ids;

  std::regex pattern(port_pattern);
  for (const auto &entry : fs::directory_iterator("/dev"))
  {
    std::string filename = entry.path().filename().string();
    if (!std::regex_match(filename, pattern))
      continue;

    std::string full_path = entry.path();
    try
    {
      std::string filename_lower = filename;
      std::transform(filename_lower.begin(), filename_lower.end(), filename_lower.begin(),
                     [](unsigned char c)
                     { return static_cast<char>(std::tolower(c)); });
      bool is_usb_serial =
          filename_lower.find("ttyusb") != std::string::npos ||
          filename_lower.find("ttyacm") != std::string::npos ||
          filename_lower.find("usbserial") != std::string::npos;
      if (is_usb_serial)
      {
        try
        {
          set_latency_timer(full_path, 1);
        }
        catch (const std::exception &e)
        {
          std::cerr << "[create_controllers] Unable to set latency timer on "
                    << full_path << ": " << e.what() << std::endl;
        }
      }

      auto ids = scan_port(full_path, baudrate);
      if (!ids.empty())
        port_to_ids[full_path] = ids;
    }
    catch (const std::exception &e)
    {
      std::cerr << "[create_controllers] Error scanning " << full_path << ": " << e.what() << std::endl;
    }
  }

  // Collect and sort all detected motor IDs
  std::vector<int> all_ids;
  for (const auto &pair : port_to_ids)
    all_ids.insert(all_ids.end(), pair.second.begin(), pair.second.end());

  std::sort(all_ids.begin(), all_ids.end());

  // Build a map from motor ID to index in global kp/kd arrays
  std::unordered_map<int, int> id_to_index;
  for (size_t i = 0; i < all_ids.size(); ++i)
    id_to_index[all_ids[i]] = i;

  // For each port, extract the corresponding slice of controller params
  for (const auto &[port, ids] : port_to_ids)
  {
    std::vector<float> kp_local, kd_local, ki_local, zero_local;
    std::vector<std::string> mode_local;

    for (int id : ids)
    {
      int idx = id_to_index.at(id);
      kp_local.push_back(kp[idx]);
      kd_local.push_back(kd[idx]);
      ki_local.push_back(ki[idx]);
      zero_local.push_back(zero_pos[idx]);
      mode_local.push_back(control_mode[idx]);
    }

    std::cout << "Detected motors on " << port << ": ";
    for (int id : ids)
      std::cout << id << " ";
    std::cout << std::endl;

    controllers.emplace_back(std::make_shared<DynamixelControl>(
        port, ids, kp_local, kd_local, ki_local, zero_local, mode_local, baudrate, return_delay));
  }
  return controllers;
}

void initialize_motors(
    const std::vector<std::shared_ptr<DynamixelControl>> &ctrls)
{
  size_t N = ctrls.size();
  std::vector<std::thread> threads;
  threads.reserve(N);

  // Initialize each controller in parallel with staggered start times
  for (size_t i = 0; i < ctrls.size(); ++i)
  {
    threads.emplace_back([ctrl = ctrls[i], delay_ms = i * 50]()
                         { 
                           std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
                           ctrl->initialize_motors(); });
  }

  for (auto &t : threads)
    t.join();
}

std::map<std::string, std::vector<int>>
get_motor_ids(const std::vector<DynamixelControl *> &controllers)
{
  std::map<std::string, std::vector<int>> motor_ids;
  for (size_t i = 0; i < controllers.size(); ++i)
  {
    motor_ids["controller_" + std::to_string(i)] = controllers[i]->get_motor_ids();
  }
  return motor_ids;
}

std::map<std::string, std::map<std::string, std::vector<float>>>
get_motor_states(const std::vector<DynamixelControl *> &controllers, int retries = 0)
{
  size_t N = controllers.size();
  std::vector<std::future<std::map<std::string, std::vector<float>>>> futures;
  futures.reserve(N);

  // Launch threads
  for (auto *ctrl : controllers)
  {
    futures.emplace_back(std::async(std::launch::async, [ctrl, retries]()
                                    { return ctrl->get_state(retries); }));
  }

  std::map<std::string, std::map<std::string, std::vector<float>>> states;

  for (size_t i = 0; i < N; ++i)
  {
    std::string key = "controller_" + std::to_string(i);
    try
    {
      states[key] = futures[i].get();
    }
    catch (const std::exception &e)
    {
      std::cerr << "[get_motor_states] Exception for " << key << " ids: ";
      for (auto id : controllers[i]->get_motor_ids())
        std::cerr << id << " ";
      std::cerr << "-- " << e.what() << std::endl;

      // Insert an empty map to preserve key
      states[key] = std::map<std::string, std::vector<float>>{};
    }
  }
  return states;
}

std::map<std::string, std::vector<float>>
get_motor_current_limits(const std::vector<DynamixelControl *> &controllers, int retries = 0)
{
  size_t N = controllers.size();
  std::vector<std::future<std::vector<float>>> futures;
  futures.reserve(N);

  for (auto *ctrl : controllers)
  {
    futures.emplace_back(std::async(std::launch::async, [ctrl, retries]()
                                    { return ctrl->get_current_limit(retries); }));
  }

  std::map<std::string, std::vector<float>> limits;
  for (size_t i = 0; i < N; ++i)
  {
    std::string key = "controller_" + std::to_string(i);
    try
    {
      limits[key] = futures[i].get();
    }
    catch (const std::exception &e)
    {
      std::cerr << "[get_motor_current_limits] Exception for " << key << ": "
                << e.what() << std::endl;
      limits[key] = std::vector<float>{};
    }
  }
  return limits;
}

void set_motor_pos(const std::vector<DynamixelControl *> &controllers,
                   const std::vector<std::vector<float>> &pos_vecs)
{
  size_t N = controllers.size();
  std::vector<std::future<void>> futures;
  futures.reserve(N);

  for (size_t i = 0; i < N && i < pos_vecs.size(); ++i)
  {
    futures.emplace_back(std::async(std::launch::async,
                                    [ctrl = controllers[i], pos = pos_vecs[i]]()
                                    {
                                      ctrl->set_pos(pos);
                                    }));
  }

  for (auto &f : futures)
    f.get(); // Wait and propagate any exceptions
}

void set_motor_vel(const std::vector<DynamixelControl *> &controllers,
                   const std::vector<std::vector<float>> &vel_vecs)
{
  size_t N = controllers.size();
  std::vector<std::future<void>> futures;
  futures.reserve(N);

  for (size_t i = 0; i < N && i < vel_vecs.size(); ++i)
  {
    futures.emplace_back(std::async(std::launch::async,
                                    [ctrl = controllers[i], vel = vel_vecs[i]]()
                                    {
                                      ctrl->set_vel(vel);
                                    }));
  }

  for (auto &f : futures)
    f.get();
}

void set_motor_cur(const std::vector<DynamixelControl *> &controllers,
                   const std::vector<std::vector<float>> &cur_vecs)
{
  size_t N = controllers.size();
  std::vector<std::future<void>> futures;
  futures.reserve(N);

  for (size_t i = 0; i < N && i < cur_vecs.size(); ++i)
  {
    futures.emplace_back(std::async(std::launch::async,
                                    [ctrl = controllers[i], cur = cur_vecs[i]]()
                                    {
                                      ctrl->set_current(cur);
                                    }));
  }

  for (auto &f : futures)
    f.get();
}

void set_motor_pwm(const std::vector<DynamixelControl *> &controllers,
                   const std::vector<std::vector<float>> &pwm_vecs)
{
  size_t N = controllers.size();
  std::vector<std::future<void>> futures;
  futures.reserve(N);

  for (size_t i = 0; i < N && i < pwm_vecs.size(); ++i)
  {
    futures.emplace_back(std::async(std::launch::async,
                                    [ctrl = controllers[i], pwm = pwm_vecs[i]]()
                                    {
                                      ctrl->set_pwm(pwm);
                                    }));
  }

  for (auto &f : futures)
    f.get();
}

void set_motor_control_mode(
    const std::vector<std::shared_ptr<DynamixelControl>> &ctrls,
    const std::vector<std::vector<std::string>> &mode_vecs)
{
  if (mode_vecs.size() != ctrls.size())
  {
    throw std::invalid_argument("mode_vecs must match number of controllers");
  }

  for (size_t i = 0; i < ctrls.size(); ++i)
  {
    if (!ctrls[i])
      continue;
    ctrls[i]->set_control_mode(mode_vecs[i]);
  }
}

void set_motor_pd(const std::vector<std::shared_ptr<DynamixelControl>> &ctrls,
                  const std::vector<std::vector<float>> &kp_vecs,
                  const std::vector<std::vector<float>> &kd_vecs)
{
  if (kp_vecs.size() != ctrls.size() || kd_vecs.size() != ctrls.size())
    throw std::invalid_argument("kp/kd vectors must match number of controllers");

  for (size_t i = 0; i < ctrls.size(); ++i)
  {
    if (!ctrls[i])
      continue;
    if (kp_vecs[i].size() != kd_vecs[i].size())
      throw std::invalid_argument("kp/kd vector size mismatch for controller");
    ctrls[i]->set_pd(kp_vecs[i], kd_vecs[i]);
  }
}

void disable_motors(const std::vector<std::shared_ptr<DynamixelControl>> &ctrls,
                    const std::vector<std::vector<int>> &ids_by_controller = {})
{
  if (!ids_by_controller.empty() && ids_by_controller.size() != ctrls.size())
  {
    throw std::invalid_argument("ids_by_controller must match number of controllers");
  }

  for (size_t i = 0; i < ctrls.size(); ++i)
  {
    if (!ids_by_controller.empty())
    {
      ctrls[i]->disable_motors(ids_by_controller[i]);
    }
    else
    {
      ctrls[i]->disable_motors();
    }
  }
}

void enable_motors(const std::vector<std::shared_ptr<DynamixelControl>> &ctrls,
                   const std::vector<std::vector<int>> &ids_by_controller = {})
{
  if (!ids_by_controller.empty() && ids_by_controller.size() != ctrls.size())
  {
    throw std::invalid_argument("ids_by_controller must match number of controllers");
  }

  for (size_t i = 0; i < ctrls.size(); ++i)
  {
    if (!ids_by_controller.empty())
    {
      ctrls[i]->enable_motors(ids_by_controller[i]);
    }
    else
    {
      ctrls[i]->enable_motors();
    }
  }
}

void close_motors(const std::vector<std::shared_ptr<DynamixelControl>> &ctrls)
{
  for (const auto &ctrl : ctrls)
  {
    ctrl->close_motors();
  }
}

PYBIND11_MODULE(dynamixel_cpp, m)
{
  py::class_<DynamixelControl, std::shared_ptr<DynamixelControl>>(m, "DynamixelControl");
  m.def("create_controllers", &create_controllers, py::arg("port_pattern"),
        py::arg("kp"), py::arg("kd"), py::arg("ki"), py::arg("zero_pos"), py::arg("control_mode"),
        py::arg("baudrate"), py::arg("return_delay"));
  m.def("scan_port", &scan_port,
        py::arg("port_name"),
        py::arg("baudrate"),
        py::arg("protocol_version"),
        py::arg("max_motor_id"),
        py::arg("retries"));
  m.def("initialize",
        &initialize_motors,
        py::arg("controllers"),
        "Connect to client and initialize motors on each controller");
  m.def("get_motor_states", &get_motor_states, py::arg("controllers"), py::arg("retries"),
        py::call_guard<py::gil_scoped_release>(),
        "Get state of all motors across all controllers");
  m.def("get_motor_current_limits", &get_motor_current_limits, py::arg("controllers"),
        py::arg("retries") = 0, py::call_guard<py::gil_scoped_release>(),
        "Get current limit for each motor across all controllers");
  m.def("get_motor_ids", &get_motor_ids, py::arg("controllers"),
        "Get motor IDs for each controller");
  m.def("set_motor_pos", &set_motor_pos, py::arg("controllers"), py::arg("pos_vecs"),
        py::call_guard<py::gil_scoped_release>(),
        "Set position for each controller's motors");
  m.def("set_motor_vel", &set_motor_vel, py::arg("controllers"), py::arg("vel_vecs"),
        py::call_guard<py::gil_scoped_release>(),
        "Set velocity for each controller's motors");
  m.def("set_motor_cur", &set_motor_cur, py::arg("controllers"), py::arg("cur_vecs"),
        py::call_guard<py::gil_scoped_release>(),
        "Set current for each controller's motors");
  m.def("set_motor_pwm", &set_motor_pwm, py::arg("controllers"), py::arg("pwm_vecs"),
        py::call_guard<py::gil_scoped_release>(),
        "Set PWM for each controller's motors");
  m.def("set_motor_control_mode", &set_motor_control_mode,
        py::arg("controllers"), py::arg("mode_vecs"),
        "Set operating mode for each controller's motors");
  m.def("set_motor_pd", &set_motor_pd, py::arg("controllers"), py::arg("kp_vecs"), py::arg("kd_vecs"),
        py::call_guard<py::gil_scoped_release>(),
        "Set per-motor PD gains for each controller");
  m.def("disable_motors", &disable_motors, py::arg("controllers"),
        py::arg("ids_by_controller") = std::vector<std::vector<int>>{},
        "Disable torque on all motors across all controllers");
  m.def("enable_motors", &enable_motors, py::arg("controllers"),
        py::arg("ids_by_controller") = std::vector<std::vector<int>>{},
        "Enable torque on all motors across all controllers");
  m.def("close", &close_motors, py::arg("controllers"),
        "Disable torque and disconnect all specified DynamixelControl instances");
}
