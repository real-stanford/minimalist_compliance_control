#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include "dynamixel_control.h"

// Add declarations for standalone functions if they’re in a different file
std::vector<std::tuple<double, std::vector<float>, std::vector<float>, std::vector<float>>>
get_motor_states(const std::vector<DynamixelControl *> &controllers, int retries);

void set_motor_pos(const std::vector<DynamixelControl *> &controllers,
                   const std::vector<std::vector<float>> &pos_vecs);

int main()
{
  std::vector<std::string> ports = {"/dev/ttyCH9344USB2", "/dev/ttyCH9344USB3",
                                    "/dev/ttyCH9344USB4", "/dev/ttyCH9344USB5"};
  const int baudrate = 2000000;
  std::vector<std::vector<int>> chains = {{2}, {3}, {4}, {5}};

  std::vector<DynamixelControl *> controllers;
  for (size_t i = 0; i < ports.size(); ++i)
  {
    size_t N = chains[i].size();
    std::vector<float> kp(N, 0.0f);
    std::vector<float> kd(N, 0.0f);
    std::vector<float> ki(N, 0.0f);
    std::vector<float> zero_pos(N, 0.0f);
    std::vector<int> control_mode(N, 3);
    int return_delay = 0;

    auto *ctrl = new DynamixelControl(
        ports[i], chains[i],
        kp, kd, ki, zero_pos, control_mode,
        baudrate, return_delay);

    if (i > 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ctrl->connectToClient();
    ctrl->initializeMotors();
    controllers.push_back(ctrl);
  }

  bool toggle = false;
  auto last_switch = std::chrono::steady_clock::now();
  const std::chrono::seconds switch_interval(1);
  float pos_values[2] = {1.0f, 1.5f};

  double total_elapsed_ms = 0.0;
  int loop_count = 100;
  for (int i = 0; i < loop_count; ++i)
  {
    try
    {
      auto t0 = std::chrono::steady_clock::now();

      auto states = get_motor_state(controllers, 3);

      auto t1 = std::chrono::steady_clock::now();
      double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      total_elapsed_ms += elapsed_ms;

      for (size_t i = 0; i < chains.size(); ++i)
      {
        const auto &[timestamp, pos, vel, cur] = states[i];
        std::cout << "Chain " << i << " [t=" << elapsed_ms << "]";
        for (size_t j = 0; j < pos.size(); ++j)
        {
          std::cout << " id=" << chains[i][j]
                    << " pos=" << pos[j]
                    << " vel=" << vel[j]
                    << " cur=" << cur[j];
        }
        std::cout << std::endl;
      }

      // Optionally toggle motor positions
      auto now = std::chrono::steady_clock::now();
      if (now - last_switch >= switch_interval)
      {
        toggle = !toggle;
        std::vector<std::vector<float>> pos_vecs;
        pos_vecs.reserve(controllers.size());
        for (size_t i = 0; i < controllers.size(); ++i)
        {
          size_t N = chains[i].size();
          std::vector<float> vec(N, pos_values[toggle]);
          pos_vecs.push_back(vec);
        }

        set_motor_pos(controllers, pos_vecs);
        last_switch = now;
      }
    }
    catch (const std::exception &e)
    {
      std::cerr << "[test_dynamixel] Exception: " << e.what() << std::endl;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  std::cout << "Average elapsed time per loop: " << (total_elapsed_ms / loop_count) << " ms" << std::endl;

  for (auto *ctrl : controllers)
    delete ctrl;

  return 0;
}
